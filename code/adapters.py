from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import TransformerEncoderLayer, TransformerDecoderLayer
from collections import OrderedDict
from transformers.modeling_outputs import SequenceClassifierOutput


def ladder_side_tuning(model, **config):
    """Adds LST wrapper around module.
    :return: nn.Module
    """
    return LST(model, config)

def get_d_model(model):
    if hasattr(model.config, "dim"):
        return model.config.dim
    elif hasattr(model.config, "hidden_size"):
        return model.config.hidden_size
    elif hasattr(model.config, "d_model"):
        return model.config.d_model
    else:
        raise ValueError("Model does not have d_model attribute, check model.config")
    
def get_d_model_ff(model):
    if hasattr(model.config, "hidden_dim"): # BERT
        return model.config.hidden_dim
    elif hasattr(model.config, "intermediate_size"):
        return model.config.intermediate_size
    elif hasattr(model.config, "d_ff"): # T5
        return model.config.d_ff
    else:
        raise ValueError("Model does not have d_model_ff attribute, check model.config")

def is_block(model, module_name):
    is_t5 = "t5" in model.config._name_or_path
    is_bert = "bert" in model.config._name_or_path
    module_name = module_name.split(".")
    if len(module_name) < 2: return False
    if is_t5: # Ends with block.{i}
        return module_name[-2] == "block" and module_name[-1].isdigit()
    elif is_bert: # Ends with layer.{i}
        return module_name[-2] == "layer" and module_name[-1].isdigit()
    else:
        raise ValueError("Only T5 and BERT models are supported")

class LST(nn.Module):
    def __init__(self, model, config):
        super().__init__()
        self.__dict__["model"] = model
        self.is_t5 = "t5" in model.config._name_or_path
        self.lst_config = config

        self.k = self.lst_config["k"] if "k" in self.lst_config else 1

        assert self.k == 1 or self.lst_config["fusion"] == "attention"

        self._d_model = get_d_model(self.model)
        self._d_model_ff = get_d_model_ff(self.model)
        self.d_side = self._d_model // self.lst_config["reduction_factor"]

        if self.k > 1:
            self.positional_embedding = nn.Embedding(self.k, self.d_side)

        self.d_side_ff = self._d_model_ff // self.lst_config["reduction_factor"]

        self.intermediate_activations = {}
        self._n_outputs = self._register_hooks()

        self.side_modules = nn.ParameterDict(self._create_side_modules(self._n_outputs))
        self.model_head = self._get_model_head(self.model, self.lst_config["freeze_head"])
        self.model.to("cuda:0" if torch.cuda.is_available() else "cpu")

    def fuse(self, downsampled_backbone, output, i):
        if self.lst_config["fusion"] == "additive":
            output = output + downsampled_backbone
        elif self.lst_config["fusion"] == "gated":
            fuse = torch.sigmoid(self.side_modules[f"fuse_{i}"])
            output = fuse * output + (1 - fuse) * downsampled_backbone
        elif self.lst_config["fusion"] == "dynamic":
            pooled = F.adaptive_avg_pool1d(downsampled_backbone.permute(0, 2, 1), 1).squeeze(-1)
            fuse = self.side_modules[f"fuse_{i}"](pooled).sigmoid().unsqueeze(-1)
            output = fuse * output + (1 - fuse) * downsampled_backbone
        elif self.lst_config["fusion"] == "attention":
            output = self.side_modules[f"fuse_{i}"](output, downsampled_backbone, downsampled_backbone)[0]
        else:
            raise ValueError("Invalid fusion strategy, must be one of 'additive', 'gated', 'attention' or 'dynamic'")
        return output

    def get_backbone_outputs(self, middle):
        if self.k % 2 == 0:
            raise RuntimeError("k should be odd for now")
        n = self._n_outputs if not self.is_t5 else self._n_outputs // 2
        _start = middle - (self.k - 1) // 2
        _end = middle + (self.k - 1) // 2
        start = max(_start, 0)
        end = min(_end, n - 1) + 1
        outputs = [self.intermediate_activations[f"backbone_{i}"][0] for i in range(start, end)]

        if _start < 0:
            start_padding = [torch.zeros_like(outputs[0]) for i in range(abs(_start))]
            outputs = start_padding + outputs
        
        if _end > n - 1:
            end_padding = [torch.zeros_like(outputs[0]) for i in range(_end - n + 1)]
            outputs = outputs + end_padding

        return outputs

    def combine_backbone_feats(self, backbone_feats):
        backbone_feats = torch.stack(backbone_feats)
        if self.k > 1:
            backbone_feats += self.positional_embedding.weight.unsqueeze(1).unsqueeze(1)       
        a, b, c, d = backbone_feats.shape
        combined = backbone_feats.permute(1, 0, 2, 3).contiguous().view(b, c * a, d)
        return combined

    def encoder(self, input):
        n = self._n_outputs if not self.is_t5 else self._n_outputs // 2
        output = self.side_modules["initial_downsample"](input) # [16]
        for i in range(n):
            backbone_outputs = self.get_backbone_outputs(i)
            downsampled_backbones = [self.side_modules[f"side_downsample_{i}"](bo) for bo in backbone_outputs]
            downsampled_backbone = self.combine_backbone_feats(downsampled_backbones)
            output = self.fuse(downsampled_backbone, output, i)
            output = self.side_modules[f"ladder_block_{i}"](output)
        return output
    
    def decoder(self, input, encoder_out):
        offset = self._n_outputs // 2 if self.is_t5 else 0
        output = self.side_modules["initial_downsample_dec"](input)
        for i in range(offset, self._n_outputs):
            backbone_output = self.intermediate_activations[f"backbone_{i}"][0]
            downsampled_backbone = self.side_modules[f"side_downsample_{i}"](backbone_output)
            output = self.fuse(downsampled_backbone, output, i)
            output = self.side_modules[f"ladder_block_{i}"](output, encoder_out)
        return output

    def forward(self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,):
        if self.is_t5:
            _ = self.model(input_ids = input_ids, attention_mask = attention_mask, decoder_input_ids = decoder_input_ids, decoder_attention_mask = decoder_attention_mask, head_mask = head_mask, decoder_head_mask = decoder_head_mask, cross_attn_head_mask = cross_attn_head_mask, encoder_outputs = encoder_outputs, past_key_values = past_key_values, inputs_embeds = inputs_embeds, decoder_inputs_embeds = decoder_inputs_embeds, labels = labels, use_cache = use_cache, output_attentions = output_attentions, output_hidden_states = output_hidden_states, return_dict = return_dict)
        else:
            _ = self.model(input_ids, attention_mask) # Just to get the intermediate activations
        input = self.intermediate_activations["embeddings"]
        output = self.encoder(input)
        if self.is_t5:
            dec_input = self.intermediate_activations["embeddings_dec"]
            masked_enc_out = output * attention_mask.unsqueeze(-1)
            output = self.decoder(dec_input, masked_enc_out)
        output = self.side_modules["side_upsample"](output)
        output = self.model_head(output)

        self.intermediate_activations = OrderedDict()

        if not self.is_t5:
            output = output[:, 0, :]  # CLS token
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(output, labels)
            if labels is not None:
                return SequenceClassifierOutput(loss=loss, logits=output)
            return SequenceClassifierOutput(logits=output)
        else:
            loss = None
            lm_logits = output
            if labels is not None:
                loss_fct = lambda x, y: F.cross_entropy(x, y, ignore_index=-100)
                labels = labels.to(lm_logits.device)
                _lm_logits = lm_logits[:, 0, :]
                labels = labels[:, 0]
                loss = loss_fct(_lm_logits, labels)
            output = lm_logits
            return ((loss,) + ([output],)) if loss is not None else output

    def _get_model_head(self, model, freeze = True):
        if self.is_t5:
            head = model.lm_head
        elif hasattr(model, "qa_outputs"):
            head = model.qa_outputs
        elif hasattr(model, "classifier"):
            head = model.classifier
        else:
            raise AttributeError("Model does not have a QA head or classifier, nor is it a T5 model")

        for param in head.parameters():
            param.requires_grad = False if freeze else True

        return head

    def _create_side_modules(self, n):
        side_modules = OrderedDict()
        side_modules["initial_downsample"] = nn.Linear(self._d_model, self.d_side)
        if self.is_t5:
            side_modules["initial_downsample_dec"] = nn.Linear(self._d_model, self.d_side)
        for i in range(n):
            side_modules[f"side_downsample_{i}"] = nn.Linear(self._d_model, self.d_side)
            block_type = TransformerEncoderLayer if i < n // 2 or (not self.is_t5) else TransformerDecoderLayer
            side_modules[f"ladder_block_{i}"] = block_type(self.d_side, 4, self.d_side_ff, batch_first=True)
            if self.lst_config["fusion"] == "dynamic":
                side_modules[f"fuse_{i}"] = nn.Linear(self.d_side, 1)
            elif self.lst_config["fusion"] == "gated":
                side_modules[f"fuse_{i}"] = nn.Parameter(torch.zeros(1))
            elif self.lst_config["fusion"] == "attention":
                side_modules[f"fuse_{i}"] = nn.MultiheadAttention(self.d_side, 1, batch_first=True)
        side_modules["side_upsample"] = nn.Linear(self.d_side, self._d_model)
        return side_modules

    def _register_hooks(self):
        n = 0
        for name, module in self.model.named_modules():
            # Assuming that the blocks end with a number, ex. "distilbert.transformer.layer.5"
            if is_block(self.model, name):
                module.register_forward_hook(self._hook(f"backbone_{n}"))
                n += 1
            elif name.split(".")[-1] == "embeddings" or name.split(".")[-1] == "shared": # BERT, T5
                module.register_forward_hook(self._hook("embeddings"))
        return n
    
    def _hook(self, name):
        # Closure
        def hook(module, input, output):
            if name in self.intermediate_activations:
                self.intermediate_activations[f"{name}_dec"] = output
            else:
                self.intermediate_activations[name] = output
        return hook

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == "model":
                raise AttributeError()
            return getattr(self.model, name)

