from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import TransformerEncoderLayer
from collections import OrderedDict
from transformers.modeling_outputs import SequenceClassifierOutput


def ladder_side_tuning(model, **config):
    """Adds LST wrapper around module.
    :return: nn.Module
    """
    return LST(model, config)

def ladder_side_distillation(model, **config):
    """Performs distillation with LST structure.
    :return: nn.Module
    """
    return LSTDistillation(model, config)

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
    else:
        raise ValueError("Model does not have d_model_ff attribute, check model.config")

def is_block(model, module_name):
    is_bert = "bert" in model.config._name_or_path
    module_name = module_name.split(".")
    if len(module_name) < 2: return False
    if is_bert: # Ends with layer.{i}
        return module_name[-2] == "layer" and module_name[-1].isdigit()
    else:
        raise ValueError("Only BERT models are supported")


class LST(nn.Module):
    def __init__(self, model, config):
        super().__init__()
        self.__dict__["model"] = model
        self.is_roberta = "roberta" in model.config._name_or_path
        self.lst_config = config

        self.k = self.lst_config["k"] if "k" in self.lst_config else 1
        self.dropout_p = self.lst_config["dropout"]
        if "downsample" not in self.lst_config: self.lst_config["downsample"] = "linear"

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

    def side_downsampled(self, i):
        backbone_output = self.intermediate_activations[f"backbone_{i}"][0]
        downsampled_backbone = self.side_modules[f"side_downsample_{i}"](backbone_output)
        return downsampled_backbone

    def get_backbone_outputs(self, middle):
        if self.k % 2 == 0:
            raise RuntimeError("k should be odd for now")
        n = self._n_outputs
        _start = middle - (self.k - 1) // 2
        _end = middle + (self.k - 1) // 2
        start = max(_start, 0)
        end = min(_end, n - 1) + 1
        
        outputs = [self.side_downsampled(i) for i in range(start, end)]

        if _start < 0:
            start_padding = [torch.zeros_like(outputs[0]) for i in range(abs(_start))]
            outputs = start_padding + outputs
        
        if _end > n - 1:
            end_padding = [torch.zeros_like(outputs[0]) for i in range(_end - n + 1)]
            outputs = outputs + end_padding

        if self.training:
            length = len(outputs)
            mask = (torch.rand(length) >= self.dropout_p).float()
            mask = mask.to(outputs[0].device)
            for i in range(length):
                outputs[i] *= mask[i]
    
        return outputs

    def combine_backbone_feats(self, backbone_feats):
        backbone_feats = torch.stack(backbone_feats)
        if self.k > 1:
            backbone_feats += self.positional_embedding.weight.unsqueeze(1).unsqueeze(1)       
        a, b, c, d = backbone_feats.shape
        combined = backbone_feats.permute(1, 0, 2, 3).contiguous().view(b, c * a, d)
        return combined

    def encoder(self, input):
        n = self._n_outputs
        output = self.side_modules["initial_downsample"](input) # [16]
        for i in range(n):
            downsampled_backbones = self.get_backbone_outputs(i)
            downsampled_backbone = self.combine_backbone_feats(downsampled_backbones)
            output = self.fuse(downsampled_backbone, output, i)
            output = self.side_modules[f"ladder_block_{i}"](output)
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
        _ = self.model(input_ids, attention_mask) # Just to get the intermediate activations
        input = self.intermediate_activations["embeddings"]
        output = self.encoder(input)
        output = self.side_modules["side_upsample"](output)
        output = self.model_head(output)

        self.intermediate_activations = OrderedDict()

        # roberta is already returning 1 token pooled token for classification
        if not self.is_roberta:
            output = output[:, 0, :]  # CLS token
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(output, labels)
        if labels is not None:
            return SequenceClassifierOutput(loss=loss, logits=output)
        return SequenceClassifierOutput(logits=output)

    def _get_model_head(self, model, freeze = True):
        if hasattr(model, "qa_outputs"):
            head = model.qa_outputs
        elif hasattr(model, "classifier"):
            head = model.classifier
        else:
            raise AttributeError("Model does not have a QA head or classifier.")

        for param in head.parameters():
            param.requires_grad = False if freeze else True

        return head

    def _create_side_modules(self, n):
        side_modules = OrderedDict()
        side_modules["initial_downsample"] = nn.Linear(self._d_model, self.d_side)
        for i in range(n):
            if self.lst_config["downsample"] == "adaptive_avg_pool1d":
                side_modules[f"side_downsample_{i}"] = nn.AdaptiveAvgPool1d(self.d_side)
            else:
                side_modules[f"side_downsample_{i}"] = nn.Linear(self._d_model, self.d_side)
            block_type = TransformerEncoderLayer
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
            elif name.split(".")[-1] == "embeddings" or name.split(".")[-1] == "shared": # BERT
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


class LSTDistillation(LST):
    def __init__(self, model, config):
        super().__init__(model, config)
        assert config["downsample"] == "adaptive_avg_pool1d", "Can't have learned downsample for distillation"
        self.distillation_weight = config["distillation_weight"]

    def encoder(self, input):
        n = self._n_outputs
        output = self.side_modules["initial_downsample"](input) # [16]
        losses = []
        loss_fct = nn.MSELoss()
        for i in range(n):
            backbone_output = self.intermediate_activations[f"backbone_{i}"][0]
            downsampled_backbone = self.side_modules[f"side_downsample_{i}"](backbone_output)
            losses.append(loss_fct(output, downsampled_backbone))
            output = self.side_modules[f"ladder_block_{i}"](output)
        return output, sum(losses)

    def forward(self, input_ids, attention_mask, *, labels=None, **kwargs):
        _ = self.model(input_ids, attention_mask, **kwargs) # Just to get the intermediate activations
        input = self.intermediate_activations["embeddings"]
        output, distillation_loss = self.encoder(input)
        output = self.side_modules["side_upsample"](output)
        output = self.model_head(output)

        self.intermediate_activations = OrderedDict()

        output = output[:, 0, :]  # CLS token
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(output, labels) + self.distillation_weight * distillation_loss
        if labels is not None:
            return SequenceClassifierOutput(loss=loss, logits=output)
        return SequenceClassifierOutput(logits=output)
