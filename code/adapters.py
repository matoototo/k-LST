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


class LST(nn.Module):
    def __init__(self, model, config):
        super().__init__()
        self.__dict__["model"] = model
        self.lst_config = config

        self._d_model = self.model.config.dim if hasattr(self.model.config, "dim") else self.model.config.hidden_size
        self._d_model_ff = self.model.config.hidden_dim if hasattr(self.model.config, "hidden_dim") else self.model.config.intermediate_size
        self.d_side = self._d_model // self.lst_config["reduction_factor"]
        self.d_side_ff = self._d_model_ff // self.lst_config["reduction_factor"]

        self.intermediate_activations = {}
        self._n_outputs = self._register_hooks()

        self.side_modules = nn.ParameterDict(self._create_side_modules(self._n_outputs))
        self.model_head = self._get_model_head(self.model, self.lst_config["freeze_head"])
        self.model.to("cuda:0" if torch.cuda.is_available() else "cpu")
                                         
    def forward(self, input_ids, attention_mask, labels = None, **kwargs):
        outputs = self.model(input_ids, attention_mask, **kwargs) # Just to get the intermediate activations
        # return self.model(*args, **kwargs)
        input = self.intermediate_activations["embeddings"]
        output = self.side_modules["initial_downsample"](input) # [16]
        for i in range(self._n_outputs):
            backbone_output = self.intermediate_activations[f"backbone_{i}"][0]
            downsampled_backbone = self.side_modules[f"side_downsample_{i}"](backbone_output)
            if self.lst_config["fusion"] == "additive":
                output = output + downsampled_backbone
            else:
                fuse = torch.sigmoid(self.side_modules[f"fuse_{i}"])
                output = fuse * output + (1 - fuse) * downsampled_backbone
            output = self.side_modules[f"ladder_block_{i}"](output)
        output = self.side_modules["side_upsample"](output)
        output = self.model_head(output)
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
            raise AttributeError("Model does not have a QA head or classifier")

        for param in head.parameters():
            param.requires_grad = False if freeze else True

        return head

    def _create_side_modules(self, n):
        side_modules = OrderedDict()
        side_modules["initial_downsample"] = nn.Linear(self._d_model, self.d_side)
        for i in range(n):
            side_modules[f"side_downsample_{i}"] = nn.Linear(self._d_model, self.d_side)
            side_modules[f"ladder_block_{i}"] = TransformerEncoderLayer(self.d_side, 4, self.d_side_ff)
            side_modules[f"fuse_{i}"] = nn.Parameter(torch.zeros(1))
        side_modules["side_upsample"] = nn.Linear(self.d_side, self._d_model)
        return side_modules

    def _register_hooks(self):
        n = 0
        for name, module in self.model.named_modules():
            # Assuming that the blocks end with a number, ex. "distilbert.transformer.layer.5"
            if name.split(".")[-1].isdigit():
                module.register_forward_hook(self._hook(f"backbone_{n}"))
                n += 1
            elif name.split(".")[-1] == "embeddings": # We need these!
                module.register_forward_hook(self._hook("embeddings"))
        return n
    
    def _hook(self, name):
        # Closure
        def hook(module, input, output):
            self.intermediate_activations[name] = output
        return hook

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == "model":
                raise AttributeError()
            return getattr(self.model, name)

