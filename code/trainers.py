import torch
import torch.nn as nn
from transformers import Trainer
from torch.nn.functional import binary_cross_entropy_with_logits
from typing import Union, Dict, Any


class PromptTrainer(Trainer):
    def __init__(self, *arg, neg_label="terrible", pos_label="great", **kwargs):
        super().__init__(*arg, **kwargs)
        self.label_encodings = [self.tokenizer.encode(label)[1] for label in [neg_label, pos_label]]

    def compute_loss(self, model, inputs, return_outputs=False):
        # remove "labels" before passing inputs through model, then add it back
        labels = inputs["labels"]
        inputs = {k: inputs[k] for k in inputs if k != "labels"}
        outputs = model(**inputs)
        inputs = inputs | {"labels": labels}

        # for each example, get logits of the label words at the mask token
        mask_idx = torch.where(inputs["input_ids"] == self.tokenizer.mask_token_id)
        mask_logits = outputs.logits[mask_idx[0], mask_idx[1]]
        outputs.logits = mask_logits[:, self.label_encodings]

        # targets with the same shape as logits
        targets = torch.tensor([[0, 1] if label == 1 else [1, 0] for label in labels], device=self.model.device)

        loss = binary_cross_entropy_with_logits(outputs.logits, targets.float())
        return (loss, outputs) if return_outputs else loss


class MezoTrainer(PromptTrainer):
    def __init__(self, *arg, eps=1e-3, **kwargs):
        super().__init__(*arg, **kwargs)
        self.eps = eps
        self.seed = 0
        self.generator = torch.Generator(device=self.model.device)

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        inputs = self._prepare_inputs(inputs)

        # implements Algorithm 1 of MeZO (https://arxiv.org/abs/2305.17333)
        self.seed = self.generator.seed()
        self.perturb_parameters(self.eps)
        with self.compute_loss_context_manager():
            l_plus = self.compute_loss(model, inputs)
        self.perturb_parameters(-2 * self.eps)
        with self.compute_loss_context_manager():
            l_minus = self.compute_loss(model, inputs)
        self.perturb_parameters(self.eps)

        projected_grad = (l_plus - l_minus) / (2 * self.eps)
        self.generator.manual_seed(self.seed)
        for name, parameter in model.named_parameters():
            z = torch.normal(0, 1, parameter.data.size(), generator=self.generator, device=self.model.device)
            # noinspection PyArgumentList
            parameter -= self._get_learning_rate() * projected_grad * z

        return l_plus.detach()

    def perturb_parameters(self, scale):
        self.generator.manual_seed(self.seed)
        for name, parameter in self.model.named_parameters():
            z = torch.normal(0, 1, parameter.data.size(), generator=self.generator, device=self.model.device)
            parameter += scale * z