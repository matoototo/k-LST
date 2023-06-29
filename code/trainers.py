import torch
import torch.nn as nn
from transformers import Trainer
from typing import Union, Dict, Any


class MezoTrainer(Trainer):
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
