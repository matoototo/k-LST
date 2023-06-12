import torch
from transformers import Trainer
from typing import Union, Dict, Any
import torch.nn as nn

from freeze_strategies import freeze_all, all_but_last_n


class MezoTrainer(Trainer):
    def __init__(self, *arg, epsilon=1e-3, **kwargs):
        self.epsilon = epsilon
        self.seed = 0
        self.generator = torch.Generator(device='cuda')
        super().__init__(*arg, **kwargs)
        freeze_all(self.model)

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            self.seed = self.generator.seed()
            self.perturb_parameters(model, self.epsilon)
            with self.compute_loss_context_manager():
                l_plus = self.compute_loss(model, inputs).detach()
            self.perturb_parameters(model, -2 * self.epsilon)
            with self.compute_loss_context_manager():
                l_minus = self.compute_loss(model, inputs).detach()
            self.perturb_parameters(model, self.epsilon)

        projected_grad = (l_plus - l_minus) / (2 * self.epsilon)
        self.generator.manual_seed(self.seed)
        for parameter in model.parameters():
            z = torch.normal(0, 1, parameter.data.size(), generator=self.generator, device='cuda')
            # setting the gradient and letting optimizer.step() inside _inner_training_loop
            # gives similar results to updating parameter.data directly
            # parameter.grad = projected_grad * z
            parameter.data = parameter.data - self.lr_scheduler.get_last_lr()[0] * projected_grad * z

        return l_plus

    def perturb_parameters(self, model, scale):
        self.generator.manual_seed(self.seed)
        for parameter in model.parameters():
            z = torch.normal(0, 1, parameter.data.size(), generator=self.generator, device='cuda')
            parameter.data = parameter.data + scale * z
