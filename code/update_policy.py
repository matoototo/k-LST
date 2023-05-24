from torch.nn.modules import Module
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl


class UpdatePolicyCallback(TrainerCallback):
    def __init__(self, model: Module):
        self.model = model

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # An update policy can be applied here
        super().on_epoch_begin(args, state, control, **kwargs)
