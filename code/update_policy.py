import torch.cuda
from torch.nn.modules import Module
from transformers import TrainingArguments, TrainerState, TrainerControl
from transformers.integrations import TensorBoardCallback
from freeze_strategies import all_but_last_n


class UpdatePolicyCallback(TensorBoardCallback):
    def __init__(self, model: Module):
        super().__init__()
        self.model = model

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # An update policy can be applied here
        if state.epoch > 1:
            all_but_last_n(self.model)
        elif state.epoch > 0:
            all_but_last_n(self.model, 3)
        super().on_epoch_begin(args, state, control, **kwargs)

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        logs["memory_allocated"] = torch.cuda.memory_allocated()
        info = torch.cuda.mem_get_info()
        logs["memory_free"] = info[0]
        logs["memory_occupied"] = info[1]
        stats = torch.cuda.memory_stats()
        logs["memory_active_current"] = stats["active_bytes.all.current"]
        logs["memory_active_allocated"] = stats["active_bytes.all.allocated"]
        logs["memory_allocated_peak"] = stats["allocated_bytes.all.peak"]
        torch.cuda.reset_peak_memory_stats()
        super().on_log(args, state, control, logs=logs, **kwargs)
