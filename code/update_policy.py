import os
from transformers import TrainingArguments, TrainerState, TrainerControl
from torch.cuda import memory_stats, reset_peak_memory_stats
from transformers.integrations import TensorBoardCallback
from shutil import rmtree


class UpdatePolicyCallback(TensorBoardCallback):

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        # Log peak allocated memory in MiB and reset it
        logs["memory"] = memory_stats()["allocated_bytes.all.peak"] / 1048576
        reset_peak_memory_stats()

        super().on_log(args, state, control, logs=logs, **kwargs)
    
    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):

        # Delete unnecessary tensorboard files and folders
        for root, dirs, files in os.walk(args.logging_dir):
            for d in dirs:
                rmtree(f"{args.logging_dir}/{d}")
            if len(files) > 0:
                files.pop()
                for file in files:
                    os.remove(f"{args.logging_dir}/{file}")

        super().on_train_end(args, state, control, **kwargs)
