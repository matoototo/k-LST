from transformers import Trainer, DataCollatorWithPadding
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments
from config import Config
from update_policy import UpdatePolicyCallback, TrainerCallback

import torch
from torch.profiler import profile, record_function


class ProfilingCallback(TrainerCallback):
        def __init__(self):
            self.profiler = None

        def on_train_begin(self, args, state, control, model=None, **kwargs):
            self.profiler = profile(activities=[
                torch.profiler.ProfilerActivity.CPU, 
                torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True, 
                profile_memory=True, 
                with_stack=True)

        def on_step_begin(self, args, state, control, **kwargs):
            self.profiler.__enter__()
            self.record_function = record_function("model_inference")
            self.record_function.__enter__()

        def on_step_end(self, args, state, control, **kwargs):
            self.record_function.__exit__(None, None, None)
            self.profiler.__exit__(None, None, None)

        def on_train_end(self, args, state, control, **kwargs):
            self.profiler.export_chrome_trace("trace.json")


def train(config: Config):
    # ========= MODEL ========= #
    # Load model and apply freezing and adapter strategies
    model = config.load_model()
    config.freeze_model(model)
    model = config.add_adapters(model)
    # ========= DATA ========= #
    dataset = config.load_dataset()

    # Tokenize the dataset with our tokenization function
    tokenized_dataset, tokenizer = config.tokenize_dataset(dataset, model)
    # Data collator for dynamic padding. Tokenizer itself does not pad.
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # ========= TRAINING ========= #
    training_args = config.load_training_args()

    train_dataset = tokenized_dataset["train"]
    eval_dataset = tokenized_dataset["validation"]

    # function called by trainer during trainer.evaluate()
    metric_function = config.load_metric_function()

    # get optimizer & scheduler
    optimizer = config.load_optimizer(model, train_dataset)

    trainer = Trainer(
        model,
        training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=metric_function,
        callbacks=[UpdatePolicyCallback(model), ProfilingCallback()],
        optimizers=optimizer
    )

    
    
    # Perform validation before training
    print("Evaluating before training (epoch 0)...")
    metrics = trainer.evaluate()
    print(metrics)

    trainer.train()


if __name__ == "__main__":
    import argparse, pathlib

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=pathlib.Path, help="Path to config file", required=True)
    args = parser.parse_args()

    config = Config(args.config)

    train(config)
