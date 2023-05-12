from transformers import DefaultDataCollator, Trainer
import torch
from tqdm.auto import tqdm
from update_policy import UpdatePolicy

from config import Config

def train(config : Config):
    # ========= MODEL ========= #
    # Load model and apply freezing strategy
    model = config.load_model()
    config.freeze_model(model)

    # ========= DATA ========= #
    dataset = config.load_dataset()
    data_collator = DefaultDataCollator()

    # Tokenize the dataset with our tokenization function
    tokenized_dataset, tokenizer = config.tokenize_dataset(dataset, model)

    # ========= TRAINING ========= #
    training_args = config.load_training_args()

    train_dataset = tokenized_dataset["train"]
    eval_dataset = tokenized_dataset["validation"]

    trainer = Trainer(
        model,
        training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    # Perform validation before training
    print("Evaluating before training (epoch 0)...")
    metrics = trainer.evaluate()
    print(metrics)

    # Perform initialization and create update policy before training
    update_policy = UpdatePolicy(model)
    num_steps = int(training_args.num_train_epochs * train_dataset.num_rows / training_args.per_device_train_batch_size)
    trainer.create_optimizer_and_scheduler(num_steps)
    progress_bar = tqdm(range(num_steps))

    # Perform training
    for epoch in range(int(training_args.num_train_epochs)):
        # Apply the update policy before each epoch
        update_policy.apply(epoch, metrics)

        for batch in train_dataset.iter(training_args.per_device_train_batch_size):
            batch = {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}
            trainer.training_step(model, batch)
            trainer.optimizer.step()
            trainer.optimizer.zero_grad()
            trainer.lr_scheduler.step()
            progress_bar.update(1)

        # Evaluate after each epoch
        print(f"Evaluating epoch {epoch + 1}...")
        metrics = trainer.evaluate()
        print(metrics)


if __name__ == "__main__":
    import argparse, pathlib

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=pathlib.Path, help="Path to config file", required=True)
    args = parser.parse_args()

    config = Config(args.config)

    train(config)
