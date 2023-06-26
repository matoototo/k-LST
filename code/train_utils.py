from transformers import DataCollatorWithPadding
from config import Config
from update_policy import UpdatePolicyCallback


def train(config: Config, resume, checkpoint):
    # ========= MODEL ========= #
    # Load model and apply freezing and adapter strategies
    model = config.load_model(checkpoint)
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

    trainer = config.load_trainer(
        model,
        training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=metric_function,
        callbacks=[UpdatePolicyCallback],
        optimizers=optimizer
    )

    # Perform validation before training
    print("Evaluating before training (epoch 0)...")
    metrics = trainer.evaluate()
    print(metrics)

    trainer.train(resume_from_checkpoint=resume)


if __name__ == "__main__":
    import argparse, pathlib

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=pathlib.Path, help="Path to config file", required=True)
    parser.add_argument("--resume", help="When set, trainer resumes from latest checkpoint", action="store_true")
    parser.add_argument("--checkpoint", help="Path to checkpoint directory to load the model from")
    args = parser.parse_args()

    config = Config(args.config)
    resume = args.resume
    checkpoint = args.checkpoint

    train(config, resume, checkpoint)
