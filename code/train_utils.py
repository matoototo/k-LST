import transformers
from transformers import DataCollatorWithPadding, DataCollatorForTokenClassification, EarlyStoppingCallback
from config import Config
from update_policy import UpdatePolicyCallback


def train(config: Config, resume_from_checkpoint, model_path):
    # ========= MODEL ========= #
    # Load model and apply freezing and adapter strategies
    model = config.load_model(model_path)
    config.freeze_model(model)
    model = config.add_adapters(model)
    # ========= DATA ========= #
    dataset = config.load_dataset()

    # Tokenize the dataset with our tokenization function
    tokenized_dataset, tokenizer = config.tokenize_dataset(dataset, model)
    # Data collator for dynamic padding. Tokenizer itself does not pad.
    if "labels" in tokenized_dataset["train"][0] and len(tokenized_dataset["train"][0]["labels"]) > 1:
        # This pads labels as well.
        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    else:
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # ========= TRAINING ========= #
    training_args = config.load_training_args()

    train_dataset = tokenized_dataset["train"]
    eval_dataset = tokenized_dataset["validation"]

    # functions called by trainer during trainer.evaluate()
    metric_function, preprocess_logits_function = config.load_metric_function(tokenizer)

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
        preprocess_logits_for_metrics=preprocess_logits_function,
        callbacks=[UpdatePolicyCallback, EarlyStoppingCallback(early_stopping_patience=training_args.save_total_limit-1)],
        optimizers=optimizer
    )

    if not resume_from_checkpoint:
        # Perform validation before training
        print("Evaluating before training (epoch 0)...")
        metrics = trainer.evaluate()
        print(metrics)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)


if __name__ == "__main__":
    import argparse, pathlib
    transformers.set_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=pathlib.Path, help="Path to config file", required=True)
    parser.add_argument("--resume_from_checkpoint", help="When set to True, trainer resumes from latest checkpoint. "
                                                         "When set to a path to a checkpoint, trainer resumes from "
                                                         "the given checkpoint.", default=False)
    parser.add_argument("--model_path", help="Path to local checkpoint directory to initialize the model from.")
    args = parser.parse_args()

    config = Config(args.config)
    resume_from_checkpoint = args.resume_from_checkpoint
    if resume_from_checkpoint == "True":
        resume_from_checkpoint = True
    elif resume_from_checkpoint == "False":
        resume_from_checkpoint = False
    model_path = args.model_path

    train(config, resume_from_checkpoint, model_path)
