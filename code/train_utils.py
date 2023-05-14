from functools import partial

import datasets as huggingface_datasets
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForSequenceClassification, \
    DataCollatorWithPadding, AdamW, get_scheduler
from tqdm.auto import tqdm
from update_policy import UpdatePolicy
import evaluate
import numpy as np
from torch.utils.data import DataLoader


def load_dataset(dataset="sst2", split=None):
    dataset = huggingface_datasets.load_dataset(dataset, split=split)
    return dataset


def tokenize(dataset, tokenizer, max_length):
    # If we need to truncate, truncate the context instead of the question
    tokenized_inputs = tokenizer(
        dataset["sentence"],
        max_length=max_length,
        truncation=True
    )

    return tokenized_inputs


def freeze(model):
    # Set requires_grad to False for all parameters
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_last(model, unfreeze_n=1):
    # Unfreeze the last n layers
    if "distilbert" in model.name_or_path:
        # DistilBERT
        for param in model.distilbert.transformer.layer[-unfreeze_n:].parameters():
            param.requires_grad = True
    elif "bert" in model.name_or_path:
        # BERT
        for param in model.bert.encoder.layer[-unfreeze_n:].parameters():
            param.requires_grad = True

    # QA head
    for param in model.qa_outputs.parameters():
        param.requires_grad = True


def train(n_train=None, n_val=None, model_name="distilbert-base-cased"):
    # ========= MODEL ========= #
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # ========= DATA ========= #
    dataset = load_dataset()

    # Take subset of dataset if specified
    if n_train:
        dataset["train"] = dataset["train"].select(range(n_train))
    if n_val:
        dataset["validation"] = dataset["validation"].select(range(n_val))

    # Tokenize the dataset with our tokenization function
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    max_length = model.config.max_position_embeddings
    tokenize_partial = partial(tokenize, tokenizer=tokenizer, max_length=max_length)
    # Keep the label column in the tokenized dataset, remove the rest
    tokenized_dataset = dataset.map(tokenize_partial, batched=True, remove_columns=["sentence", "idx"])
    # Data collator for dynamic padding. Tokenizer itself does not pad.
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # ========= TRAINING ========= #
    training_args = TrainingArguments(
        output_dir="results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01
    )

    train_dataset = tokenized_dataset["train"]
    eval_dataset = tokenized_dataset["validation"]

    # compute_metrics function called by trainer during trainer.evaluate()
    def compute_metrics(eval_preds):
        metric = evaluate.load("accuracy")
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    trainer = Trainer(
        model,
        training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
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
    train_dataloader = DataLoader(
        tokenized_dataset["train"], shuffle=True, batch_size=training_args.per_device_train_batch_size,
        collate_fn=trainer.data_collator
    )
    optimizer = AdamW(model.parameters(), lr=training_args.learning_rate)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_steps,
    )

    # Perform training
    for epoch in range(int(training_args.num_train_epochs)):
        # Apply the update policy before each epoch
        # update_policy.apply(epoch, metrics)

        for batch in train_dataloader:
            trainer.training_step(model, batch)
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
            progress_bar.update(1)

        # Evaluate after each epoch
        print(f"Evaluating epoch {epoch + 1}...")
        metrics = trainer.evaluate()
        print(metrics)


if __name__ == "__main__":
    train(model_name="distilbert-base-cased")
