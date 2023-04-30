from functools import partial

import datasets as huggingface_datasets
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, DefaultDataCollator, TrainingArguments, Trainer


def load_dataset(dataset = "squad", split = None):
    dataset = huggingface_datasets.load_dataset(dataset, split=split)
    return dataset


def tokenize(dataset, tokenizer, max_length):
    # If we need to truncate, truncate the context instead of the question
    tokenized_inputs = tokenizer(
        dataset["question"],
        dataset["context"],
        max_length=max_length,
        truncation="only_second",
        padding="max_length",
        return_offsets_mapping=True
    )

    offset_mapping = tokenized_inputs.pop("offset_mapping")
    start_positions = []
    end_positions = []

    # Update the start and end positions
    for i, offsets in enumerate(offset_mapping):
        answer = dataset["answers"][i]
        # Start/end character index of the answer in the text
        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])

        sequence_ids = tokenized_inputs.sequence_ids(i)
        # String of 1s in sequence_ids[i] is the context, find first and last
        context_start = sequence_ids.index(1)
        context_end = len(sequence_ids) - 1 - sequence_ids[::-1].index(1)

        # If the answer is out of the span (in the question) or after the context, set to 0,0
        if end_char < offsets[context_start][0] or start_char > offsets[context_end][1]:
            start_positions.append(0)
            end_positions.append(0)
        else:
            idx = context_start

            while offsets[idx][0] <= start_char and idx < context_end:
                idx += 1
            start_positions.append(idx - 1)

            while idx >= context_start and offsets[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    tokenized_inputs["start_positions"] = start_positions
    tokenized_inputs["end_positions"] = end_positions

    return tokenized_inputs

def freeze(model):
    # Set requires_grad to False for all parameters
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_last(model, unfreeze_n = 1):
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


def train(n_train = None, n_val = None, model_name = "distilbert-base-cased", freeze = True, unfreeze_n = 1):
    # ========= MODEL ========= #
    # Load model
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    # Freeze all but last layer + QA head
    if freeze:
        freeze(model)
        unfreeze_last(model, unfreeze_n=unfreeze_n)


    # ========= DATA ========= #
    dataset = load_dataset()
    data_collator = DefaultDataCollator()

    # Take subset of dataset if specified
    if n_train:
        dataset["train"] = dataset["train"].select(range(n_train))
    if n_val:
        dataset["validation"] = dataset["validation"].select(range(n_val))

    # Tokenize the dataset with our tokenization function
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    max_length = model.config.max_position_embeddings
    tokenize_partial = partial(tokenize, tokenizer=tokenizer, max_length=max_length)
    tokenized_dataset = dataset.map(tokenize_partial, batched=True, remove_columns=dataset["train"].column_names)


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

    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    # Perform validation before training
    print("Evaluating before training (epoch 0)...")
    metrics = trainer.evaluate()
    print(metrics)

    trainer.train()


if __name__ == "__main__":
    train(5000, 500, model_name = "distilbert-base-cased", freeze=False)
