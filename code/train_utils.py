from functools import partial

import datasets as huggingface_datasets
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, DefaultDataCollator, TrainingArguments, Trainer
import torch
from tqdm.auto import tqdm
from update_policy import UpdatePolicy
import evaluate
import numpy as np
import collections


def load_dataset(dataset="squad", split=None):
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
        return_offsets_mapping=True,
        return_overflowing_tokens=True
    )

    start_positions = []
    end_positions = []
    example_ids = []

    # Update the start and end positions
    for i, offsets in enumerate(tokenized_inputs["offset_mapping"]):
        sample_idx = tokenized_inputs["overflow_to_sample_mapping"][i]
        answer = dataset["answers"][sample_idx]

        example_ids.append(dataset["id"][sample_idx])
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
    tokenized_inputs["example_id"] = example_ids
    tokenized_inputs.pop("overflow_to_sample_mapping")

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


def train(n_train=None, n_val=None, model_name="distilbert-base-cased", freeze=True, unfreeze_n=1):
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
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=5,
        weight_decay=0.01
    )

    train_dataset = tokenized_dataset["train"]
    eval_dataset = tokenized_dataset["validation"]
    eval_dataset_for_model = eval_dataset.remove_columns(["example_id", "offset_mapping"])
    train_dataset_for_model = train_dataset.remove_columns(["example_id", "offset_mapping"])

    trainer = Trainer(
        model,
        training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    metric = evaluate.load("squad")

    def compute_metrics(features, examples):
        start_logits = []
        end_logits = []
        for batch in eval_dataset_for_model.iter(training_args.per_device_eval_batch_size):
            batch = {k: torch.tensor(batch[k], device="cuda") for k in eval_dataset_for_model.column_names}
            with torch.no_grad():
                outputs = model(**batch)
            start_logits.append(outputs.start_logits.cpu())
            end_logits.append(outputs.end_logits.cpu())

        start_logits = np.concatenate(start_logits)
        end_logits = np.concatenate(end_logits)
        start_logits = start_logits[:len(eval_dataset)]
        end_logits = end_logits[:len(eval_dataset)]

        n_best = 20
        example_to_features = collections.defaultdict(list)
        for idx, feature in enumerate(features):
            example_to_features[feature["example_id"]].append(idx)

        predicted_answers = []
        for example in tqdm(examples):
            example_id = example["id"]
            context = example["context"]
            answers = []

            # Loop through all features associated with that example
            for feature_index in example_to_features[example_id]:
                start_logit = start_logits[feature_index]
                end_logit = end_logits[feature_index]
                offsets = features[feature_index]["offset_mapping"]

                start_indexes = np.argsort(start_logit)[-1: -n_best - 1: -1].tolist()
                end_indexes = np.argsort(end_logit)[-1: -n_best - 1: -1].tolist()
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        answer = {
                            "text": context[offsets[start_index][0]: offsets[end_index][1]],
                            "logit_score": start_logit[start_index] + end_logit[end_index],
                        }
                        answers.append(answer)

            # Select the answer with the best score
            if len(answers) > 0:
                best_answer = max(answers, key=lambda x: x["logit_score"])
                predicted_answers.append(
                    {"id": example_id, "prediction_text": best_answer["text"]}
                )
            else:
                predicted_answers.append({"id": example_id, "prediction_text": ""})

        theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
        return metric.compute(predictions=predicted_answers, references=theoretical_answers)

    # Perform validation before training
    print("Evaluating before training (epoch 0)...")
    metrics = compute_metrics(eval_dataset, dataset["validation"])
    print(metrics)

    # Perform initialization and create update policy before training
    update_policy = UpdatePolicy(model)
    num_steps = int(training_args.num_train_epochs * train_dataset.num_rows / training_args.per_device_train_batch_size)
    trainer.create_optimizer_and_scheduler(num_steps)
    progress_bar = tqdm(range(num_steps))

    # Perform training
    for epoch in range(int(training_args.num_train_epochs)):
        # Apply the update policy before each epoch
        # update_policy.apply(epoch, metrics)

        for batch in train_dataset_for_model.iter(training_args.per_device_train_batch_size):
            batch = {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}
            trainer.training_step(model, batch)
            trainer.optimizer.step()
            trainer.optimizer.zero_grad()
            trainer.lr_scheduler.step()
            progress_bar.update(1)

        # Evaluate after each epoch
        print(f"Evaluating epoch {epoch + 1}...")
        metrics = compute_metrics(eval_dataset, dataset["validation"])
        print(metrics)


if __name__ == "__main__":
    train(5000, 500, model_name="distilbert-base-uncased", freeze=False)
