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

# https://huggingface.co/learn/nlp-course/chapter7/7
def compute_metrics(start_logits, end_logits, features, examples):
    n_best = 20
    max_answer_length = 30
    metric = evaluate.load("squad")

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

            start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip answers that are not fully in the context
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # Skip answers with a length that is either < 0 or > max_answer_length
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue

                    answer = {
                        "text": context[offsets[start_index][0] : offsets[end_index][1]],
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
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01
    )

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
    
    # Evaluation
    def preprocess_validation_examples(examples):
        max_length = 384
        stride = 128

        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=max_length,
            truncation="only_second",
            stride=stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_map = inputs.pop("overflow_to_sample_mapping")
        example_ids = []

        for i in range(len(inputs["input_ids"])):
            sample_idx = sample_map[i]
            example_ids.append(examples["id"][sample_idx])

            sequence_ids = inputs.sequence_ids(i)
            offset = inputs["offset_mapping"][i]
            inputs["offset_mapping"][i] = [
                o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
            ]

        inputs["example_id"] = example_ids
        return inputs
    
    validation_dataset = dataset["validation"].map(
        preprocess_validation_examples,
        batched=True,
        remove_columns=dataset["validation"].column_names,
    )

    predictions, _, _ = trainer.predict(validation_dataset)
    start_logits, end_logits = predictions
    final_metrics = compute_metrics(start_logits, end_logits, validation_dataset, dataset["validation"])
    print("Evaluation Results:", final_metrics)


if __name__ == "__main__":
    train(5000, 500, model_name="distilbert-base-cased", freeze=False)
