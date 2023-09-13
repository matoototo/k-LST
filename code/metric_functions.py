import evaluate
import numpy as np
import torch

def get_labels(references):
    labels = []
    for i in references:
        if len(labels) == 0:
            labels.append(i)
        elif len(labels) == 2:
            break
        else:
            if i not in labels:
                labels.append(i)
    return labels

# compute_metrics function called by trainer during trainer.evaluate()

def compute_metrics_sst2_bert(eval_preds):
    metric_accuracy = evaluate.load("accuracy")
    metric_f1 = evaluate.load("f1")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    accuracy = metric_accuracy.compute(predictions=predictions, references=labels)
    f1 = metric_f1.compute(predictions=predictions, references=labels)
    return accuracy | f1

def compute_metrics_stsb_bert(eval_preds):
    metric_accuracy = evaluate.load("pearsonr")
    logits, labels = eval_preds
    # predictions = np.argmax(logits, axis=-1)
    predictions = logits

    pearsonr = metric_accuracy.compute(predictions=predictions, references=labels)
    return pearsonr

def compute_metrics_sst2_t5(eval_preds):
    metric_accuracy = evaluate.load("accuracy")
    metric_f1 = evaluate.load("f1")
    logits, labels = eval_preds
    # First item of the tuple gives token scores
    predictions = np.argmax(logits[0], axis=-1)

    # Compare the first token, which is either "positive" or "negative"
    accuracy = metric_accuracy.compute(predictions=predictions[:, 0], references=labels[:, 0])
    f1 = metric_f1.compute(predictions=predictions[:, 0], references=labels[:, 0], labels=get_labels(labels[:, 0]), average="macro")
    return accuracy | f1


def compute_metrics_sst2_bert_prompt(eval_preds, tokenizer, neg_label="terrible"):
    metric_accuracy = evaluate.load("accuracy")
    metric_f1 = evaluate.load("f1")
    logits, labels = eval_preds

    # Get label encoding only at the position of the mask token
    mask_idx = np.argmin(labels, axis=1) - 3
    labels = labels[range(len(labels)), mask_idx]
    # Replace encoding with 0 or 1
    labels = np.where(labels == tokenizer.encode(" " + neg_label)[1], 0, 1)

    predictions = np.argmax(logits, axis=-1)

    accuracy = metric_accuracy.compute(predictions=predictions, references=labels)
    f1 = metric_f1.compute(predictions=predictions, references=labels)
    return accuracy | f1


def preprocess_logits_sst2_prompt(logits, labels, tokenizer, neg_label="terrible", pos_label="great"):
    mask_idx = torch.argmin(labels, dim=1) - 3
    label_encodings = [tokenizer.encode(" " + label)[1] for label in [neg_label, pos_label]]
    # Logits only for the two label words at the mask position
    logits = logits[torch.arange(logits.shape[0]), mask_idx][:, label_encodings]
    return logits
