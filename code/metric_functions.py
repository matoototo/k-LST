import evaluate
import numpy as np

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