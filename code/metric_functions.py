import evaluate
import numpy as np


# compute_metrics function called by trainer during trainer.evaluate()

def compute_metrics_sst2_bert(eval_preds):
    metric = evaluate.load("accuracy")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def compute_metrics_sst2_t5(eval_preds):
    metric = evaluate.load("accuracy")
    logits, labels = eval_preds
    predictions = np.argmax(logits[0], axis=-1)
    return metric.compute(predictions=predictions[:, 0], references=labels[:, 0])
