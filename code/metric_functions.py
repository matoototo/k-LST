import evaluate
import numpy as np


# compute_metrics function called by trainer during trainer.evaluate()

def compute_accuracy(eval_preds):
    metric = evaluate.load("accuracy")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)
