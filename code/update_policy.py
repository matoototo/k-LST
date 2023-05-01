import math

from torch.nn.modules import Module


class UpdatePolicy:
    def __init__(self, model: Module):
        self.model = model

    best_eval_loss = math.inf
    unfreeze_n = 1

    # Applies the update policy by setting requires_grad on parameters
    def apply(self, epoch=0, metrics=None):
        # Update all parameters
        for param in self.model.parameters():
            param.requires_grad = True

        # Disable updates on individual parameter after the first 2 epochs
        parameter = self.model.get_parameter("distilbert.embeddings.word_embeddings.weight")
        parameter.requires_grad = epoch in range(2)

        # Freeze all layers but the last one if eval_loss is better than best_eval_loss by a margin
        if metrics["eval_loss"] < self.best_eval_loss - 1:
            self.freeze()
            self.unfreeze_last()

        self.best_eval_loss = min(self.best_eval_loss, metrics["eval_loss"])

    def freeze(self):
        # Set requires_grad to False for all parameters
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_last(self):
        # Unfreeze the last n layers
        if "distilbert" in self.model.name_or_path:
            # DistilBERT
            for param in self.model.distilbert.transformer.layer[-self.unfreeze_n:].parameters():
                param.requires_grad = True
        elif "bert" in self.model.name_or_path:
            # BERT
            for param in self.model.bert.encoder.layer[-self.unfreeze_n:].parameters():
                param.requires_grad = True

        # QA head
        for param in self.model.qa_outputs.parameters():
            param.requires_grad = True
