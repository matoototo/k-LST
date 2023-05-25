def freeze_all(model):
    # Set requires_grad to False for all parameters
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_last(model, unfreeze_n=1):
    if unfreeze_n != 0:
        # Unfreeze the last n layers
        if "distilbert" in model.name_or_path:
            # DistilBERT
            for param in model.distilbert.transformer.layer[-unfreeze_n:].parameters():
                param.requires_grad = True
        elif "bert" in model.name_or_path:
            # BERT
            for param in model.bert.encoder.layer[-unfreeze_n:].parameters():
                param.requires_grad = True

    # QA head if it exists
    if hasattr(model, "qa_outputs"):
        for param in model.qa_outputs.parameters():
            param.requires_grad = True
    elif hasattr(model, "classifier"):
        for param in model.classifier.parameters():
            param.requires_grad = True

def all_but_last_n(model, n=1):
    freeze_all(model)
    unfreeze_last(model, unfreeze_n=n)
