import yaml as PyYAML
import datasets as huggingface_datasets
from functools import partial
from datasets import concatenate_datasets
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, TrainingArguments, \
    AutoModelForSequenceClassification, AutoModel, T5ForConditionalGeneration, AutoModelForMaskedLM
from dataset_tokenizers import tokenize_squad, tokenize_sst2, tokenize_sst2_t5, tokenize_sst2_prompt
from freeze_strategies import all_but_last_n
from metric_functions import compute_metrics_sst2_bert, compute_metrics_sst2_t5, preprocess_logits_sst2_prompt, \
    compute_metrics_sst2_bert_prompt
from models.lora import modify_with_lora
from optimizer import get_optimizer, get_scheduler
from adapters import ladder_side_tuning, ladder_side_distillation
from transformers import Trainer
from trainers import MezoTrainer


class Config:
    def __init__(self, path):
        yaml = PyYAML.load(open(path).read(), Loader=PyYAML.FullLoader)
        self.model = yaml["model"]
        self.train = yaml["train"]
        self.freeze = yaml["freeze"]
        self.dataset = yaml["dataset"]
        self.optimizer = yaml["optimizer"]
        self.adapter = yaml["adapter"]

    def load_model(self, model_path):
        """Load model for training
        :return: transformers.PreTrainedModel
        """
        # Return a model for the task based on the config
        base_model = model_path if model_path is not None else self.model["base_model"]
        if self.model["model_type"] == "t5":
            model = T5ForConditionalGeneration.from_pretrained(base_model)
        elif "prompt" in self.model["model_type"]:
            model = AutoModelForMaskedLM.from_pretrained(base_model)
        else:
            match self.dataset["name"]:
                case "squad":
                    model = AutoModelForQuestionAnswering.from_pretrained(self.model["base_model"])
                case "sst2":
                    model = AutoModelForSequenceClassification.from_pretrained(self.model["base_model"])
                case _:
                    model = AutoModel.from_pretrained(self.model["base_model"])

        if "lora" in self.model:
            model = modify_with_lora(model, self.model["lora"])

        return model

    def freeze_model(self, model):
        """Apply freezing strategy to model
        :return: None
        """
        if self.freeze["strategy"] == "none":
            return
        # Pass freeze["args"] to the strategy function
        strategy_map = {"all_but_last_n": all_but_last_n}
        strategy_map[self.freeze["strategy"]](model, **self.freeze["args"])

    def add_adapters(self, model):
        """Add adapters to model
        :return: None
        """
        if self.adapter["strategy"] == "none": return model
        if "args" not in self.adapter: self.adapter["args"] = {}
        strategy_map = {"lst": ladder_side_tuning, "lst_distill": ladder_side_distillation}
        return strategy_map[self.adapter["strategy"]](model,
                                                      **self.adapter["args"] | {"model_type": self.model["model_type"]})

    def load_dataset(self):
        """Load dataset and take subset if specified
        :return: datasets.Dataset
        """
        dataset = huggingface_datasets.load_dataset(self.dataset["name"], split=None)
        if "n_train" in self.dataset:
            dataset["train"] = dataset["train"].select(range(self.dataset["n_train"]))
        if "n_val" in self.dataset:
            dataset["validation"] = dataset["validation"].select(range(self.dataset["n_val"]))
        if "k" in self.dataset:
            k = self.dataset["k"]
            pos = dataset["train"].filter(lambda example: example["label"] == 0).shuffle()
            neg = dataset["train"].filter(lambda example: example["label"] == 1).shuffle()
            train_pos = pos.select(range(k))
            train_neg = neg.select(range(k))
            val_pos = pos.select(range(k, 2 * k))
            val_neg = neg.select(range(k, 2 * k))
            dataset["train"] = concatenate_datasets([train_pos, train_neg]).shuffle()
            dataset["validation"] = concatenate_datasets([val_pos, val_neg]).shuffle()
        return dataset

    def tokenize_dataset(self, dataset, model):
        """Tokenize dataset
        :return: datasets.Dataset, transformers.PreTrainedTokenizer
        """
        tokenize_func_map = {"squad bert": tokenize_squad, "sst2 bert": tokenize_sst2, "sst2 t5": tokenize_sst2_t5,
                             "sst2 bert prompt": tokenize_sst2_prompt}
        tokenize_func = tokenize_func_map[f"{self.dataset['name']} {self.model['model_type']}"]

        tokenizer = AutoTokenizer.from_pretrained(self.model["base_model"])

        if self.model["base_model"] == "t5-base":
            max_length = 512
        else:
            max_length = model.config.max_position_embeddings

        tokenize_partial = partial(tokenize_func, tokenizer=tokenizer, max_length=max_length)
        # Remove columns of the tokenized dataset that the model does not accept
        columns_to_remove = []
        if self.dataset["name"] == "squad":
            columns_to_remove = dataset["train"].column_names
        elif self.dataset["name"] == "sst2":
            columns_to_remove = ["idx", "sentence"]
            if "modifier" in self.model and self.model["modifier"] == "mezo" or "prompt" in self.model["model_type"]:
                columns_to_remove.append("label")
                if "modifier_args" in self.model:
                    if "neg_label" in self.model["modifier_args"]:
                        tokenize_partial = partial(tokenize_partial, neg_label=self.model["modifier_args"]["neg_label"])
                    if "pos_label" in self.model["modifier_args"]:
                        tokenize_partial = partial(tokenize_partial, pos_label=self.model["modifier_args"]["pos_label"])
        return (
            dataset.map(tokenize_partial, batched=True, remove_columns=columns_to_remove),
            tokenizer
        )

    def load_training_args(self):
        """Load training arguments
        :return: transformers.TrainingArguments
        """
        return TrainingArguments(**self.train)

    def load_metric_function(self, tokenizer):
        """Load metric function and the function to preprocess logits to be used for evaluation
        :return: function, function
        """
        metric_func = None
        preprocess_logits_func = None
        if self.dataset['name'] == "sst2":
            if "bert" in self.model['model_type']:
                if "prompt" in self.model['model_type']:
                    metric_func = partial(compute_metrics_sst2_bert_prompt, tokenizer=tokenizer)
                    preprocess_logits_func = partial(preprocess_logits_sst2_prompt, tokenizer=tokenizer)
                    if "modifier_args" in self.model:
                        if "neg_label" in self.model["modifier_args"]:
                            neg_label = self.model["modifier_args"]["neg_label"]
                            metric_func = partial(metric_func, neg_label=neg_label)
                            preprocess_logits_func = partial(preprocess_logits_func, neg_label=neg_label)
                        if "pos_label" in self.model["modifier_args"]:
                            pos_label = self.model["modifier_args"]["pos_label"]
                            metric_func = partial(metric_func, pos_label=pos_label)
                            preprocess_logits_func = partial(preprocess_logits_func, pos_label=pos_label)
                else:
                    metric_func = compute_metrics_sst2_bert
            elif "t5" in self.model['model_type']:
                metric_func = compute_metrics_sst2_t5
        return metric_func, preprocess_logits_func

    def load_optimizer(self, model, train_dataset):
        """Load optimizer
        :return: transformers.Optimizer, transformers.Scheduler
        """

        num_training_samples = len(train_dataset)
        num_steps = num_training_samples // self.train["per_device_train_batch_size"] * self.train["num_train_epochs"]
        self.optimizer["num_steps"] = num_steps

        if "trainable_param_names" not in self.optimizer:
            self.optimizer["trainable_param_names"] = ".*"

        optimizer = get_optimizer(model, self.optimizer)
        scheduler = get_scheduler(optimizer, self.optimizer)
        return optimizer, scheduler

    def load_trainer(self, *args, **kwargs):
        """Loads an appropriate trainer instance given model modifiers
        :return: transformers.Trainer
        """
        if "modifier" in self.model and self.model["modifier"] == "mezo":
            if "modifier_args" in self.model and "eps" in self.model["modifier_args"]:
                kwargs |= {"eps": self.model["modifier_args"]["eps"]}
            return MezoTrainer(*args, **kwargs)
        else:
            return Trainer(*args, **kwargs)
