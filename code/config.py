import pathlib
import yaml as PyYAML
import datasets as huggingface_datasets

from functools import partial
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, TrainingArguments, \
    AutoModelForSequenceClassification, AutoModel

from dataset_tokenizers import tokenize_squad, tokenize_sst2
from freeze_strategies import all_but_last_n
from metric_functions import compute_accuracy
from models.lora import LoRAConfig, modify_with_lora
from optimizer import get_optimizer, get_scheduler


class Config:
    def __init__(self, path: pathlib.Path):
        yaml = PyYAML.load(open(path).read(), Loader=PyYAML.FullLoader)
        self.model = yaml["model"]
        self.train = yaml["train"]
        self.freeze = yaml["freeze"]
        self.dataset = yaml["dataset"]
        self.evaluate = yaml["evaluate"]
        self.optimizer = yaml["optimizer"]

    def load_model(self):
        """Load model for training
        :return: transformers.PreTrainedModel
        """
        # Return a model for the task based on the config
        match self.dataset["name"]:
            case "squad":
                model = AutoModelForQuestionAnswering.from_pretrained(self.model["base_model"])
            case "sst2":
                model = AutoModelForSequenceClassification.from_pretrained(self.model["base_model"])
            case _:
                model = AutoModel.from_pretrained(self.model["base_model"])

        if "modifier" in self.model and (self.model["modifier"] == "lora" or self.model["modifier"] == "ia3"):
            model = modify_with_lora(model, LoRAConfig(model.base_model_prefix, self.model["modifier"]))

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

    def load_dataset(self):
        """Load dataset and take subset if specified
        :return: datasets.Dataset
        """
        dataset = huggingface_datasets.load_dataset(self.dataset["name"], split=None)
        if "n_train" in self.dataset:
            dataset["train"] = dataset["train"].select(range(self.dataset["n_train"]))
        if "n_val" in self.dataset:
            dataset["validation"] = dataset["validation"].select(range(self.dataset["n_val"]))
        return dataset

    def tokenize_dataset(self, dataset, model):
        """Tokenize dataset
        :return: datasets.Dataset, transformers.PreTrainedTokenizer
        """
        tokenize_func_map = {"squad": tokenize_squad, "sst2": tokenize_sst2}
        tokenize_func = tokenize_func_map[self.dataset["name"]]

        tokenizer = AutoTokenizer.from_pretrained(self.model["base_model"])
        max_length = model.config.max_position_embeddings
        tokenize_partial = partial(tokenize_func, tokenizer=tokenizer, max_length=max_length)
        # Remove columns of the tokenized dataset that the model does not accept
        columns_to_remove = {"squad": dataset["train"].column_names, "sst2": ["idx", "sentence"]}
        return (
            dataset.map(tokenize_partial, batched=True, remove_columns=columns_to_remove[self.dataset["name"]]),
            tokenizer
        )

    def load_training_args(self):
        """Load training arguments
        :return: transformers.TrainingArguments
        """
        return TrainingArguments(**self.train)

    def load_metric_function(self):
        """Load metric function to be used during evaluation
        :return: function
        """
        metric_func_map = {"none": None, "accuracy": compute_accuracy}
        return metric_func_map[self.evaluate["metric_function"]]
    
    def load_optimizer(self, model, train_dataset):
        """Load optimizer
        :return: transformers.Optimizer, transformers.Scheduler
        """

        num_training_samples = len(train_dataset)
        num_steps = num_training_samples // self.train["per_device_train_batch_size"] * self.train["num_train_epochs"]
        self.optimizer["num_steps"] = num_steps

        if "trainable_param_names" not in self.optimizer:
            if "modifier" in self.model:
                if self.model["modifier"] == "ia3":
                    self.optimizer["trainable_param_names"] = ".*lora_b.*"
                elif self.model["modifier"] == "lora":
                    self.optimizer["trainable_param_names"] = ".*layer_norm.*|.*lora_[ab].*"
                else:
                    self.optimizer["trainable_param_names"] = ".*"
            else:
                self.optimizer["trainable_param_names"] = ".*"
            
        optimizer = get_optimizer(model, self.optimizer)
        scheduler = get_scheduler(optimizer, self.optimizer)
        return optimizer, scheduler