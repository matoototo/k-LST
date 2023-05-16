import pathlib
import yaml as PyYAML
import datasets as huggingface_datasets

from functools import partial
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, TrainingArguments, \
    AutoModelForSequenceClassification, AutoModel

from dataset_tokenizers import tokenize_squad, tokenize_sst2
from freeze_strategies import all_but_last_n
from metric_functions import compute_accuracy


class Config:
    def __init__(self, path: pathlib.Path):
        yaml = PyYAML.load(open(path).read(), Loader=PyYAML.FullLoader)
        self.model = yaml["model"]
        self.train = yaml["train"]
        self.freeze = yaml["freeze"]
        self.dataset = yaml["dataset"]
        self.evaluate = yaml["evaluate"]

    def load_model(self):
        """Load model for training
        :return: transformers.PreTrainedModel
        """
        # Return a model for the task based on the config
        match self.dataset["name"]:
            case "squad":
                return AutoModelForQuestionAnswering.from_pretrained(self.model["base_model"])
            case "sst2":
                return AutoModelForSequenceClassification.from_pretrained(self.model["base_model"])
            case _:
                return AutoModel.from_pretrained(self.model["base_model"])

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
