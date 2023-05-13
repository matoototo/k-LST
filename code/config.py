import pathlib
import yaml as PyYAML
import datasets as huggingface_datasets

from functools import partial
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, TrainingArguments

from dataset_tokenizers import tokenize_squad
from freeze_strategies import all_but_last_n

class Config:
    def __init__(self, path : pathlib.Path):
        yaml = PyYAML.load(open(path).read(), Loader=PyYAML.FullLoader)
        self.model = yaml["model"]
        self.train = yaml["train"]
        self.freeze = yaml["freeze"]
        self.dataset = yaml["dataset"]

    def load_model(self):
        """Load model for training
        :return: transformers.PreTrainedModel
        """
        # We can expand this to return models for other tasks based on the rest of the config
        return AutoModelForQuestionAnswering.from_pretrained(self.model["base_model"])

    def freeze_model(self, model):
        """Apply freezing strategy to model
        :return: None
        """
        # Pass freeze["args"] to the strategy function
        strategy_map = { "none": lambda model: None, "all_but_last_n": all_but_last_n }
        strategy_map[self.freeze["strategy"]](model, **self.freeze["args"])

    def load_dataset(self):
        """Load dataset and take subset if specified
        :return: datasets.Dataset
        """
        dataset = huggingface_datasets.load_dataset(self.dataset["name"], split=None)
        if self.dataset["n_train"]:
            dataset["train"] = dataset["train"].select(range(self.dataset["n_train"]))
        if self.dataset["n_val"]:
            dataset["validation"] = dataset["validation"].select(range(self.dataset["n_val"]))
        return dataset

    def tokenize_dataset(self, dataset, model):
        """Tokenize dataset
        :return: datasets.Dataset, transformers.PreTrainedTokenizer
        """
        tokenize_func_map = { "squad": tokenize_squad }
        tokenize_func = tokenize_func_map[self.dataset["name"]]

        tokenizer = AutoTokenizer.from_pretrained(self.model["base_model"])
        max_length = model.config.max_position_embeddings
        tokenize_partial = partial(tokenize_func, tokenizer=tokenizer, max_length=max_length)
        return (
            dataset.map(tokenize_partial, batched=True, remove_columns=dataset["train"].column_names),
            tokenizer
        )

    def load_training_args(self):
        """Load training arguments
        :return: transformers.TrainingArguments
        """
        return TrainingArguments(**self.train)
