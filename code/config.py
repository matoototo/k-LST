import yaml as PyYAML
import datasets as huggingface_datasets
from functools import partial
from datasets import concatenate_datasets
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, TrainingArguments, \
    AutoModelForSequenceClassification, AutoModel, T5ForConditionalGeneration, AutoModelForMaskedLM
from dataset_tokenizers import *
from freeze_strategies import all_but_last_n
from metric_functions import *
from models.lora import modify_with_lora
from models.siva import modify_with_siva
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
        self.modifier = self.model["modifier"] if "modifier" in self.model else "none"

    def load_model(self, model_path):
        """Load model for training
        :return: transformers.PreTrainedModel
        """
        # Return a model for the task based on the config
        base_model = model_path if model_path is not None else self.model["base_model"]
        if self.model["model_type"] == "t5":
            model = T5ForConditionalGeneration.from_pretrained(base_model)
        elif self.modifier in ["mezo", "prompt_based"]:
            model = AutoModelForMaskedLM.from_pretrained(base_model)
        else:
            match self.dataset["name"]:
                case "squad":
                    model = AutoModelForQuestionAnswering.from_pretrained(base_model)
                case "sst2":
                    model = AutoModelForSequenceClassification.from_pretrained(base_model)
                case "glue":
                    if self.dataset["subset"] == "stsb":
                        num_labels = 1
                    else:
                        num_labels = 2
                    model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=num_labels)
                case _:
                    model = AutoModelForSequenceClassification.from_pretrained(base_model)

        if "lora" in self.model:
            model = modify_with_lora(model, self.model["lora"])
        
        if "siva" in self.model:
            model = modify_with_siva(model, self.model["siva"])

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
        additional_args = {"model_type": self.model["model_type"], "modifier": self.modifier}
        return strategy_map[self.adapter["strategy"]](model, **self.adapter["args"] | additional_args)

    def load_dataset(self):
        """Load dataset and take subset if specified
        :return: datasets.Dataset
        """
        if "subset" not in self.dataset:
            dataset = huggingface_datasets.load_dataset(self.dataset["name"], split=None)
        else:
            dataset = huggingface_datasets.load_dataset(self.dataset["name"], self.dataset["subset"], split=None)
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
        tokenize_func_map = {"squad bert none": tokenize_squad, "sst2 bert none": tokenize_sst2,
                             "sst2 t5 none": tokenize_sst2_t5, "sst2 bert prompt_based": tokenize_sst2_prompt,
                             "sst2 bert with_prompt": tokenize_sst2_prompt_no_label,
                             "sst2 bert mezo": tokenize_sst2_prompt,
                             "cola bert none": tokenize_cola, "qnli bert none": tokenize_qnli,
                             "rte bert none": tokenize_rte, "stsb bert none": tokenize_stsb}
        if "subset" not in self.dataset:
            tokenize_func = tokenize_func_map[f"{self.dataset['name']} {self.model['model_type']} {self.modifier}"]
        else:
            tokenize_func = tokenize_func_map[f"{self.dataset['subset']} {self.model['model_type']} {self.modifier}"]

        tokenizer = AutoTokenizer.from_pretrained(self.model["base_model"])

        # Hack: the roberta pretrained models have a misconfigured max_position_embeddings parameter, defined as 514,
        # even though they only support a max length of 512 which leads to errors during truncation
        if self.model["base_model"] in ["roberta-large", "robert-base", "t5-base"]:
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
            if self.modifier in ["mezo", "prompt_based", "with_prompt"]:
                # Read the prompt from config
                if "modifier_args" in self.model and "prompt" in self.model["modifier_args"]:
                    prompt = self.model["modifier_args"]["prompt"]
                    assert "[SENTENCE]" in prompt and "[MASK]" in prompt, \
                        'Prompt needs to contain "[SENTENCE]" and "[MASK]".'
                    tokenize_partial = partial(tokenize_partial, prompt=prompt)
                if self.modifier != "with_prompt":
                    # Only for prompt-based FT we remove 'label' as the tokenizer adds 'labels'
                    # and potentially we read positive and negative label words from config
                    columns_to_remove.append("label")
                    if "modifier_args" in self.model:
                        if "neg_label" in self.model["modifier_args"]:
                            tokenize_partial = partial(tokenize_partial,
                                                       neg_label=self.model["modifier_args"]["neg_label"])
                        if "pos_label" in self.model["modifier_args"]:
                            tokenize_partial = partial(tokenize_partial,
                                                       pos_label=self.model["modifier_args"]["pos_label"])
        elif self.dataset["subset"] == "cola":
            columns_to_remove = ["idx", "sentence"]
        elif self.dataset["subset"] == "qnli":
            columns_to_remove = ["idx", "sentence", "question"]
        elif self.dataset["subset"] == "rte":
            columns_to_remove = ["idx", "sentence1", "sentence2"]
        elif self.dataset["subset"] == "stsb":
            columns_to_remove = ["idx", "sentence1", "sentence2"]
        return (
            dataset.map(tokenize_partial, batched=True, remove_columns=columns_to_remove, load_from_cache_file=False),
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
            if self.model['model_type'] == "bert":
                if self.modifier in ["mezo", "prompt_based"]:
                    metric_func = partial(compute_metrics_sst2_bert_prompt, tokenizer=tokenizer)
                    preprocess_logits_func = partial(preprocess_logits_sst2_prompt, tokenizer=tokenizer)
                    if "modifier_args" in self.model:
                        if "neg_label" in self.model["modifier_args"]:
                            neg_label = self.model["modifier_args"]["neg_label"]
                            metric_func = partial(metric_func, neg_label=neg_label)
                            preprocess_logits_func = partial(preprocess_logits_func, neg_label=neg_label)
                        if "pos_label" in self.model["modifier_args"]:
                            pos_label = self.model["modifier_args"]["pos_label"]
                            preprocess_logits_func = partial(preprocess_logits_func, pos_label=pos_label)
                else:
                    metric_func = compute_metrics_sst2_bert
            elif self.model['model_type'] == "t5":
                metric_func = compute_metrics_sst2_t5
        elif self.dataset['subset'] == "stsb":
            metric_func = compute_metrics_stsb_bert
        else:
            metric_func = compute_metrics_sst2_bert
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
        if self.modifier == "mezo":
            if "modifier_args" in self.model and "eps" in self.model["modifier_args"]:
                kwargs |= {"eps": self.model["modifier_args"]["eps"]}
            return MezoTrainer(*args, **kwargs)
        else:
            return Trainer(*args, **kwargs)
