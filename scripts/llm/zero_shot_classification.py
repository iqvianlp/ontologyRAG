import re
import csv
from typing import Union

import pandas
import torch
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer, AutoConfig, AutoModelForCausalLM


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NEWLINE = "\n"

DEFAULT_MODEL_MAX_LENGTH = 512


class LLMPipeline(object):
    """
    Implementation of zero-shot classification using Hugging Face pre-trained models and Hugging Face Pipeline
    framework

    Args:
        model_name: path to Hugging Face pre-trained model
        cache_dir: local cache directory to store pertinent Hugging Face resources
        labels: list of candidate labels
    """
    def __init__(self, model_name: str, labels: list, cache_dir: str = None):
        # instantiate tokenizer outside of pipeline in order to enforce max_length for truncation
        model_max_length = max_length_from_configs(
            configs=AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=model_max_length, cache_dir=cache_dir)

        # instantiate pipeline from local cache
        self._pipeline = pipeline(
            "zero-shot-classification",
            model=model_name,
            tokenizer=tokenizer,
            device=torch.cuda.current_device() if DEVICE == torch.device("cuda") else DEVICE,
            model_kwargs={"cache_dir": cache_dir} if cache_dir is not None else {}
        )
        self._labels = labels

    def classify(self, text: str) -> str:
        """
        Generates zero-shot classification predictions.
        Args:
            text: the input text.

        Returns:
            the predicted class
        """
        pipeline_output = self._pipeline(text, self._labels)
        return pipeline_output["labels"][0]


class PromptBasedLLM(object):
    """
    Prompt-based implementation of generative zero-shot classification using Hugging Face pre-trained models

    Args:
        model_name: path to Hugging Face pre-trained model
        cache_dir: local cache directory to store pertinent Hugging Face resources
        max_new_tokens: [DEFAULT: 300] the maximum number of tokens the model can generate
    """
    def __init__(self, model_name: str, cache_dir: str = None, max_new_tokens: int = 300, seq_to_seq: bool = True):
        self._seq_to_seq = seq_to_seq

        # enforce max length for truncation
        model_max_length = max_length_from_configs(
            configs=AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
        )
        # tokenizer loaded from local cache
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name, model_max_length=model_max_length, cache_dir=cache_dir
        )

        # model loaded from local cache
        self._model = self._init_model(model_name=model_name, model_cache_dir=cache_dir)
        self._model.to(DEVICE)

        # the maximum number of tokens the model can generate
        self._max_new_tokens = max_new_tokens

    def classify(self, prompt: str) -> str:
        """
        Generates zero-shot classification predictions.
        Args:
            prompt: the formatted prompt

        Returns:
            the predicted class
        """
        return self._generate(prompt=prompt)[0].strip().strip("'").strip().strip('"').strip()

    def extract(self, prompt: str) -> list:
        """
        Generates entity extraction predictions.
        Args:
            prompt: the formatted prompt

        Returns:
            a list of extracted entities
        """
        return [entity.strip().strip("'").strip().strip('"').strip() for entity in self._generate(prompt=prompt)]

    def _generate(self, prompt: str):
        # tokenize the prompt
        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True)

        # set tokenized content to same device as model
        inputs = {k: inputs[k].to(DEVICE) for k in inputs}

        # generate and decode predictions
        outputs = self._model.generate(**inputs, max_new_tokens=self._max_new_tokens)
        return self._tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def generate_prompt(
            self, text: str, labels: list, task: str, text_descriptor: str = "input", input_text_first: bool = True
    ) -> str:
        """
        Formats the prompt.

        Args:
            text: the input text
            labels: list of labels
            task: wording of the task to be performed by the model
            text_descriptor: a descriptor of the input text (DEFAULT: "Input")
            input_text_first: whether the input text comes first in the formatted prompt

        Returns:
            The formatted prompt:

                input_text_first = True
                    <capitalized descriptor>: <text as as single line>
                    Task: <task> <descriptor>
                    Choices: <comma-delimited list of choices>
                    Output:

                input_text_first = True
                    Task: <task> <descriptor>
                    Choices: <comma-delimited list of choices>
                    <capitalized descriptor>: <text as as single line>
                    Output:
        """
        text_as_single_line = self._text_as_single_line(text)

        # put input text first in the prompt
        if input_text_first:
            return \
                f"{text_descriptor.capitalize()}: {text_as_single_line}{NEWLINE}Task: {task} {text_descriptor}." \
                f"{NEWLINE}Choices: {', '.join(labels)}{NEWLINE}Output: "

        # put input text last in the prompt
        else:
            return \
                f"Task: {task} {text_descriptor}.{NEWLINE}Choices: {', '.join(labels)}{NEWLINE}" \
                f"{text_descriptor.capitalize()}: {text}{NEWLINE}Output: "

    def generate_binary_prompt(self, text: str, question: str):
        """
        Formats binary prompt.

        Args:
            text: the input text
            question: closed-ended question to be answered by "yes" or "no"

        Returns:
            The formatted prompt.
        """
        return f'{question}{NEWLINE}Sentence: “{self._text_as_single_line(text)}”{NEWLINE}Respond with "yes" or "no".'

    def _init_model(self, model_name: str, model_cache_dir):
        if self._seq_to_seq:
            return AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=model_cache_dir)
        else:
            return AutoModelForCausalLM.from_pretrained(model_name, cache_dir=model_cache_dir)

    @staticmethod
    def _text_as_single_line(text: str):
        """
        Replaces newline characters with space characters; input text become a single line.
        """
        # replace newline with single space character
        text = re.sub(NEWLINE, " ", text)

        # replace consecutive white space characters with a single white space character
        text = re.sub(" {2,}", " ", text)

        return text


def preprocess_csv_for_zero_shot_classification(
        path_to_csv: str, text_column: str, id_column: str = None
) -> Union[list, dict]:
    """
    Args:
        path_to_csv: path to the csv file that contains the input texts
        text_column: name of the column in the csv file that contains the input texts
        id_column: name of the column in the csv file that contains the IDs of the input texts

    Returns:
        a dictionary mapping text IDs to their corresponding texts, if the id_column is in the csv file; otherwise a
        list of texts as read from the csv
    """
    input_data = pandas.read_csv(path_to_csv, dtype=str, quoting=csv.QUOTE_ALL)
    if text_column not in input_data:
        raise ValueError(f"column '{text_column}' not found in input file")

    if id_column is not None and id_column in input_data:
        output = dict()
        for row, text_id in enumerate(input_data[id_column]):
            output[text_id.strip()] = input_data[text_column][row].strip()
        return output

    else:
        return [text.strip() for text in input_data[text_column]]


def max_length_from_configs(configs):
    model_max_length = DEFAULT_MODEL_MAX_LENGTH
    for mml_key in ["n_positions", "max_sequence_length", "max_position_embeddings"]:
        if hasattr(configs, mml_key):
            model_max_length = getattr(configs, mml_key)
    return model_max_length
