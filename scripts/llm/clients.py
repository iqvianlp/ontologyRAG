import abc
import logging
from datetime import datetime, timedelta
from typing import Any, Sequence, Union

import requests
import transformers
from azure.identity import ClientSecretCredential
from langchain.schema import HumanMessage
from langchain_community.adapters.openai import convert_openai_messages
from langchain_openai import AzureChatOpenAI
from transformers import AutoTokenizer, AutoConfig
from transformers.models.auto.tokenization_auto import TOKENIZER_MAPPING_NAMES

from scripts.llm.zero_shot_classification import max_length_from_configs
from scripts.llm.deployments import *
from scripts.llm.configs import *


LOGGER = logging.getLogger(__name__)

PromptType = Union[str, Sequence[dict[str, str]]]

class BaseLLMClient(metaclass=abc.ABCMeta):
    """
    Abstract base class for prompting LLMs.

    Attributes:
        model_name (str): Name of the model used by this client.
        chat_completion_enabled (bool): Whether chat completion is enabled for the model(s) this client is prompting.
        temperature: Temperature to use, usually between 0 and 1. Higher values make the output more random, while lower values make it more focused and deterministic.
        _timeout (int): Timeout to use when prompting.
    """

    def __init__(self, model_name):
        self.model_name = model_name
        self.chat_completion_enabled = False
        self.temperature = 1.0
        self._timeout = 300

    def __repr__(self):
        return f'model={self.model_name}; max_tokens={self.model_max_length}; url={self._url}'

    @abc.abstractmethod
    def invoke(self, prompt: PromptType, **kwargs: dict[str, Any]) -> str:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def model_max_length(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def _url(self):
        raise NotImplementedError

    def is_valid_prompt(self, prompt: str):
        tokens = self._token_count(prompt)
        return tokens, tokens <= self.model_max_length

    @abc.abstractmethod
    def _token_count(self, prompt: PromptType):
        raise NotImplementedError

    @abc.abstractmethod
    def is_alive(self):
        raise NotImplementedError


class BaseHuggingFaceLLMClient(BaseLLMClient):
    """
    Abstract base class implementing some common methods used by clients that interact with Hugging Face models.
    """

    def __init__(self, model_name, temperature: float = 1.0, chat_completion_enabled: bool = False):
        """
        Initializes a new BaseHuggingFaceLLMClient object.

        Args:
            model_name: Hugging Face model ID.
            temperature: Temperature to use, usually between 0 and 1. Higher values make the output more random, while lower values make it more focused and deterministic.
            chat_completion_enabled: Whether to treat the model as a chat model.
        """
        super().__init__(model_name)
        self.chat_completion_enabled = chat_completion_enabled
        self.temperature = temperature
        self._tokenizer = self._determine_tokenizer()

    def _determine_tokenizer(self):
        try:
            config = AutoConfig.from_pretrained(self.model_name, token=ACCESS_TOKEN)
            if hasattr(config, 'model_type') and config.model_type == 'whatever_llama':
                # This is a workaround to avoid crashing when AutoTokenizer is used with badly configured LLaMA
                # tokenizers.
                tok_class_name, fast_tok_class_name = TOKENIZER_MAPPING_NAMES[config.model_type]
                # Depending on available packages, different tokenizer classes may be available.
                tokenizer_class = getattr(transformers, tok_class_name if tok_class_name else fast_tok_class_name)
                tokenizer = tokenizer_class.from_pretrained(self.model_name)
                if hasattr(config, 'eos_token_id') and hasattr(config, 'bos_token_id'):
                    tokenizer.bos_token = tokenizer.convert_tokens_to_string(
                        tokenizer.convert_ids_to_tokens(config.bos_token_id)
                    )
                    tokenizer.eos_token = tokenizer.convert_tokens_to_string(
                        tokenizer.convert_ids_to_tokens(config.eos_token_id)
                    )
            else:
                tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=ACCESS_TOKEN)
        except requests.exceptions.ConnectionError:
            raise requests.exceptions.ConnectionError(
                'Error loading tokenizer; can the host access HuggingFace?')
        return tokenizer

    def _token_count(self, prompt: PromptType) -> int:
        if len(prompt) > 0 and isinstance(prompt[0], dict):
            prompt = self._tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        return len(self._tokenizer(prompt)['input_ids'])


class AWSClient(BaseHuggingFaceLLMClient):

    def __init__(self, ip, model_name, chat_completion_enabled: bool = False):
        if model_name not in AWS_DEPLOYMENTS:
            raise ValueError(f'Model "{model_name}" is invalid; please choose one of:\n{", ".join(AWS_DEPLOYMENTS)}')
        super().__init__(model_name, chat_completion_enabled)
        self.ip = ip
        self._tokenizer = self._determine_tokenizer()

    def invoke(self, prompt: PromptType, params: dict = None) -> str:
        """
        Generates text using the given prompt and generation params.

        Args:
            prompt: Text to submit to the LLM or a list of dicts with "role" and "content" keys.
            params: Generation params to submit to the LLM.

        Returns:
            Generated text.

        Raises:
            LLMClientException: Text generation failed due to a server error.
        """
        if len(prompt) > 0 and isinstance(prompt[0], dict):
            prompt = self._tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        response = self._post(prompt, params)
        if 200 <= response.status_code <= 299:
            return response.text
        else:
            raise LLMClientException(f'Generation failed with code {response.status_code}: {response.json()["error"]}')

    def _post(self, prompt: str, params: dict = None) -> requests.Response:
        params = params if params else {'max_new_tokens': 100}
        return requests.post(
            self._url,
            json={'text': prompt, 'params': params},
            timeout=self._timeout
        )

    def _determine_tokenizer(self):
        try:
            config = AutoConfig.from_pretrained(self.model_name)
            if hasattr(config, 'model_type') and config.model_type == 'llama':
                # This is a workaround to avoid crashing when AutoTokenizer is used with badly configured LLaMA
                # tokenizers.
                tok_class_name, fast_tok_class_name = TOKENIZER_MAPPING_NAMES[config.model_type]
                # Depending on available packages, different tokenizer classes may be available.
                tokenizer_class = getattr(transformers, tok_class_name if tok_class_name else fast_tok_class_name)
                tokenizer = tokenizer_class.from_pretrained(self.model_name)
            else:
                tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        except requests.exceptions.ConnectionError:
            raise requests.exceptions.ConnectionError(
                'Error loading tokenizer; can the host access HuggingFace?')
        return tokenizer

    @property
    def model_max_length(self):
        if self.model_name == FLAN_XXL:
            return 4096  # hard-coding for this model because we're hard-coding the tokenizer on AWS as per this:
            # https://huggingface.co/google/flan-t5-xxl/discussions/41#65117d0d33ddefa58fee136f
        return max_length_from_configs(configs=AutoConfig.from_pretrained(self.model_name))

    @property
    def _url(self):
        return f'http://{self.ip}:8080/{AWS_MODEL_PATHS[self.model_name]}'

    def is_alive(self):
        try:
            response = self._post('Hello World!')
            return 200 <= response.status_code <= 299
        except requests.exceptions.ConnectTimeout:
            return False


class TGIClient(BaseHuggingFaceLLMClient):
    """
    Client for communicating with a Text Generation Inference server.

    Attributes:
        _base_url (str): Base URL of the TGI server.
        _model_max_length (int): Max length the model can support.
    """

    def __init__(self, ip: str, model_name: str, port: int = 8080, temperature: float = 1.0, chat_completion_enabled: bool = False):
        """
        Initializes a new TGIClient object.

        Args:
            ip: IP address of the text-generation-inference server.
            model_name: Name of the model deployed on the server.
            port: Port of the server.
            temperature: Temperature to use, usually between 0 and 1. Higher values make the output more random, while lower values make it more focused and deterministic.
            chat_completion_enabled: Whether to treat the model as a chat model.
        """
        super().__init__(model_name, temperature, chat_completion_enabled)
        self._base_url = f'http://{ip}:{port}'
        self._verify_model_id(model_name)
        self._model_max_length = self._get_info().json()['max_input_length']

    def invoke(self, prompt: PromptType, params: dict = None) -> str:
        """
        Generates text using the given prompt and generation params. If an error occurs, this will always return the
        text of the response rather than raising an error.

        Args:
            prompt: Text to submit to the LLM.
            params: Generation params to submit to the LLM.

        Returns:
            Generated text.

        Raises:
            LLMClientException: Text generation failed due to a server error.
        """
        if len(prompt) > 0 and isinstance(prompt[0], dict):
            prompt = self._tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        response = self._post(prompt, params)
        try:
            if response.status_code == 200:
                return response.json()['generated_text']
            else:
                raise LLMClientException(
                    f'Generation failed with code {response.status_code}: {response.json()["error"]}'
                )
        except requests.exceptions.JSONDecodeError:
            raise LLMClientException(f'Generation failed with error: {response.text}')

    def _post(self, prompt: str, params: dict = None) -> requests.Response:
        params = params if params else {'max_new_tokens': 400, 'temperature': self.temperature}
        return requests.post(
            self._url,
            json={'inputs': prompt, 'parameters': params},
            timeout=self._timeout
        )

    @property
    def model_max_length(self):
        return self._model_max_length

    @property
    def _url(self):
        return f'{self._base_url}/generate'

    @property
    def _info_url(self):
        return f'{self._base_url}/info'

    def _get_info(self) -> requests.Response:
        return requests.get(self._info_url)

    def _verify_model_id(self, model_name) -> None:
        """
        Sanity check to ensure the deployed model is the same as the expected model.

        Returns:
            None

        Raises:
            LLMClientException: The server can't be reached or the served LLM is different than what was used to
                initialize this object.
        """
        response = self._get_info()
        if response.status_code == 200:
            if model_name != response.json()['model_id']:
                raise LLMClientException(
                    'The model being served does not match the model used to initialize this object.'
                )
        else:
            raise LLMClientException('The endpoint cannot be reached.')

    def is_alive(self) -> bool:
        try:
            response = self._get_info()
            return response.status_code == 200
        except requests.exceptions.ConnectTimeout:
            return False


class AzureClient(BaseLLMClient):

    def __init__(self, model_name, temperature: float = 1.0, chat_completion_enabled: bool = False):
        if model_name not in AZURE_DEPLOYMENTS:
            raise ValueError(f'Model "{model_name}" is invalid; please choose one of:\n{", ".join(AZURE_DEPLOYMENTS)}')
        super().__init__(model_name)
        self._token_requester = ClientSecretCredential(
                TENANT_ID,
                SERVICE_PRINCIPAL,
                SERVICE_PRINCIPAL_SECRET
            )
        self._connect()
        self.chat_completion_enabled = chat_completion_enabled
        self.temperature = temperature

    def _connect(self):
        self._token_requested_at = datetime.now().replace(microsecond=0)
        self._token_object = self._token_requester.get_token(SCOPE_NON_INTERACTIVE)
        LOGGER.info(f'New token will expire at {self._token_requested_at + timedelta(hours=1)} (client time; '
                    f'server time: {datetime.fromtimestamp(self._token_object.expires_on + 3600)})')
        self._connector = AzureChatOpenAI(
            azure_endpoint=self._url,
            openai_api_version=OPENAI_API_VERSION,
            openai_api_key=self._token_object.token,
            deployment_name=self.model_name,
            openai_api_type='azure_ad',
            temperature=self.temperature
        )

    def invoke(self, prompt: PromptType, **kwargs: dict[str, Any]) -> str:
        """
        Generates text using the given prompt.

        Args:
            prompt: Text to submit to the LLM or a list of dicts with "role" and "content" keys.
            kwargs: Unused.

        Returns:
            Generated text.
        """
        if self._token_expired():
            LOGGER.info('Azure token expired; requesting new one')
            self._connect()
        if isinstance(prompt, str):
            messages = [HumanMessage(content=prompt)]
        elif len(prompt) > 0 and isinstance(prompt[0], dict):
            messages = convert_openai_messages(prompt)
        else:
            messages = []
        return self._connector.invoke(messages).content

    def _token_expired(self):
        """Start requesting a new token just before the 1h expiry window."""
        return datetime.now() >= self._token_requested_at + timedelta(minutes=59)

    @property
    def model_max_length(self):
        return MAX_TOKENS[self.model_name]

    @property
    def _url(self):
        return f'{OPENAI_API_BASE}/{OPENAI_API_TYPE}/{OPENAI_ACCOUNT_NAME}'

    def _token_count(self, prompt):
        if len(prompt) > 0 and isinstance(prompt[0], dict):
            return self._connector.get_num_tokens_from_messages(convert_openai_messages(prompt))
        else:
            return self._connector.get_num_tokens(prompt)

    def is_alive(self):
        return self._token_object.token is not None


class LLMClientException(Exception):
    pass
