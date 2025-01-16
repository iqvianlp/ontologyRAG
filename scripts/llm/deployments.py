# This file contains all deployed models.

# Open AI models hosted on Azure:
GPT_35_TURBO = "gpt-35-turbo-0613"
GPT_4 = "gpt-4"

AZURE_DEPLOYMENTS = [
    GPT_35_TURBO,
    GPT_4,
]

# Source: https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/models @ 29 Sep 2023
MAX_TOKENS = {
    GPT_35_TURBO: 4096,
    GPT_4: 8192,
    'meta-llama/Meta-Llama-3-8B-Instruct': 8191, # Customizable
}

# Huggingface models hosted on AWS:
FLAN_XXL = 'google/flan-t5-xxl'
AWS_MODEL_PATHS = {
    FLAN_XXL: 'flan-t5-xxl',
    'meta-llama/Meta-Llama-3-8B-Instruct': 'meta-llama3-instruct'
}
AWS_DEPLOYMENTS = list(AWS_MODEL_PATHS)
