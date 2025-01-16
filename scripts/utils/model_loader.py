from scripts.llm.clients import BaseLLMClient, AzureClient, TGIClient
from scripts.llm.deployments import AZURE_DEPLOYMENTS, AWS_DEPLOYMENTS
from scripts.llm.configs import IP


class ModelLoader:
    """
    Utility class for loading and retrieving LLMs.
    """

    def __init__(self, model_name: str, ip: str = IP, temperature: float = 1):
        self.model_name = model_name
        self.ip = ip
        self.temperature = temperature
        """
        Args:
            model_name (str): Name of the model deployed on the server.
            ip (str): IP address of the text-generation-inference server.
        """

    def get_client(self) -> BaseLLMClient:
        """
        Creates a Client object based on the model specified.
        """
        if self.model_name in AZURE_DEPLOYMENTS:
            return AzureClient(self.model_name, temperature=self.temperature)
        elif self.model_name in AWS_DEPLOYMENTS:
            return TGIClient(self.ip, self.model_name, temperature=self.temperature)
        else:
            raise ValueError(f"Cannot find model with the name {self.model_name} deployed anywhere!")

    def get_deployment_type(self) -> str:
        if self.model_name in AZURE_DEPLOYMENTS:
            return "azure"
        elif self.model_name in AWS_DEPLOYMENTS:
            return "aws"
        else:
            raise ValueError(f"Cannot find model with the name {self.model_name} deployed anywhere!")
