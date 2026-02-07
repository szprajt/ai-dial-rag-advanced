import json
import requests

DIAL_EMBEDDINGS = 'https://ai-proxy.lab.epam.com/openai/deployments/{model}/embeddings'


class DialEmbeddingsClient:
    def __init__(self, deployment_name: str, api_key: str):
        if not api_key:
            raise ValueError("API key must be provided")
        self.deployment_name = deployment_name
        self.api_key = api_key
        self.endpoint = DIAL_EMBEDDINGS.format(model=deployment_name)

    def get_embeddings(self, input_list: list[str]) -> dict[int, list[float]]:
        headers = {
            "api-key": self.api_key,
            "Content-Type": "application/json"
        }
        
        payload = {
            "input": input_list
        }
        
        response = requests.post(self.endpoint, headers=headers, json=payload)
        response.raise_for_status()
        
        data = response.json()
        
        embeddings = {}
        for item in data.get("data", []):
            index = item.get("index")
            embedding = item.get("embedding")
            if index is not None and embedding:
                embeddings[index] = embedding
                
        return embeddings
