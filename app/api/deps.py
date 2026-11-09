from fastapi import Request

from app.llm.bedrock import BedrockClient
from app.llm.ollama import OllamaClient


def get_ollama(request: Request) -> OllamaClient:
    return request.app.state.ollama


def get_bedrock(request: Request) -> BedrockClient:
    return request.app.state.bedrock
