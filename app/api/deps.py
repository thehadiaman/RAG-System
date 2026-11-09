from fastapi import Request

from app.llm.ollama import OllamaClient


def get_ollama(request: Request) -> OllamaClient:
    return request.app.state.ollama
