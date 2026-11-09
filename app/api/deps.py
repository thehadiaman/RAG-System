from fastapi import Request

from app.llm.bedrock import BedrockClient
from app.llm.embedder import Embedder
from app.llm.ollama import OllamaClient
from app.store.db import Database
from app.store.vectors import VectorStore


def get_ollama(request: Request) -> OllamaClient:
    return request.app.state.ollama


def get_bedrock(request: Request) -> BedrockClient:
    return request.app.state.bedrock


def get_embedder(request: Request) -> Embedder:
    return request.app.state.embedder


def get_db(request: Request) -> Database:
    return request.app.state.db


def get_vectors(request: Request) -> VectorStore:
    return request.app.state.vectors
