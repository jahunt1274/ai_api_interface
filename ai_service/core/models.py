"""
Core data models for the AI service.

These Pydantic models define the structure of requests and responses
for interacting with AI providers.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union, Any

from pydantic import BaseModel, Field, validator


class Role(str, Enum):
    """Role enumeration for chat messages."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"


class Message(BaseModel):
    """A chat message with role and content."""
    role: Role
    content: str
    name: Optional[str] = None


class UsageInfo(BaseModel):
    """Token usage information returned by the provider."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionRequest(BaseModel):
    """Request model for text completions."""
    prompt: str
    model: Optional[str] = None
    max_tokens: Optional[int] = 150
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    extra_params: Optional[Dict[str, Any]] = Field(default_factory=dict)


class CompletionResponse(BaseModel):
    """Response model for text completions."""
    completion: str
    model: str
    usage: Optional[UsageInfo] = None
    provider: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    provider_response: Optional[Dict[str, Any]] = None


class ChatCompletionRequest(BaseModel):
    """Request model for chat completions."""
    messages: List[Message]
    model: Optional[str] = None
    max_tokens: Optional[int] = 150
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    extra_params: Optional[Dict[str, Any]] = Field(default_factory=dict)


class ChatCompletionResponse(BaseModel):
    """Response model for chat completions."""
    message: Message
    model: str
    usage: Optional[UsageInfo] = None
    provider: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    provider_response: Optional[Dict[str, Any]] = None


class EmbeddingRequest(BaseModel):
    """Request model for embeddings."""
    input: Union[str, List[str]]
    model: Optional[str] = None
    extra_params: Optional[Dict[str, Any]] = Field(default_factory=dict)


class Embedding(BaseModel):
    """A single embedding vector."""
    embedding: List[float]
    index: int
    object: str = "embedding"


class EmbeddingResponse(BaseModel):
    """Response model for embeddings."""
    embeddings: List[Embedding]
    model: str
    usage: Optional[UsageInfo] = None
    provider: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    provider_response: Optional[Dict[str, Any]] = None