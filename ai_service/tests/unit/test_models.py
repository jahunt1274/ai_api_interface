"""
Unit tests for data models.

This module contains unit tests for the data models used in the AI service.
"""

import json
import pytest
from datetime import datetime
from typing import Dict, List, Any

from ai_service.core.models import (
    Role,
    Message,
    UsageInfo,
    CompletionRequest,
    CompletionResponse,
    ChatCompletionRequest,
    ChatCompletionResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    Embedding
)


class TestRole:
    """Tests for the Role enum."""

    def test_role_values(self):
        """Test that Role enum contains the expected values."""
        assert Role.SYSTEM == "system"
        assert Role.USER == "user"
        assert Role.ASSISTANT == "assistant"
        assert Role.FUNCTION == "function"

    def test_role_comparison(self):
        """Test that Role values can be compared with strings."""
        assert Role.USER == "user"
        assert "system" == Role.SYSTEM
        assert Role.ASSISTANT != "user"


class TestMessage:
    """Tests for the Message model."""

    def test_message_creation(self):
        """Test that a Message can be created with expected values."""
        msg = Message(role=Role.USER, content="Hello, world!")
        assert msg.role == Role.USER
        assert msg.content == "Hello, world!"
        assert msg.name is None

    def test_message_with_name(self):
        """Test that a Message can be created with a name."""
        msg = Message(role=Role.FUNCTION, content="Result", name="calculator")
        assert msg.role == Role.FUNCTION
        assert msg.content == "Result"
        assert msg.name == "calculator"

    def test_message_serialization(self):
        """Test that a Message can be serialized to JSON."""
        msg = Message(role=Role.SYSTEM, content="You are a helpful assistant.", name="system")
        msg_dict = msg.model_dump()
        assert msg_dict["role"] == "system"
        assert msg_dict["content"] == "You are a helpful assistant."
        assert msg_dict["name"] == "system"


class TestUsageInfo:
    """Tests for the UsageInfo model."""

    def test_usage_info_creation(self):
        """Test that a UsageInfo can be created with expected values."""
        usage = UsageInfo(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 20
        assert usage.total_tokens == 30

    def test_usage_info_serialization(self):
        """Test that a UsageInfo can be serialized to JSON."""
        usage = UsageInfo(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        usage_dict = usage.model_dump()
        assert usage_dict["prompt_tokens"] == 10
        assert usage_dict["completion_tokens"] == 20
        assert usage_dict["total_tokens"] == 30


class TestCompletionRequest:
    """Tests for the CompletionRequest model."""

    def test_completion_request_creation(self):
        """Test that a CompletionRequest can be created with expected values."""
        request = CompletionRequest(prompt="Tell me a joke")
        assert request.prompt == "Tell me a joke"
        assert request.max_tokens == 150  # Default value
        assert request.temperature == 0.7  # Default value

    def test_completion_request_with_params(self):
        """Test that a CompletionRequest can be created with custom parameters."""
        request = CompletionRequest(
            prompt="Tell me a joke",
            model="gpt-4",
            max_tokens=100,
            temperature=0.5,
            top_p=0.9,
            n=2,
            stream=True,
            stop=["END"]
        )
        assert request.prompt == "Tell me a joke"
        assert request.model == "gpt-4"
        assert request.max_tokens == 100
        assert request.temperature == 0.5
        assert request.top_p == 0.9
        assert request.n == 2
        assert request.stream is True
        assert request.stop == ["END"]

    def test_completion_request_extra_params(self):
        """Test that a CompletionRequest can include extra parameters."""
        request = CompletionRequest(
            prompt="Tell me a joke",
            extra_params={"presence_penalty": 0.5, "frequency_penalty": 0.5}
        )
        assert request.extra_params["presence_penalty"] == 0.5
        assert request.extra_params["frequency_penalty"] == 0.5


class TestCompletionResponse:
    """Tests for the CompletionResponse model."""

    def test_completion_response_creation(self):
        """Test that a CompletionResponse can be created with expected values."""
        response = CompletionResponse(
            completion="Why did the chicken cross the road?",
            model="gpt-3.5-turbo",
            provider="openai"
        )
        assert response.completion == "Why did the chicken cross the road?"
        assert response.model == "gpt-3.5-turbo"
        assert response.provider == "openai"
        assert response.usage is None
        assert isinstance(response.created_at, datetime)

    def test_completion_response_with_usage(self):
        """Test that a CompletionResponse can include usage information."""
        usage = UsageInfo(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        response = CompletionResponse(
            completion="Why did the chicken cross the road?",
            model="gpt-3.5-turbo",
            provider="openai",
            usage=usage
        )
        assert response.usage == usage
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 20
        assert response.usage.total_tokens == 30


class TestChatCompletionRequest:
    """Tests for the ChatCompletionRequest model."""

    def test_chat_completion_request_creation(self):
        """Test that a ChatCompletionRequest can be created with expected values."""
        messages = [
            Message(role=Role.SYSTEM, content="You are a helpful assistant."),
            Message(role=Role.USER, content="Tell me a joke")
        ]
        request = ChatCompletionRequest(messages=messages)
        assert len(request.messages) == 2
        assert request.messages[0].role == Role.SYSTEM
        assert request.messages[0].content == "You are a helpful assistant."
        assert request.messages[1].role == Role.USER
        assert request.messages[1].content == "Tell me a joke"
        assert request.max_tokens == 150  # Default value
        assert request.temperature == 0.7  # Default value

    def test_chat_completion_request_with_params(self):
        """Test that a ChatCompletionRequest can be created with custom parameters."""
        messages = [
            Message(role=Role.USER, content="Tell me a joke")
        ]
        request = ChatCompletionRequest(
            messages=messages,
            model="gpt-4",
            max_tokens=100,
            temperature=0.5,
            top_p=0.9,
            n=2,
            stream=True,
            stop=["END"]
        )
        assert len(request.messages) == 1
        assert request.model == "gpt-4"
        assert request.max_tokens == 100
        assert request.temperature == 0.5
        assert request.top_p == 0.9
        assert request.n == 2
        assert request.stream is True
        assert request.stop == ["END"]


class TestChatCompletionResponse:
    """Tests for the ChatCompletionResponse model."""

    def test_chat_completion_response_creation(self):
        """Test that a ChatCompletionResponse can be created with expected values."""
        message = Message(
            role=Role.ASSISTANT,
            content="Why did the chicken cross the road?"
        )
        response = ChatCompletionResponse(
            message=message,
            model="gpt-3.5-turbo",
            provider="openai"
        )
        assert response.message == message
        assert response.message.role == Role.ASSISTANT
        assert response.message.content == "Why did the chicken cross the road?"
        assert response.model == "gpt-3.5-turbo"
        assert response.provider == "openai"
        assert response.usage is None
        assert isinstance(response.created_at, datetime)

    def test_chat_completion_response_with_usage(self):
        """Test that a ChatCompletionResponse can include usage information."""
        message = Message(
            role=Role.ASSISTANT,
            content="Why did the chicken cross the road?"
        )
        usage = UsageInfo(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        response = ChatCompletionResponse(
            message=message,
            model="gpt-3.5-turbo",
            provider="openai",
            usage=usage
        )
        assert response.usage == usage
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 20
        assert response.usage.total_tokens == 30


class TestEmbeddingRequest:
    """Tests for the EmbeddingRequest model."""

    def test_embedding_request_with_string(self):
        """Test that an EmbeddingRequest can be created with a string input."""
        request = EmbeddingRequest(input="Hello, world!")
        assert request.input == "Hello, world!"
        assert request.model is None
        assert request.extra_params == {}

    def test_embedding_request_with_list(self):
        """Test that an EmbeddingRequest can be created with a list of strings."""
        request = EmbeddingRequest(input=["Hello", "world"])
        assert request.input == ["Hello", "world"]
        assert len(request.input) == 2
        assert request.input[0] == "Hello"
        assert request.input[1] == "world"

    def test_embedding_request_with_model(self):
        """Test that an EmbeddingRequest can specify a model."""
        request = EmbeddingRequest(
            input="Hello, world!",
            model="text-embedding-ada-002"
        )
        assert request.input == "Hello, world!"
        assert request.model == "text-embedding-ada-002"


class TestEmbeddingResponse:
    """Tests for the EmbeddingResponse model."""

    def test_embedding_response_creation(self):
        """Test that an EmbeddingResponse can be created with expected values."""
        embeddings = [
            Embedding(embedding=[0.1, 0.2, 0.3], index=0, object="embedding"),
            Embedding(embedding=[0.4, 0.5, 0.6], index=1, object="embedding")
        ]
        response = EmbeddingResponse(
            embeddings=embeddings,
            model="text-embedding-ada-002",
            provider="openai"
        )
        assert len(response.embeddings) == 2
        assert response.embeddings[0].embedding == [0.1, 0.2, 0.3]
        assert response.embeddings[0].index == 0
        assert response.embeddings[1].embedding == [0.4, 0.5, 0.6]
        assert response.embeddings[1].index == 1
        assert response.model == "text-embedding-ada-002"
        assert response.provider == "openai"
        assert response.usage is None
        assert isinstance(response.created_at, datetime)

    def test_embedding_response_with_usage(self):
        """Test that an EmbeddingResponse can include usage information."""
        embeddings = [
            Embedding(embedding=[0.1, 0.2, 0.3], index=0, object="embedding")
        ]
        usage = UsageInfo(prompt_tokens=10, completion_tokens=0, total_tokens=10)
        response = EmbeddingResponse(
            embeddings=embeddings,
            model="text-embedding-ada-002",
            provider="openai",
            usage=usage
        )
        assert response.usage == usage
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 0
        assert response.usage.total_tokens == 10