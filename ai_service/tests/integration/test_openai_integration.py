"""
Integration tests for the OpenAI provider.

This module contains integration tests for the OpenAI provider implementation.
These tests make actual API calls to OpenAI and should be run only when necessary.
"""

import os
import pytest
from typing import List

from ai_service.core.models import (
    ChatCompletionRequest,
    CompletionRequest,
    EmbeddingRequest,
    Message,
    Role,
)
from ai_service.providers.openai import OpenAIProvider


# Skip all tests if OPENAI_API_KEY is not set or if SKIP_INTEGRATION_TESTS is set
pytestmark = [
    pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY environment variable not set"
    ),
    pytest.mark.skipif(
        os.environ.get("SKIP_INTEGRATION_TESTS"),
        reason="Integration tests are skipped"
    ),
    pytest.mark.integration,
]


@pytest.fixture
def api_key() -> str:
    """Get the OpenAI API key from the environment."""
    return os.environ.get("OPENAI_API_KEY", "")


@pytest.fixture
def provider(api_key) -> OpenAIProvider:
    """Create an OpenAI provider for testing."""
    return OpenAIProvider(api_key=api_key, default_model="gpt-3.5-turbo")


class TestOpenAIProviderIntegration:
    """Integration tests for the OpenAIProvider class."""

    def test_create_completion(self, provider: OpenAIProvider):
        """Test creating a completion with a real API call."""
        request = CompletionRequest(
            prompt="Write a haiku about coding.",
            max_tokens=50,
            temperature=0.7
        )
        
        response = provider.create_completion(request)
        
        # Check that we got a valid response
        assert response.completion
        assert len(response.completion) > 0
        assert response.model
        assert response.provider == "openai"
        assert response.usage is not None
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        assert response.usage.total_tokens > 0

    def test_create_chat_completion(self, provider: OpenAIProvider):
        """Test creating a chat completion with a real API call."""
        request = ChatCompletionRequest(
            messages=[
                Message(role=Role.SYSTEM, content="You are a helpful assistant that speaks like a pirate."),
                Message(role=Role.USER, content="Tell me about the weather today.")
            ],
            max_tokens=50,
            temperature=0.7
        )
        
        response = provider.create_chat_completion(request)
        
        # Check that we got a valid response
        assert response.message
        assert response.message.content
        assert len(response.message.content) > 0
        assert response.message.role == Role.ASSISTANT
        assert response.model
        assert response.provider == "openai"
        assert response.usage is not None
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        assert response.usage.total_tokens > 0

    def test_create_embedding(self, provider: OpenAIProvider):
        """Test creating embeddings with a real API call."""
        request = EmbeddingRequest(
            input=["Hello, world!", "Embeddings are useful for semantic search."],
            model="text-embedding-ada-002"
        )
        
        response = provider.create_embedding(request)
        
        # Check that we got valid embeddings
        assert len(response.embeddings) == 2  # We sent 2 texts
        assert len(response.embeddings[0].embedding) > 0  # Should have non-empty vector
        assert response.embeddings[0].index == 0
        assert response.embeddings[1].index == 1
        assert response.model == "text-embedding-ada-002"
        assert response.provider == "openai"
        assert response.usage is not None
        assert response.usage.prompt_tokens > 0
        assert response.usage.total_tokens > 0

    def test_get_model_list(self, provider: OpenAIProvider):
        """Test getting the list of available models."""
        models = provider.get_model_list()
        
        # Check that we got a non-empty list of models
        assert isinstance(models, List)
        assert len(models) > 0
        assert all(isinstance(model, str) for model in models)
        
        # Check that commonly available models are included
        common_models = ["gpt-3.5-turbo", "text-embedding-ada-002"]
        for model in common_models:
            matching_models = [m for m in models if model in m]
            assert len(matching_models) > 0, f"Expected to find model containing '{model}'"

    def test_get_token_count(self, provider: OpenAIProvider):
        """Test counting tokens."""
        text = "This is a test of the token counting functionality."
        count = provider.get_token_count(text, model="gpt-3.5-turbo")
        
        # Token count should be reasonable for this text
        # Roughly 1 token per word in English
        assert count > 0
        assert 5 <= count <= 15