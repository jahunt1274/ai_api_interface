"""
Unit tests for the OpenAI provider.

This module contains unit tests for the OpenAI provider implementation.
"""

import pytest
from unittest.mock import MagicMock, patch

from ai_service.core.errors import (
    AIServiceError,
    AuthenticationError,
    InvalidRequestError,
    ModelNotAvailableError,
    ProviderError,
    RateLimitError,
)
from ai_service.core.models import (
    ChatCompletionRequest,
    CompletionRequest,
    EmbeddingRequest,
    Message,
    Role,
)
from ai_service.providers.openai import OpenAIProvider


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client."""
    with patch("ai_service.providers.openai.OpenAI") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        yield mock_client


@pytest.fixture
def provider(mock_openai_client):
    """Create an OpenAI provider with a mock client."""
    return OpenAIProvider(api_key="test-key", default_model="gpt-3.5-turbo")


class TestOpenAIProvider:
    """Tests for the OpenAIProvider class."""

    def test_initialization(self):
        """Test that the provider initializes correctly."""
        provider = OpenAIProvider(api_key="test-key", default_model="gpt-3.5-turbo")
        assert provider.api_key == "test-key"
        assert provider.default_model == "gpt-3.5-turbo"
        assert provider.timeout == 60.0
        assert provider.organization is None

    def test_initialization_with_organization(self):
        """Test that the provider initializes with an organization ID."""
        provider = OpenAIProvider(
            api_key="test-key",
            default_model="gpt-3.5-turbo",
            organization="test-org"
        )
        assert provider.organization == "test-org"

    def test_create_completion(self, provider, mock_openai_client):
        """Test creating a completion."""
        # Mock the API response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="test response", role="assistant"))]
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15
        mock_response.model_dump.return_value = {"test": "response"}
        
        # Set up the mock client
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        # Create a request
        request = CompletionRequest(prompt="test prompt", max_tokens=10)
        
        # Call the provider
        response = provider.create_completion(request)
        
        # Check that the client was called correctly
        mock_openai_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_openai_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "gpt-3.5-turbo"
        assert len(call_kwargs["messages"]) == 1
        assert call_kwargs["messages"][0]["role"] == "user"
        assert call_kwargs["messages"][0]["content"] == "test prompt"
        assert call_kwargs["max_tokens"] == 10
        
        # Check the response
        assert response.completion == "test response"
        assert response.model == "gpt-3.5-turbo"
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 5
        assert response.usage.total_tokens == 15
        assert response.provider == "openai"
        assert response.provider_response == {"test": "response"}

    def test_create_chat_completion(self, provider, mock_openai_client):
        """Test creating a chat completion."""
        # Mock the API response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="test response", role="assistant"))]
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15
        mock_response.model_dump.return_value = {"test": "response"}
        
        # Set up the mock client
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        # Create a request
        request = ChatCompletionRequest(
            messages=[
                Message(role=Role.SYSTEM, content="You are a test assistant."),
                Message(role=Role.USER, content="test message")
            ],
            max_tokens=10
        )
        
        # Call the provider
        response = provider.create_chat_completion(request)
        
        # Check that the client was called correctly
        mock_openai_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_openai_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "gpt-3.5-turbo"
        assert len(call_kwargs["messages"]) == 2
        assert call_kwargs["messages"][0]["role"] == "system"
        assert call_kwargs["messages"][0]["content"] == "You are a test assistant."
        assert call_kwargs["messages"][1]["role"] == "user"
        assert call_kwargs["messages"][1]["content"] == "test message"
        assert call_kwargs["max_tokens"] == 10
        
        # Check the response
        assert response.message.content == "test response"
        assert response.message.role == Role.ASSISTANT
        assert response.model == "gpt-3.5-turbo"
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 5
        assert response.usage.total_tokens == 15
        assert response.provider == "openai"
        assert response.provider_response == {"test": "response"}

    def test_create_embedding(self, provider, mock_openai_client):
        """Test creating embeddings."""
        # Mock the API response
        mock_embedding = MagicMock(embedding=[0.1, 0.2, 0.3], index=0, object="embedding")
        mock_response = MagicMock()
        mock_response.data = [mock_embedding]
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.total_tokens = 10
        mock_response.model_dump.return_value = {"test": "response"}
        
        # Set up the mock client
        mock_openai_client.embeddings.create.return_value = mock_response
        
        # Create a request
        request = EmbeddingRequest(input="test text", model="text-embedding-ada-002")
        
        # Call the provider
        response = provider.create_embedding(request)
        
        # Check that the client was called correctly
        mock_openai_client.embeddings.create.assert_called_once()
        call_kwargs = mock_openai_client.embeddings.create.call_args.kwargs
        assert call_kwargs["model"] == "text-embedding-ada-002"
        assert call_kwargs["input"] == "test text"
        
        # Check the response
        assert len(response.embeddings) == 1
        assert response.embeddings[0].embedding == [0.1, 0.2, 0.3]
        assert response.embeddings[0].index == 0
        assert response.model == "text-embedding-ada-002"
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 0  # OpenAI embeddings don't return completion tokens
        assert response.usage.total_tokens == 10
        assert response.provider == "openai"
        assert response.provider_response == {"test": "response"}

    def test_get_model_list(self, provider, mock_openai_client):
        """Test getting the list of available models."""
        # Mock the API response
        mock_model = MagicMock(id="model-id")
        mock_response = MagicMock()
        mock_response.data = [mock_model]
        
        # Set up the mock client
        mock_openai_client.models.list.return_value = mock_response
        
        # Call the provider
        models = provider.get_model_list()
        
        # Check that the client was called correctly
        mock_openai_client.models.list.assert_called_once()
        
        # Check the response
        assert len(models) == 1
        assert models[0] == "model-id"

    def test_get_token_count(self, provider):
        """Test counting tokens."""
        # Test with a simple string
        count = provider.get_token_count("This is a test.", model="gpt-3.5-turbo")
        assert count > 0

    @pytest.mark.parametrize(
        "error_class,expected_error",
        [
            (MagicMock(__name__="openai.AuthenticationError"), AuthenticationError),
            (MagicMock(__name__="openai.RateLimitError"), RateLimitError),
            (MagicMock(__name__="openai.BadRequestError"), InvalidRequestError),
            (MagicMock(__name__="openai.APIError"), ProviderError),
            (Exception, AIServiceError),
        ],
    )
    def test_error_handling(self, provider, mock_openai_client, error_class, expected_error):
        """Test that API errors are converted to appropriate AI service errors."""
        # Configure the mock to raise an error
        mock_openai_client.chat.completions.create.side_effect = error_class("test error")
        
        # Create a request
        request = CompletionRequest(prompt="test prompt")
        
        # Call the provider and check the error
        with pytest.raises(expected_error) as exc_info:
            provider.create_completion(request)
        
        # Check error details
        assert "test error" in str(exc_info.value)
        assert provider.PROVIDER_NAME in str(exc_info.value) or hasattr(exc_info.value, "provider")
        if hasattr(exc_info.value, "provider"):
            assert exc_info.value.provider == provider.PROVIDER_NAME