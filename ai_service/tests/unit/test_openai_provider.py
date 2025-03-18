"""
Unit tests for the OpenAI provider.

This module contains unit tests for the OpenAI provider implementation.
"""

import pytest
from unittest.mock import MagicMock, patch

from ai_service.core.errors import (
    AIServiceError,
    AuthenticationError as AIAuthenticationError,
    InvalidRequestError,
    ModelNotAvailableError,
    ProviderError,
    RateLimitError as AIRateLimitError,
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
        # Mock the message to the API
        mock_message = MagicMock()
        mock_message.content = "test response"
        mock_message.role = "assistant"
        mock_message.name = None
        
        # Mock the API response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=mock_message)]
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
        # Mock the message to the API
        mock_message = MagicMock()
        mock_message.content = "test response"
        mock_message.role = "assistant"
        mock_message.name = None
        
        # Mock the API response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=mock_message)]
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

class TestOpenAIErrorHandling:
    """Tests for error handling in the OpenAI provider."""
    @pytest.mark.parametrize(
        "openai_error_class,expected_error_class,needs_response",
        [
            ("AuthenticationError", AIAuthenticationError, True),
            ("RateLimitError", AIRateLimitError, True),
            ("BadRequestError", InvalidRequestError, True),
            ("APIError", ProviderError, False),
        ],
    )
    def test_error_conversions(self, provider, openai_error_class, expected_error_class, needs_response):
        """Test all error conversion cases."""
        from ai_service.providers.openai import openai as provider_openai
        
        # Get the actual exception class from the provider's openai module
        error_class = getattr(provider_openai, openai_error_class)
        mock_body = {"error": {"message": "test error", "type": "invalid_request_error"}}

        # Create a mock that raises this specific error with all required arguments
        def mock_chat_completion(*args, **kwargs):
            if needs_response:
                # Create mock response and body objects as required by OpenAI exceptions
                mock_response = MagicMock()
                mock_response.status_code = 401  # Use appropriate status code for each error
                mock_response.headers = {}
                mock_response.text = "Error text"
                
                raise error_class(
                    message="test error",
                    response=mock_response,
                    body=mock_body
                )
            else:
                # For APIError, use a simpler constructor
                raise error_class(
                    message="test error",
                    request="test",
                    body=mock_body
                )
        
        # Patch and test
        with patch.object(provider, 'create_chat_completion', side_effect=mock_chat_completion):
            request = CompletionRequest(prompt="test prompt", model="gpt-3.5-turbo")
            
            with pytest.raises(expected_error_class) as exc_info:
                provider.create_completion(request)
            
            assert "test error" in str(exc_info.value)
    
    def test_generic_error(self, provider):
        """Test handling of generic exceptions not specific to OpenAI."""
        
        def mock_chat_completion(*args, **kwargs):
            # Raise a generic exception that isn't an OpenAI-specific type
            raise ValueError("some unexpected error")
        
        with patch.object(provider, 'create_chat_completion', side_effect=mock_chat_completion):
            request = CompletionRequest(prompt="test prompt", model="gpt-3.5-turbo")
            
            with pytest.raises(AIServiceError) as exc_info:
                provider.create_completion(request)
            
            # Check that the error message is wrapped in our expected format
            assert "Unexpected error: some unexpected error" in str(exc_info.value)
            assert provider.PROVIDER_NAME == exc_info.value.provider
            assert "original_error" in exc_info.value.details