"""
Unit tests for the AI service client.

This module contains unit tests for the client interface for interacting with
AI providers.
"""

import pytest
from unittest.mock import MagicMock, patch

from ai_service.client import AIServiceClient
from ai_service.core.errors import ConfigurationError, ProviderNotAvailableError
from ai_service.core.models import (
    ChatCompletionRequest,
    CompletionRequest,
    EmbeddingRequest,
    Message,
    Role,
)
from ai_service.providers.openai import OpenAIProvider
from ai_service.providers.mock import MockProvider


class TestAIServiceClient:
    """Tests for the AIServiceClient class."""

    def test_initialization_with_openai(self):
        """Test initializing the client with OpenAI provider."""
        client = AIServiceClient(provider="openai", api_key="test-key")
        assert client.provider_name == "openai"
        assert isinstance(client.provider, OpenAIProvider)
        assert client.provider.api_key == "test-key"

    def test_initialization_with_mock(self):
        """Test initializing the client with mock provider."""
        client = AIServiceClient(provider="mock")
        assert client.provider_name == "mock"
        assert isinstance(client.provider, MockProvider)

    def test_initialization_with_unknown_provider(self):
        """Test that initialization fails with an unknown provider."""
        with pytest.raises(ProviderNotAvailableError):
            AIServiceClient(provider="unknown")

    def test_initialization_without_api_key(self):
        """Test that initialization fails without an API key for OpenAI."""
        with pytest.raises(ConfigurationError):
            AIServiceClient(provider="openai")

    def test_create_completion_from_string(self):
        """Test creating a completion from a string prompt."""
        # Create a mock provider
        mock_provider = MagicMock()
        mock_provider.create_completion.return_value = "test response"
        
        # Create a client with the mock provider
        client = AIServiceClient(provider="mock")
        client.provider = mock_provider
        
        # Call create_completion with a string
        result = client.create_completion("test prompt")
        
        # Check that the provider was called with the right request
        mock_provider.create_completion.assert_called_once()
        request = mock_provider.create_completion.call_args[0][0]
        assert isinstance(request, CompletionRequest)
        assert request.prompt == "test prompt"
        
        # Check the result
        assert result == "test response"

    def test_create_completion_from_request(self):
        """Test creating a completion from a request object."""
        # Create a mock provider
        mock_provider = MagicMock()
        mock_provider.create_completion.return_value = "test response"
        
        # Create a client with the mock provider
        client = AIServiceClient(provider="mock")
        client.provider = mock_provider
        
        # Create a request
        request = CompletionRequest(
            prompt="test prompt",
            max_tokens=10,
            temperature=0.5
        )
        
        # Call create_completion with the request
        result = client.create_completion(request)
        
        # Check that the provider was called with the right request
        mock_provider.create_completion.assert_called_once_with(request)
        
        # Check the result
        assert result == "test response"

    def test_create_chat_completion(self):
        """Test creating a chat completion."""
        # Create a mock provider
        mock_provider = MagicMock()
        mock_provider.create_chat_completion.return_value = "test response"
        
        # Create a client with the mock provider
        client = AIServiceClient(provider="mock")
        client.provider = mock_provider
        
        # Create a request
        request = ChatCompletionRequest(
            messages=[
                Message(role=Role.SYSTEM, content="You are a helpful assistant."),
                Message(role=Role.USER, content="test message")
            ],
            max_tokens=10
        )
        
        # Call create_chat_completion
        result = client.create_chat_completion(request)
        
        # Check that the provider was called with the right request
        mock_provider.create_chat_completion.assert_called_once_with(request)
        
        # Check the result
        assert result == "test response"

    def test_create_embedding_from_string(self):
        """Test creating an embedding from a string."""
        # Create a mock provider
        mock_provider = MagicMock()
        mock_provider.create_embedding.return_value = "test response"
        
        # Create a client with the mock provider
        client = AIServiceClient(provider="mock")
        client.provider = mock_provider
        
        # Call create_embedding with a string
        result = client.create_embedding("test text")
        
        # Check that the provider was called with the right request
        mock_provider.create_embedding.assert_called_once()
        request = mock_provider.create_embedding.call_args[0][0]
        assert isinstance(request, EmbeddingRequest)
        assert request.input == "test text"
        
        # Check the result
        assert result == "test response"

    def test_create_embedding_from_list(self):
        """Test creating embeddings from a list of strings."""
        # Create a mock provider
        mock_provider = MagicMock()
        mock_provider.create_embedding.return_value = "test response"
        
        # Create a client with the mock provider
        client = AIServiceClient(provider="mock")
        client.provider = mock_provider
        
        # Call create_embedding with a list
        result = client.create_embedding(["test1", "test2"])
        
        # Check that the provider was called with the right request
        mock_provider.create_embedding.assert_called_once()
        request = mock_provider.create_embedding.call_args[0][0]
        assert isinstance(request, EmbeddingRequest)
        assert request.input == ["test1", "test2"]
        
        # Check the result
        assert result == "test response"

    def test_create_embedding_from_request(self):
        """Test creating embeddings from a request object."""
        # Create a mock provider
        mock_provider = MagicMock()
        mock_provider.create_embedding.return_value = "test response"
        
        # Create a client with the mock provider
        client = AIServiceClient(provider="mock")
        client.provider = mock_provider
        
        # Create a request
        request = EmbeddingRequest(
            input="test text",
            model="text-embedding-ada-002"
        )
        
        # Call create_embedding with the request
        result = client.create_embedding(request)
        
        # Check that the provider was called with the right request
        mock_provider.create_embedding.assert_called_once_with(request)
        
        # Check the result
        assert result == "test response"

    def test_get_token_count(self):
        """Test getting token count."""
        # Create a mock provider
        mock_provider = MagicMock()
        mock_provider.get_token_count.return_value = 10
        
        # Create a client with the mock provider
        client = AIServiceClient(provider="mock")
        client.provider = mock_provider
        
        # Call get_token_count
        result = client.get_token_count("test text", model="gpt-3.5-turbo")
        
        # Check that the provider was called with the right parameters
        mock_provider.get_token_count.assert_called_once_with("test text", "gpt-3.5-turbo")
        
        # Check the result
        assert result == 10

    def test_get_model_list(self):
        """Test getting the list of available models."""
        # Create a mock provider
        mock_provider = MagicMock()
        mock_provider.get_model_list.return_value = ["model1", "model2"]
        
        # Create a client with the mock provider
        client = AIServiceClient(provider="mock")
        client.provider = mock_provider
        
        # Call get_model_list
        result = client.get_model_list()
        
        # Check that the provider was called
        mock_provider.get_model_list.assert_called_once()
        
        # Check the result
        assert result == ["model1", "model2"]

    def test_register_provider(self):
        """Test registering a new provider."""
        # Create a mock provider class
        MockProviderClass = MagicMock()
        
        # Register the provider
        AIServiceClient.register_provider("custom", MockProviderClass)
        
        # Check that the provider was registered
        assert "custom" in AIServiceClient._provider_registry
        assert AIServiceClient._provider_registry["custom"] == MockProviderClass
        
        # Create a client with the new provider
        with patch.object(MockProviderClass, "__call__", return_value=MagicMock()):
            client = AIServiceClient(provider="custom", api_key="test-key")
            assert client.provider_name == "custom"