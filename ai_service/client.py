"""
Main client class for the AI service.

This module provides the main client interface for interacting with AI providers.
"""

import logging
from typing import Any, Dict, List, Optional, Type, Union

from ai_service.core.errors import ConfigurationError, ProviderNotAvailableError
from ai_service.core.interfaces import AIProvider
from ai_service.core.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionRequest,
    CompletionResponse,
    EmbeddingRequest,
    EmbeddingResponse,
)
from ai_service.providers.openai import OpenAIProvider
from ai_service.providers.mock import MockProvider
from ai_service.utils.logging import setup_logging
from ai_service.utils.rate_limit import RateLimiter
from ai_service.utils.retry import RetryStrategy


class AIServiceClient:
    """
    Main client for interacting with AI providers.
    
    This client provides a simplified interface for using AI services
    and handles provider selection, configuration, and common functionality.
    """

    # Registry of available providers
    _provider_registry: Dict[str, Type[AIProvider]] = {
        "openai": OpenAIProvider,
        "mock": MockProvider,
    }
    
    # Default configuration
    _default_config = {
        "default_model": None,
        "timeout": 60.0,
        "retry_strategy": RetryStrategy(),
        "rate_limiter": RateLimiter(),
    }

    def __init__(
        self,
        provider: str = "openai",
        api_key: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        **kwargs: Any,
    ):
        """
        Initialize the AI service client.
        
        Args:
            provider: Name of the provider to use
            api_key: API key for the provider
            logger: Logger instance for logging client activity
            **kwargs: Additional provider-specific configuration options
        """
        self.logger = logger or setup_logging(module_name="ai_service.client")
        
        # Check that provider is supported
        if provider not in self._provider_registry:
            supported = ", ".join(self._provider_registry.keys())
            raise ProviderNotAvailableError(
                f"Provider '{provider}' not supported. Available providers: {supported}"
            )
        
        self.provider_name = provider
        
        # Check for API key if not using mock provider
        if provider != "mock" and not api_key:
            raise ConfigurationError(f"API key is required for provider '{provider}'")
        
        # Merge default config with kwargs
        config = {**self._default_config, **kwargs}
        
        # Initialize provider
        provider_class = self._provider_registry[provider]
        self.provider = provider_class(
            api_key=api_key or "mock-key",
            logger=self.logger,
            **config,
        )
        
        self.logger.info(f"Initialized AI service client with provider: {provider}")
    
    def create_completion(self, request: Union[CompletionRequest, str]) -> CompletionResponse:
        """
        Generate a text completion.
        
        Args:
            request: Completion request object or prompt string
            
        Returns:
            Completion response with the generated text
        """
        # Convert string to request if needed
        if isinstance(request, str):
            request = CompletionRequest(prompt=request)
        
        self.logger.info(f"Creating completion with provider: {self.provider_name}")
        return self.provider.create_completion(request)
    
    def create_chat_completion(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """
        Generate a chat completion.
        
        Args:
            request: Chat completion request
            
        Returns:
            Chat completion response with the generated message
        """
        self.logger.info(f"Creating chat completion with provider: {self.provider_name}")
        return self.provider.create_chat_completion(request)
    
    def create_embedding(self, request: Union[EmbeddingRequest, str, List[str]]) -> EmbeddingResponse:
        """
        Generate embeddings.
        
        Args:
            request: Embedding request object, text string, or list of strings
            
        Returns:
            Embedding response with the generated embeddings
        """
        # Convert string or list to request if needed
        if isinstance(request, (str, list)):
            request = EmbeddingRequest(input=request)
        
        self.logger.info(f"Creating embeddings with provider: {self.provider_name}")
        return self.provider.create_embedding(request)
    
    def get_token_count(self, text: str, model: Optional[str] = None) -> int:
        """
        Get the number of tokens in the given text.
        
        Args:
            text: Text to count tokens for
            model: Optional model to use for token counting
            
        Returns:
            Number of tokens in the text
        """
        return self.provider.get_token_count(text, model)
    
    def get_model_list(self) -> List[str]:
        """
        Get the list of available models.
        
        Returns:
            List of model identifiers
        """
        return self.provider.get_model_list()
    
    @classmethod
    def register_provider(cls, name: str, provider_class: Type[AIProvider]) -> None:
        """
        Register a new provider class.
        
        Args:
            name: Name to register the provider under
            provider_class: Provider class to register
        """
        cls._provider_registry[name] = provider_class