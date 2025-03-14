"""
Base implementation for AI providers.

This module contains a base class with common functionality that
can be shared across different provider implementations.
"""

import logging
from abc import ABC
from typing import Any, Dict, List, Optional, Union

from ai_service.core.errors import ConfigurationError
from ai_service.core.interfaces import AIProvider
from ai_service.core.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionRequest,
    CompletionResponse,
    EmbeddingRequest,
    EmbeddingResponse,
)
from ai_service.utils.retry import RetryStrategy


class BaseProvider(AIProvider, ABC):
    """Base implementation with common functionality for AI providers."""

    def __init__(
        self,
        api_key: str,
        default_model: Optional[str] = None,
        retry_strategy: Optional[RetryStrategy] = None,
        logger: Optional[logging.Logger] = None,
        **kwargs: Any,
    ):
        """
        Initialize the base provider.
        
        Args:
            api_key: API key for authentication with the provider
            default_model: Default model to use if not specified in requests
            retry_strategy: Strategy for retrying failed requests
            logger: Logger instance for logging provider activity
            **kwargs: Additional provider-specific configuration options
        """
        if not api_key:
            raise ConfigurationError("API key is required")
            
        self.api_key = api_key
        self.default_model = default_model
        self.retry_strategy = retry_strategy
        self.logger = logger or logging.getLogger(f"ai_service.provider.{self.__class__.__name__}")
        self.config = kwargs
    
    def _log_request(self, endpoint: str, request: Any) -> None:
        """
        Log information about an outgoing request.
        
        Args:
            endpoint: The API endpoint being called
            request: The request being sent
        """
        if self.logger.isEnabledFor(logging.DEBUG):
            # Create a sanitized version of the request for logging
            sanitized = self._sanitize_request(request)
            self.logger.debug(f"Request to {endpoint}: {sanitized}")
    
    def _log_response(self, endpoint: str, response: Any) -> None:
        """
        Log information about a received response.
        
        Args:
            endpoint: The API endpoint that was called
            response: The response received
        """
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"Response from {endpoint}: {response}")
    
    def _sanitize_request(self, request: Any) -> Dict[str, Any]:
        """
        Create a sanitized version of a request for logging.
        
        This removes sensitive fields like full prompts and replaces them
        with length indicators.
        
        Args:
            request: The request to sanitize
            
        Returns:
            Sanitized request data safe for logging
        """
        if hasattr(request, "dict"):
            # For Pydantic models
            data = request.dict()
        elif isinstance(request, dict):
            # For regular dictionaries
            data = request.copy()
        else:
            # For other types, just use string representation
            return str(request)
        
        # Sanitize prompt and messages
        if "prompt" in data and isinstance(data["prompt"], str):
            data["prompt"] = f"<prompt of length {len(data['prompt'])}>"
        
        if "messages" in data and isinstance(data["messages"], list):
            data["messages"] = f"<{len(data['messages'])} messages>"
            
        if "input" in data:
            if isinstance(data["input"], str):
                data["input"] = f"<input of length {len(data['input'])}>"
            elif isinstance(data["input"], list):
                data["input"] = f"<list of {len(data['input'])} inputs>"
        
        return data
    
    def _get_model(self, model: Optional[str] = None) -> str:
        """
        Get the model to use, falling back to the default if none is specified.
        
        Args:
            model: Model specified in the request, or None
            
        Returns:
            The model to use
            
        Raises:
            ConfigurationError: If no model is specified and no default is configured
        """
        resolved_model = model or self.default_model
        if not resolved_model:
            raise ConfigurationError(
                "No model specified and no default model configured"
            )
        return resolved_model