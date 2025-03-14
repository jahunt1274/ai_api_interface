"""
Custom exception types for the AI service.

These exceptions provide a consistent error handling interface
across different providers.
"""

from typing import Any, Dict, Optional


class AIServiceError(Exception):
    """Base exception for all AI service errors."""
    
    def __init__(
        self, 
        message: str, 
        provider: Optional[str] = None, 
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.provider = provider
        self.details = details or {}
        super().__init__(message)


class AuthenticationError(AIServiceError):
    """Raised when authentication with the provider fails."""
    pass


class RateLimitError(AIServiceError):
    """Raised when a provider's rate limit is exceeded."""
    
    def __init__(
        self, 
        message: str, 
        provider: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        retry_after: Optional[int] = None
    ):
        super().__init__(message, provider, details)
        self.retry_after = retry_after


class InvalidRequestError(AIServiceError):
    """Raised when the request to the provider is invalid."""
    pass


class ProviderError(AIServiceError):
    """Raised when there's an error on the provider's side."""
    pass


class ProviderNotAvailableError(AIServiceError):
    """Raised when a provider is not available or not supported."""
    pass


class ModelNotAvailableError(AIServiceError):
    """Raised when a requested model is not available."""
    pass


class ConfigurationError(AIServiceError):
    """Raised when there's an error in the service configuration."""
    pass


class APITimeoutError(AIServiceError):
    """Raised when a request to the provider times out."""
    pass