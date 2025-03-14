"""
Core interfaces for the AI service.

These abstract base classes define the contract that all AI providers must implement.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from ai_service.core.models import (
    CompletionRequest, 
    CompletionResponse,
    ChatCompletionRequest,
    ChatCompletionResponse,
    EmbeddingRequest,
    EmbeddingResponse
)


class AIProvider(ABC):
    """Abstract base class for AI providers."""

    @abstractmethod
    def create_completion(self, request: CompletionRequest) -> CompletionResponse:
        """
        Generate a text completion for the given prompt.
        
        Args:
            request: The completion request containing the prompt and parameters.
            
        Returns:
            A completion response containing the generated text.
            
        Raises:
            AIServiceError: If there's an error communicating with the provider.
            AuthenticationError: If authentication fails.
            RateLimitError: If the provider's rate limit is exceeded.
            InvalidRequestError: If the request is invalid.
        """
        pass
    
    @abstractmethod
    def create_chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """
        Generate a response for the given chat messages.
        
        Args:
            request: The chat completion request containing messages and parameters.
            
        Returns:
            A chat completion response containing the assistant's message.
            
        Raises:
            AIServiceError: If there's an error communicating with the provider.
            AuthenticationError: If authentication fails.
            RateLimitError: If the provider's rate limit is exceeded.
            InvalidRequestError: If the request is invalid.
        """
        pass
    
    @abstractmethod
    def create_embedding(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """
        Generate embeddings for the given texts.
        
        Args:
            request: The embedding request containing texts and parameters.
            
        Returns:
            An embedding response containing the generated embeddings.
            
        Raises:
            AIServiceError: If there's an error communicating with the provider.
            AuthenticationError: If authentication fails.
            RateLimitError: If the provider's rate limit is exceeded.
            InvalidRequestError: If the request is invalid.
        """
        pass
    
    @abstractmethod
    def get_model_list(self) -> List[str]:
        """
        Get the list of available models for this provider.
        
        Returns:
            A list of model identifiers as strings.
            
        Raises:
            AIServiceError: If there's an error communicating with the provider.
            AuthenticationError: If authentication fails.
        """
        pass
    
    @abstractmethod
    def get_token_count(self, text: str, model: Optional[str] = None) -> int:
        """
        Get the number of tokens in the given text for the specified model.
        
        Args:
            text: The text to count tokens for.
            model: Optional model to use for token counting.
                   If not provided, uses the provider's default model.
            
        Returns:
            The number of tokens in the text.
        """
        pass