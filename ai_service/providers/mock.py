"""
Mock provider implementation for testing.

This module provides a mock implementation of the AI provider interface
to be used in testing without making real API calls.
"""

import logging
import random
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from ai_service.core.errors import ModelNotAvailableError, RateLimitError
from ai_service.core.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionRequest,
    CompletionResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    Embedding,
    Message,
    Role,
    UsageInfo,
)
from ai_service.providers.base import BaseProvider


class MockProvider(BaseProvider):
    """Mock implementation of the AI provider interface for testing."""

    PROVIDER_NAME = "mock"

    def __init__(
        self,
        api_key: str = "mock-api-key",
        default_model: str = "mock-model",
        fail_rate: float = 0.0,
        rate_limit_rate: float = 0.0,
        response_delay: Optional[float] = None,
        logger: Optional[logging.Logger] = None,
        **kwargs: Any,
    ):
        """
        Initialize the mock provider.
        
        Args:
            api_key: Mock API key (not actually used)
            default_model: Default model to use if not specified in requests
            fail_rate: Probability of simulating a failure (0.0 to 1.0)
            rate_limit_rate: Probability of simulating a rate limit error (0.0 to 1.0)
            response_delay: Fixed delay in seconds, or None for random delay
            logger: Logger instance for logging provider activity
            **kwargs: Additional mock configuration options
        """
        super().__init__(
            api_key=api_key,
            default_model=default_model,
            logger=logger,
            **kwargs,
        )
        self.fail_rate = fail_rate
        self.rate_limit_rate = rate_limit_rate
        self.response_delay = response_delay
        
        # Mock responses for different prompts/messages
        self.mock_responses: Dict[str, str] = kwargs.get("mock_responses", {})
        
        # Available models
        self.available_models = kwargs.get(
            "available_models", 
            ["mock-model", "mock-gpt-4", "mock-gpt-3.5-turbo"]
        )
    
    def _maybe_fail(self) -> None:
        """
        Randomly fail based on configured fail_rate and rate_limit_rate.
        
        Raises:
            RateLimitError: If a rate limit error is simulated
            Exception: If a general failure is simulated
        """
        if random.random() < self.rate_limit_rate:
            retry_after = random.randint(1, 5)
            raise RateLimitError(
                message="Mock rate limit exceeded",
                provider=self.PROVIDER_NAME,
                retry_after=retry_after,
            )
        
        if random.random() < self.fail_rate:
            raise Exception("Mock failure")
    
    def _simulate_delay(self) -> None:
        """Simulate a delay in the response."""
        if self.response_delay is not None:
            time.sleep(self.response_delay)
        else:
            # Random delay between 0.1 and 1.0 seconds
            time.sleep(random.uniform(0.1, 1.0))
    
    def _generate_mock_usage(self, input_length: int, output_length: int) -> UsageInfo:
        """
        Generate mock usage information.
        
        Args:
            input_length: Length of the input (prompt or messages)
            output_length: Length of the generated output
            
        Returns:
            Mock usage information
        """
        # Approximate token counts based on text length
        prompt_tokens = max(1, input_length // 4)
        completion_tokens = max(1, output_length // 4)
        
        return UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
    
    def _get_mock_response(self, text: str) -> str:
        """
        Get a mock response for the given text.
        
        If the text matches a key in mock_responses, return the corresponding value.
        Otherwise, generate a generic response.
        
        Args:
            text: Input text (prompt or message content)
            
        Returns:
            Mock response text
        """
        # Check for exact matches
        if text in self.mock_responses:
            return self.mock_responses[text]
        
        # Check for partial matches
        for key, value in self.mock_responses.items():
            if key in text:
                return value
        
        # Generate a generic response
        return f"This is a mock response to: {text[:50]}..."
    
    def create_completion(self, request: CompletionRequest) -> CompletionResponse:
        """
        Generate a mock text completion.
        
        Args:
            request: The completion request
            
        Returns:
            A mock completion response
            
        Raises:
            Various errors based on configured fail_rate and rate_limit_rate
        """
        self._log_request("completions", request)
        self._maybe_fail()
        self._simulate_delay()
        
        model = self._get_model(request.model)
        
        # Generate mock completion
        completion = self._get_mock_response(request.prompt)
        
        # Generate mock usage
        usage = self._generate_mock_usage(len(request.prompt), len(completion))
        
        response = CompletionResponse(
            completion=completion,
            model=model,
            usage=usage,
            provider=self.PROVIDER_NAME,
            created_at=datetime.utcnow(),
            provider_response={"mock": True},
        )
        
        self._log_response("completions", response)
        return response
    
    def create_chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """
        Generate a mock chat completion.
        
        Args:
            request: The chat completion request
            
        Returns:
            A mock chat completion response
            
        Raises:
            Various errors based on configured fail_rate and rate_limit_rate
        """
        self._log_request("chat/completions", request)
        self._maybe_fail()
        self._simulate_delay()
        
        model = self._get_model(request.model)
        
        # Extract the last user message
        last_user_message = None
        for message in reversed(request.messages):
            if message.role == Role.USER:
                last_user_message = message.content
                break
        
        # Generate mock response
        if last_user_message:
            content = self._get_mock_response(last_user_message)
        else:
            content = "I don't have a specific message to respond to."
        
        # Calculate input length (all messages combined)
        input_length = sum(len(msg.content) for msg in request.messages)
        
        # Generate mock usage
        usage = self._generate_mock_usage(input_length, len(content))
        
        response = ChatCompletionResponse(
            message=Message(role=Role.ASSISTANT, content=content),
            model=model,
            usage=usage,
            provider=self.PROVIDER_NAME,
            created_at=datetime.utcnow(),
            provider_response={"mock": True},
        )
        
        self._log_response("chat/completions", response)
        return response
    
    def create_embedding(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """
        Generate mock embeddings.
        
        Args:
            request: The embedding request
            
        Returns:
            A mock embedding response
            
        Raises:
            Various errors based on configured fail_rate and rate_limit_rate
        """
        self._log_request("embeddings", request)
        self._maybe_fail()
        self._simulate_delay()
        
        model = self._get_model(request.model)
        
        # Convert input to list if it's a string
        input_texts = request.input if isinstance(request.input, list) else [request.input]
        
        # Generate mock embeddings
        embeddings = []
        total_length = 0
        
        for i, text in enumerate(input_texts):
            # Generate a deterministic but seemingly random embedding
            # The hash ensures the same text always gets the same embedding
            seed = hash(text) % 10000
            random.seed(seed)
            
            # Create a mock embedding of length 1536 (like OpenAI's ada-002)
            vector = [random.uniform(-1, 1) for _ in range(1536)]
            
            # Normalize the vector
            magnitude = sum(x*x for x in vector) ** 0.5
            normalized = [x/magnitude for x in vector]
            
            embeddings.append(
                Embedding(
                    embedding=normalized,
                    index=i,
                    object="embedding",
                )
            )
            
            total_length += len(text)
        
        # Generate mock usage
        usage = self._generate_mock_usage(total_length, 0)
        
        response = EmbeddingResponse(
            embeddings=embeddings,
            model=model,
            usage=usage,
            provider=self.PROVIDER_NAME,
            created_at=datetime.utcnow(),
            provider_response={"mock": True},
        )
        
        self._log_response("embeddings", response)
        return response
    
    def get_model_list(self) -> List[str]:
        """
        Get the list of available models.
        
        Returns:
            List of mock model identifiers
            
        Raises:
            Various errors based on configured fail_rate
        """
        self._maybe_fail()
        return self.available_models
    
    def get_token_count(self, text: str, model: Optional[str] = None) -> int:
        """
        Get the number of tokens in the given text.
        
        Args:
            text: The text to count tokens for
            model: Optional model to use for token counting
            
        Returns:
            Approximate number of tokens in the text
            
        Raises:
            ModelNotAvailableError: If the model is not in the available models list
        """
        model_name = self._get_model(model)
        
        if model_name not in self.available_models:
            raise ModelNotAvailableError(
                message=f"Model {model_name} not available",
                provider=self.PROVIDER_NAME,
            )
        
        # Simple approximation: 1 token ≈ 4 characters
        return max(1, len(text) // 4)