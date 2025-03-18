"""
OpenAI provider implementation.

This module provides an implementation of the AI service interface
using the OpenAI API.
"""

import logging
import tiktoken
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, cast

import openai
from openai import OpenAI
from openai.types.chat import ChatCompletionMessage
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

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
from ai_service.utils.retry import RetryStrategy


class OpenAIProvider(BaseProvider):
    """Implementation of the AI provider interface for OpenAI."""

    PROVIDER_NAME = "openai"

    def __init__(
        self,
        api_key: str,
        default_model: Optional[str] = "gpt-3.5-turbo",
        organization: Optional[str] = None,
        timeout: Optional[float] = 60.0,
        retry_strategy: Optional[RetryStrategy] = None,
        logger: Optional[logging.Logger] = None,
        **kwargs: Any,
    ):
        """
        Initialize the OpenAI provider.
        
        Args:
            api_key: OpenAI API key
            default_model: Default model to use if not specified in requests
            organization: OpenAI organization ID (optional)
            timeout: Timeout for API requests in seconds
            retry_strategy: Strategy for retrying failed requests
            logger: Logger instance for logging provider activity
            **kwargs: Additional provider-specific configuration options
        """
        super().__init__(
            api_key=api_key,
            default_model=default_model,
            retry_strategy=retry_strategy,
            logger=logger,
            **kwargs,
        )
        self.timeout = timeout
        self.organization = organization

        # Filter out ai_service specific parameters that the OpenAI client doesn't need
        openai_kwargs = {k: v for k, v in kwargs.items() 
                         if k not in ['rate_limiter', 'retry_strategy']}
        
        # Initialize the OpenAI client
        self.client = OpenAI(
            api_key=self.api_key,
            organization=self.organization,
            timeout=self.timeout,
            # **kwargs,
            **openai_kwargs,
        )
        
        # Cache for available models
        self._available_models_cache: Optional[List[str]] = None
    
    def create_completion(self, request: CompletionRequest) -> CompletionResponse:
        """
        Generate a text completion using the OpenAI API.
        
        Args:
            request: The completion request
            
        Returns:
            A completion response with the generated text
            
        Raises:
            Various AIServiceError subclasses based on the error encountered
        """
        model = self._get_model(request.model)
        self._log_request("completions", request)
        
        try:
            # For newer models, use chat completions as text completions
            if model.startswith(("gpt-3.5", "gpt-4")):
                # Convert to chat format
                chat_request = ChatCompletionRequest(
                    messages=[Message(role=Role.USER, content=request.prompt)],
                    model=model,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    n=request.n,
                    stream=request.stream,
                    stop=request.stop,
                    extra_params=request.extra_params,
                )
                chat_response = self.create_chat_completion(chat_request)
                
                # Convert chat response to completion response
                response = CompletionResponse(
                    completion=chat_response.message.content,
                    model=chat_response.model,
                    usage=chat_response.usage,
                    provider=self.PROVIDER_NAME,
                    created_at=chat_response.created_at,
                    provider_response=chat_response.provider_response,
                )
            else:
                # Legacy models use the completions endpoint
                params = {
                    "model": model,
                    "prompt": request.prompt,
                    "max_tokens": request.max_tokens,
                    "temperature": request.temperature,
                    "top_p": request.top_p,
                    "n": request.n,
                    "stream": request.stream,
                    "stop": request.stop,
                    **request.extra_params,
                }
                
                openai_response = self.client.completions.create(**params)
                
                response = CompletionResponse(
                    completion=openai_response.choices[0].text,
                    model=model,
                    usage=UsageInfo(
                        prompt_tokens=openai_response.usage.prompt_tokens,
                        completion_tokens=openai_response.usage.completion_tokens,
                        total_tokens=openai_response.usage.total_tokens,
                    ),
                    provider=self.PROVIDER_NAME,
                    provider_response=openai_response.model_dump(),
                )
            
            self._log_response("completions", response)
            return response
            
        except openai.AuthenticationError as e:
            raise AuthenticationError(
                message=str(e),
                provider=self.PROVIDER_NAME,
                details={"original_error": str(e)},
            )
        except openai.RateLimitError as e:
            raise RateLimitError(
                message=str(e),
                provider=self.PROVIDER_NAME,
                details={"original_error": str(e)},
            )
        except openai.BadRequestError as e:
            raise InvalidRequestError(
                message=str(e),
                provider=self.PROVIDER_NAME,
                details={"original_error": str(e)},
            )
        except openai.APIError as e:
            raise ProviderError(
                message=str(e),
                provider=self.PROVIDER_NAME,
                details={"original_error": str(e)},
            )
        except Exception as e:
            raise AIServiceError(
                message=f"Unexpected error: {str(e)}",
                provider=self.PROVIDER_NAME,
                details={"original_error": str(e)},
            )
    
    def create_embedding(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """
        Generate embeddings using the OpenAI API.
        
        Args:
            request: The embedding request
            
        Returns:
            An embedding response with the generated embeddings
            
        Raises:
            Various AIServiceError subclasses based on the error encountered
        """
        model = self._get_model(request.model)
        self._log_request("embeddings", request)
        
        try:
            params = {
                "model": model,
                "input": request.input,
                **request.extra_params,
            }
            
            openai_response = self.client.embeddings.create(**params)
            
            # Convert to our embedding model
            embeddings = []
            for i, embed_data in enumerate(openai_response.data):
                embeddings.append(
                    Embedding(
                        embedding=embed_data.embedding,
                        index=embed_data.index,
                        object=embed_data.object,
                    )
                )
            
            response = EmbeddingResponse(
                embeddings=embeddings,
                model=model,
                usage=UsageInfo(
                    prompt_tokens=openai_response.usage.prompt_tokens,
                    # OpenAI embeddings don't return completion tokens
                    completion_tokens=0,
                    total_tokens=openai_response.usage.total_tokens,
                ),
                provider=self.PROVIDER_NAME,
                provider_response=openai_response.model_dump(),
            )
            
            self._log_response("embeddings", response)
            return response
            
        except openai.AuthenticationError as e:
            raise AuthenticationError(
                message=str(e),
                provider=self.PROVIDER_NAME,
                details={"original_error": str(e)},
            )
        except openai.RateLimitError as e:
            raise RateLimitError(
                message=str(e),
                provider=self.PROVIDER_NAME,
                details={"original_error": str(e)},
            )
        except openai.BadRequestError as e:
            raise InvalidRequestError(
                message=str(e),
                provider=self.PROVIDER_NAME,
                details={"original_error": str(e)},
            )
        except openai.APIError as e:
            raise ProviderError(
                message=str(e),
                provider=self.PROVIDER_NAME,
                details={"original_error": str(e)},
            )
        except Exception as e:
            raise AIServiceError(
                message=f"Unexpected error: {str(e)}",
                provider=self.PROVIDER_NAME,
                details={"original_error": str(e)},
            )
    
    def get_model_list(self) -> List[str]:
        """
        Get the list of available models from the OpenAI API.
        
        Returns:
            A list of model identifiers
            
        Raises:
            Various AIServiceError subclasses based on the error encountered
        """
        # Use cached list if available
        if self._available_models_cache is not None:
            return self._available_models_cache
        
        try:
            response = self.client.models.list()
            self._available_models_cache = [model.id for model in response.data]
            return self._available_models_cache
            
        except openai.AuthenticationError as e:
            raise AuthenticationError(
                message=str(e),
                provider=self.PROVIDER_NAME,
                details={"original_error": str(e)},
            )
        except openai.APIError as e:
            raise ProviderError(
                message=str(e),
                provider=self.PROVIDER_NAME,
                details={"original_error": str(e)},
            )
        except Exception as e:
            raise AIServiceError(
                message=f"Unexpected error: {str(e)}",
                provider=self.PROVIDER_NAME,
                details={"original_error": str(e)},
            )
    
    def get_token_count(self, text: str, model: Optional[str] = None) -> int:
        """
        Get the number of tokens in the given text for the specified model.
        
        Args:
            text: The text to count tokens for
            model: Optional model to use for token counting
            
        Returns:
            The number of tokens in the text
            
        Raises:
            ModelNotAvailableError: If the model or its tokenizer is not available
        """
        model_name = self._get_model(model)
        
        try:
            # Determine which tokenizer to use based on the model
            if model_name.startswith("gpt-4"):
                encoding = tiktoken.encoding_for_model("gpt-4")
            elif model_name.startswith("gpt-3.5-turbo"):
                encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
            elif model_name.startswith("text-embedding"):
                encoding = tiktoken.encoding_for_model("text-embedding-ada-002")
            else:
                # Fall back to cl100k_base for newer models
                encoding = tiktoken.get_encoding("cl100k_base")
            
            # Count tokens
            tokens = encoding.encode(text)
            return len(tokens)
            
        except Exception as e:
            raise ModelNotAvailableError(
                message=f"Failed to count tokens for model {model_name}: {str(e)}",
                provider=self.PROVIDER_NAME,
                details={"original_error": str(e)},
            )
    
    def create_chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """
        Generate a chat completion using the OpenAI API.
        
        Args:
            request: The chat completion request
            
        Returns:
            A chat completion response with the generated message
            
        Raises:
            Various AIServiceError subclasses based on the error encountered
        """
        model = self._get_model(request.model)
        self._log_request("chat/completions", request)
        
        try:
            # Convert messages to OpenAI format
            openai_messages = [
                {"role": msg.role.value, "content": msg.content, **({"name": msg.name} if msg.name else {})}
                for msg in request.messages
            ]
            
            params = {
                "model": model,
                "messages": openai_messages,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "n": request.n,
                "stream": request.stream,
                "stop": request.stop,
                **request.extra_params,
            }
            
            openai_response = self.client.chat.completions.create(**params)
            
            # Get the assistant's message
            assistant_message = openai_response.choices[0].message
            
            response = ChatCompletionResponse(
                message=Message(
                    role=Role(assistant_message.role),
                    content=assistant_message.content or "",
                    name=getattr(assistant_message, "name", None),
                ),
                model=model,
                usage=UsageInfo(
                    prompt_tokens=openai_response.usage.prompt_tokens,
                    completion_tokens=openai_response.usage.completion_tokens,
                    total_tokens=openai_response.usage.total_tokens,
                ),
                provider=self.PROVIDER_NAME,
                provider_response=openai_response.model_dump(),
            )
            
            self._log_response("chat/completions", response)
            return response
            
        except openai.AuthenticationError as e:
            raise AuthenticationError(
                message=str(e),
                provider=self.PROVIDER_NAME,
                details={"original_error": str(e)},
            )
        except openai.RateLimitError as e:
            raise RateLimitError(
                message=str(e),
                provider=self.PROVIDER_NAME,
                details={"original_error": str(e)},
            )
        except openai.BadRequestError as e:
            raise InvalidRequestError(
                message=str(e),
                provider=self.PROVIDER_NAME,
                details={"original_error": str(e)},
            )
        except openai.APIError as e:
            raise ProviderError(
                message=str(e),
                provider=self.PROVIDER_NAME,
                details={"original_error": str(e)},
            )
        except Exception as e:
            raise AIServiceError(
                message=f"Unexpected error: {str(e)}",
                provider=self.PROVIDER_NAME,
                details={"original_error": str(e)},
            )