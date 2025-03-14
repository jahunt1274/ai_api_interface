"""
Rate limiting utilities for the AI service.

This module provides rate limiting mechanisms to control the rate
of requests to AI providers.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar, cast, Union

from ai_service.core.errors import RateLimitError

# Type variable for rate limit decorator
F = TypeVar('F', bound=Callable[..., Any])


@dataclass
class TokenBucket:
    """
    Token bucket rate limiter implementation.
    
    This classic rate limiting algorithm uses tokens that are added to a bucket
    at a constant rate. Each request consumes one or more tokens, and if there
    are not enough tokens, the request is delayed or rejected.
    """
    
    # Maximum number of tokens the bucket can hold
    capacity: float
    
    # Rate at which tokens are added to the bucket (tokens per second)
    refill_rate: float
    
    # Current number of tokens in the bucket
    tokens: float = field(default=0.0)
    
    # Last time the bucket was refilled
    last_refill: float = field(default_factory=time.time)
    
    # Lock for thread safety
    _lock: threading.RLock = field(default_factory=threading.RLock)
    
    def __post_init__(self) -> None:
        """Initialize the token bucket with full capacity."""
        self.tokens = self.capacity
    
    def _refill(self) -> None:
        """Refill the bucket based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        
        # Calculate how many tokens to add based on elapsed time
        new_tokens = elapsed * self.refill_rate
        
        # Update token count, not exceeding capacity
        self.tokens = min(self.capacity, self.tokens + new_tokens)
        
        # Update last refill time
        self.last_refill = now
    
    def try_consume(self, tokens: float = 1.0) -> bool:
        """
        Try to consume tokens from the bucket.
        
        Args:
            tokens: Number of tokens to consume
            
        Returns:
            True if tokens were consumed, False if not enough tokens
        """
        with self._lock:
            self._refill()
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            
            return False
    
    def consume(self, tokens: float = 1.0, wait: bool = True, max_wait: Optional[float] = None) -> float:
        """
        Consume tokens from the bucket, waiting if necessary.
        
        Args:
            tokens: Number of tokens to consume
            wait: Whether to wait if not enough tokens are available
            max_wait: Maximum time to wait in seconds, or None for no limit
            
        Returns:
            Wait time in seconds (0 if no wait)
            
        Raises:
            RateLimitError: If not enough tokens and wait is False or max_wait exceeded
        """
        start_time = time.time()
        
        with self._lock:
            self._refill()
            
            # If we have enough tokens, consume them immediately
            if self.tokens >= tokens:
                self.tokens -= tokens
                return 0.0
            
            # If we can't wait, raise an error
            if not wait:
                raise RateLimitError(
                    message="Rate limit exceeded",
                    details={"tokens_available": self.tokens, "tokens_requested": tokens}
                )
            
            # Calculate how long to wait until we have enough tokens
            wait_time = (tokens - self.tokens) / self.refill_rate
            
            # If max_wait is specified and we would exceed it, raise an error
            if max_wait is not None and wait_time > max_wait:
                raise RateLimitError(
                    message="Rate limit exceeded and max wait time would be exceeded",
                    details={
                        "tokens_available": self.tokens,
                        "tokens_requested": tokens,
                        "wait_required": wait_time,
                        "max_wait": max_wait
                    },
                    retry_after=int(wait_time)
                )
            
            # Wait for tokens to become available
            time.sleep(wait_time)
            
            # Now consume the tokens
            self.tokens = self.capacity - tokens
            self.last_refill = time.time()
            
            return time.time() - start_time


@dataclass
class RateLimiter:
    """
    Rate limiter for AI service requests.
    
    This provides a more sophisticated rate limiting implementation that
    can handle different limits for different models and endpoints.
    """
    
    # Default token bucket settings
    default_capacity: float = 60.0
    default_refill_rate: float = 1.0
    
    # Token buckets by model
    model_buckets: Dict[str, TokenBucket] = field(default_factory=dict)
    
    # Global token bucket for all requests
    global_bucket: Optional[TokenBucket] = None
    
    # Logger for rate limit events
    logger: Optional[logging.Logger] = None
    
    # Lock for thread safety
    _lock: threading.RLock = field(default_factory=threading.RLock)
    
    def __post_init__(self) -> None:
        """Initialize logger if not provided."""
        if self.logger is None:
            self.logger = logging.getLogger("ai_service.rate_limit")
            
        if self.global_bucket is None:
            self.global_bucket = TokenBucket(
                capacity=self.default_capacity,
                refill_rate=self.default_refill_rate
            )
    
    def add_model_limit(self, model: str, capacity: float, refill_rate: float) -> None:
        """
        Add a rate limit for a specific model.
        
        Args:
            model: The model identifier
            capacity: Maximum tokens (requests) the bucket can hold
            refill_rate: Rate at which tokens are added (tokens per second)
        """
        with self._lock:
            self.model_buckets[model] = TokenBucket(
                capacity=capacity,
                refill_rate=refill_rate
            )
    
    def get_bucket_for_model(self, model: str) -> TokenBucket:
        """
        Get the token bucket for a specific model.
        
        Args:
            model: The model identifier
            
        Returns:
            The token bucket for the model, or the default bucket if not found
        """
        assert self.global_bucket is not None
        
        with self._lock:
            return self.model_buckets.get(model, self.global_bucket)
    
    def try_acquire(self, model: Optional[str] = None, tokens: float = 1.0) -> bool:
        """
        Try to acquire tokens for a request, not waiting if not available.
        
        Args:
            model: The model being used, or None
            tokens: Number of tokens to consume
            
        Returns:
            True if tokens were acquired, False otherwise
        """
        assert self.global_bucket is not None
        
        # Always check the global bucket first
        if not self.global_bucket.try_consume(tokens):
            return False
        
        # If a model is specified, check the model-specific bucket
        if model is not None:
            model_bucket = self.get_bucket_for_model(model)
            if not model_bucket.try_consume(tokens):
                # Refund tokens to the global bucket
                self.global_bucket.tokens += tokens
                return False
        
        return True
    
    def acquire(
        self,
        model: Optional[str] = None,
        tokens: float = 1.0,
        wait: bool = True,
        max_wait: Optional[float] = None
    ) -> float:
        """
        Acquire tokens for a request, optionally waiting if not available.
        
        Args:
            model: The model being used, or None
            tokens: Number of tokens to consume
            wait: Whether to wait if not enough tokens are available
            max_wait: Maximum time to wait in seconds, or None for no limit
            
        Returns:
            Wait time in seconds (0 if no wait)
            
        Raises:
            RateLimitError: If not enough tokens and wait is False or max_wait exceeded
        """
        assert self.global_bucket is not None
        
        start_time = time.time()
        model_str = model or "global"
        
        try:
            # Always check the global bucket first
            wait_time = self.global_bucket.consume(tokens, wait, max_wait)
            
            # If a model is specified, check the model-specific bucket
            if model is not None:
                model_bucket = self.get_bucket_for_model(model)
                try:
                    model_wait_time = model_bucket.consume(tokens, wait, max_wait)
                    wait_time += model_wait_time
                except RateLimitError:
                    # Refund tokens to the global bucket
                    self.global_bucket.tokens += tokens
                    raise
            
            # Log the rate limit event
            if wait_time > 0 and self.logger:
                self.logger.info(
                    f"Rate limit: waited {wait_time:.2f}s for {tokens} tokens "
                    f"for model {model_str}"
                )
            
            return wait_time
            
        except RateLimitError as e:
            if self.logger:
                self.logger.warning(
                    f"Rate limit exceeded for model {model_str}: {e.message}"
                )
            raise


def with_rate_limit(
    rate_limiter: RateLimiter,
    tokens: float = 1.0,
    model_arg_name: str = "model",
    wait: bool = True,
    max_wait: Optional[float] = None
) -> Callable[[F], F]:
    """
    Decorator to apply rate limiting to a function.
    
    Args:
        rate_limiter: The rate limiter to use
        tokens: Number of tokens to consume per call
        model_arg_name: Name of the argument or attribute that contains the model name
        wait: Whether to wait if not enough tokens are available
        max_wait: Maximum time to wait in seconds, or None for no limit
        
    Returns:
        Decorated function with rate limiting
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Try to get the model name from kwargs
            model = kwargs.get(model_arg_name)
            
            # If not found in kwargs, try to get from the first argument (assuming it's a request)
            if model is None and args and hasattr(args[0], model_arg_name):
                model = getattr(args[0], model_arg_name)
            
            # Acquire tokens, waiting if necessary
            rate_limiter.acquire(model, tokens, wait, max_wait)
            
            # Call the original function
            return func(*args, **kwargs)
            
        return cast(F, wrapper)
    
    return decorator