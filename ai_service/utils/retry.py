"""
Retry utilities for handling transient failures.

This module provides retry strategies and decorators for handling
transient failures in AI service requests.
"""

import logging
import time
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, List, Optional, Type, TypeVar, Union, cast

from ai_service.core.errors import (
    AIServiceError,
    AuthenticationError, 
    InvalidRequestError,
    RateLimitError,
)

# Type variable for retry decorator
F = TypeVar('F', bound=Callable[..., Any])


@dataclass
class RetryStrategy:
    """Configuration for retry behavior."""
    
    # Maximum number of retry attempts
    max_attempts: int = 3
    
    # Base delay between retries in seconds
    base_delay: float = 1.0
    
    # Maximum delay between retries in seconds
    max_delay: float = 60.0
    
    # Exponential backoff factor
    backoff_factor: float = 2.0
    
    # Jitter factor (0.0 for no jitter, 1.0 for full jitter)
    jitter: float = 0.1
    
    # List of exception types to retry
    retryable_exceptions: List[Type[Exception]] = field(
        default_factory=lambda: [RateLimitError, AIServiceError]
    )
    
    # List of exception types to never retry
    non_retryable_exceptions: List[Type[Exception]] = field(
        default_factory=lambda: [AuthenticationError, InvalidRequestError]
    )
    
    # Logger for retry events
    logger: Optional[logging.Logger] = None
    
    def __post_init__(self) -> None:
        """Initialize logger if not provided."""
        if self.logger is None:
            self.logger = logging.getLogger("ai_service.retry")
    
    def calculate_delay(self, attempt: int) -> float:
        """
        Calculate the delay before the next retry attempt.
        
        Args:
            attempt: The current attempt number (0-based)
            
        Returns:
            Delay in seconds
        """
        # Calculate exponential backoff
        delay = min(
            self.max_delay,
            self.base_delay * (self.backoff_factor ** attempt)
        )
        
        # Add jitter if configured
        if self.jitter > 0:
            import random
            jitter_amount = delay * self.jitter
            delay = delay - jitter_amount + (2 * jitter_amount * random.random())
        
        return delay
    
    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """
        Determine if a retry should be attempted based on the exception and attempt number.
        
        Args:
            exception: The exception that was raised
            attempt: The current attempt number (0-based)
            
        Returns:
            True if a retry should be attempted, False otherwise
        """
        # Check if we've reached the maximum number of attempts
        if attempt >= self.max_attempts:
            if self.logger:
                self.logger.debug(f"Not retrying after attempt {attempt}: max attempts reached")
            return False
        
        # Check if the exception is in the non-retryable list
        for exc_type in self.non_retryable_exceptions:
            if isinstance(exception, exc_type):
                if self.logger:
                    self.logger.debug(f"Not retrying after attempt {attempt}: {type(exception).__name__} is not retryable")
                return False
        
        # Check if the exception is in the retryable list
        for exc_type in self.retryable_exceptions:
            if isinstance(exception, exc_type):
                if self.logger:
                    self.logger.info(f"Retrying after attempt {attempt} due to {type(exception).__name__}")
                return True
        
        # If the exception is not explicitly categorized, don't retry
        if self.logger:
            self.logger.debug(f"Not retrying after attempt {attempt}: {type(exception).__name__} is not categorized")
        return False


def with_retry(strategy: Optional[RetryStrategy] = None) -> Callable[[F], F]:
    """
    Decorator to retry a function according to the specified strategy.
    
    Args:
        strategy: The retry strategy to use, or None to use defaults
        
    Returns:
        Decorated function with retry behavior
    """
    retry_strategy = strategy or RetryStrategy()
    
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception = None
            
            for attempt in range(retry_strategy.max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if not retry_strategy.should_retry(e, attempt):
                        break
                    
                    if attempt < retry_strategy.max_attempts - 1:
                        delay = retry_strategy.calculate_delay(attempt)
                        if retry_strategy.logger:
                            retry_strategy.logger.info(
                                f"Retrying {func.__name__} after {delay:.2f}s delay "
                                f"(attempt {attempt + 1}/{retry_strategy.max_attempts})"
                            )
                        time.sleep(delay)
            
            # If we get here, we've either exhausted our retries or hit a non-retryable exception
            assert last_exception is not None
            raise last_exception
            
        return cast(F, wrapper)
    
    return decorator