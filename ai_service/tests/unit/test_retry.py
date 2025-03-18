"""
Unit tests for retry utilities.

This module contains unit tests for the retry strategies and decorators.
"""

import logging
import pytest
import time
from unittest.mock import MagicMock, patch, call

from ai_service.core.errors import (
    AIServiceError,
    AuthenticationError,
    InvalidRequestError,
    RateLimitError,
)
from ai_service.utils.retry import RetryStrategy, with_retry


class TestRetryStrategy:
    """Tests for the RetryStrategy class."""

    def test_initialization(self):
        """Test that RetryStrategy initializes with default values."""
        strategy = RetryStrategy()
        assert strategy.max_attempts == 3
        assert strategy.base_delay == 1.0
        assert strategy.max_delay == 60.0
        assert strategy.backoff_factor == 2.0
        assert strategy.jitter == 0.1
        assert RateLimitError in strategy.retryable_exceptions
        assert AIServiceError in strategy.retryable_exceptions
        assert AuthenticationError in strategy.non_retryable_exceptions
        assert InvalidRequestError in strategy.non_retryable_exceptions
        assert strategy.logger is not None

    def test_initialization_with_custom_values(self):
        """Test that RetryStrategy can be initialized with custom values."""
        strategy = RetryStrategy(
            max_attempts=5,
            base_delay=2.0,
            max_delay=30.0,
            backoff_factor=3.0,
            jitter=0.2
        )
        assert strategy.max_attempts == 5
        assert strategy.base_delay == 2.0
        assert strategy.max_delay == 30.0
        assert strategy.backoff_factor == 3.0
        assert strategy.jitter == 0.2

    def test_calculate_delay(self):
        """Test calculating delay between retries."""
        strategy = RetryStrategy(
            base_delay=1.0,
            max_delay=10.0,
            backoff_factor=2.0,
            jitter=0.0  # Disable jitter for deterministic testing
        )
        
        # First attempt (0-based) should use base_delay
        assert strategy.calculate_delay(0) == 1.0
        
        # Second attempt should use base_delay * backoff_factor
        assert strategy.calculate_delay(1) == 2.0
        
        # Third attempt should use base_delay * backoff_factor^2
        assert strategy.calculate_delay(2) == 4.0
        
        # Fourth attempt should use base_delay * backoff_factor^3, but capped at max_delay
        assert strategy.calculate_delay(3) == 8.0
        
        # Fifth attempt should be capped at max_delay
        assert strategy.calculate_delay(4) == 10.0

    def test_calculate_delay_with_jitter(self):
        """Test that jitter adds randomness to the delay."""
        strategy = RetryStrategy(
            base_delay=1.0,
            backoff_factor=1.0,  # Disable backoff for clearer testing
            jitter=0.5  # 50% jitter
        )
        
        # Since jitter is random, we can only check the range
        delay = strategy.calculate_delay(0)
        assert 0.5 <= delay <= 1.5  # base_delay ± 50%

    def test_should_retry_max_attempts(self):
        """Test that should_retry returns False after max_attempts."""
        strategy = RetryStrategy(max_attempts=3)
        
        # First three attempts should be retryable for a retryable exception
        assert strategy.should_retry(RateLimitError("test"), 0) is True
        assert strategy.should_retry(RateLimitError("test"), 1) is True
        assert strategy.should_retry(RateLimitError("test"), 2) is True
        
        # Fourth attempt should not be retried
        assert strategy.should_retry(RateLimitError("test"), 3) is False

    def test_should_retry_non_retryable_exception(self):
        """Test that should_retry returns False for non-retryable exceptions."""
        strategy = RetryStrategy()
        
        # Authentication errors should not be retried
        assert strategy.should_retry(AuthenticationError("test"), 0) is False
        
        # Invalid request errors should not be retried
        assert strategy.should_retry(InvalidRequestError("test"), 0) is False
        
        # Other exceptions that aren't explicitly categorized should not be retried
        assert strategy.should_retry(ValueError("test"), 0) is False

    def test_should_retry_retryable_exception(self):
        """Test that should_retry returns True for retryable exceptions."""
        strategy = RetryStrategy()
        
        # Rate limit errors should be retried
        assert strategy.should_retry(RateLimitError("test"), 0) is True
        
        # Generic AI service errors should be retried
        assert strategy.should_retry(AIServiceError("test"), 0) is True
        
        # Custom retryable exceptions should be retried
        strategy = RetryStrategy(retryable_exceptions=[ValueError])
        assert strategy.should_retry(ValueError("test"), 0) is True


class TestWithRetry:
    """Tests for the with_retry decorator."""

    def test_successful_execution(self):
        """Test that the decorator doesn't interfere with successful execution."""
        mock_func = MagicMock(return_value="success")
        decorated = with_retry()(mock_func)
        
        result = decorated("arg1", kwarg1="value1")
        
        assert result == "success"
        mock_func.assert_called_once_with("arg1", kwarg1="value1")

    def test_retry_on_exception(self):
        """Test that the function is retried on retryable exceptions."""
        mock_func = MagicMock(side_effect=[
            RateLimitError("rate limited"),
            AIServiceError("service error"),
            "success"
        ])
        mock_func.__name__ = "mock_func"
        
        # Create a strategy with no delay for faster testing
        strategy = RetryStrategy(base_delay=0.01, max_delay=0.01, jitter=0)
        decorated = with_retry(strategy)(mock_func)
        
        result = decorated("arg")
        
        assert result == "success"
        assert mock_func.call_count == 3
        
        # Check that all calls were with the same arguments
        for i in range(3):
            assert mock_func.call_args_list[i] == call("arg")

    def test_no_retry_on_non_retryable_exception(self):
        """Test that the function is not retried on non-retryable exceptions."""
        mock_func = MagicMock(side_effect=AuthenticationError("auth error"))
        decorated = with_retry()(mock_func)
        
        with pytest.raises(AuthenticationError) as exc_info:
            decorated()
        
        assert "auth error" in str(exc_info.value)
        mock_func.assert_called_once()

    def test_max_retries_exceeded(self):
        """Test that the function stops retrying after max_attempts."""
        # Create a function that always fails with a retryable error
        mock_func = MagicMock(side_effect=RateLimitError("rate limited"))
        mock_func.__name__ = "mock_func"
        
        # Create a strategy with 2 max attempts and no delay for faster testing
        strategy = RetryStrategy(max_attempts=2, base_delay=0.01, jitter=0)
        decorated = with_retry(strategy)(mock_func)
        
        with pytest.raises(RateLimitError) as exc_info:
            decorated()
        
        assert "rate limited" in str(exc_info.value)
        assert mock_func.call_count == 2  # Initial attempt + 1 retry

    @patch("time.sleep")
    def test_exponential_backoff(self, mock_sleep):
        """Test that retry uses exponential backoff."""
        # Create a function that always fails with a retryable error
        mock_func = MagicMock(side_effect=RateLimitError("rate limited"))
        mock_func.__name__ = "mock_func"
        
        # Create a strategy with specific parameters and no jitter
        strategy = RetryStrategy(
            max_attempts=3,
            base_delay=1.0,
            backoff_factor=2.0,
            jitter=0.0
        )
        decorated = with_retry(strategy)(mock_func)
        
        with pytest.raises(RateLimitError):
            decorated()
        
        # Check that sleep was called with exponentially increasing delays
        assert mock_sleep.call_count == 2  # Called after first and second attempts
        assert mock_sleep.call_args_list[0] == call(1.0)  # First retry: base_delay
        assert mock_sleep.call_args_list[1] == call(2.0)  # Second retry: base_delay * backoff_factor