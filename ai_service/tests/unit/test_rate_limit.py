"""
Unit tests for rate limiting utilities.

This module contains unit tests for the rate limiting mechanisms.
"""

import pytest
import threading
import time
from unittest.mock import MagicMock, patch

from ai_service.core.errors import RateLimitError
from ai_service.utils.rate_limit import TokenBucket, RateLimiter, with_rate_limit


class TestTokenBucket:
    """Tests for the TokenBucket class."""

    def test_initialization(self):
        """Test that TokenBucket initializes with the correct capacity."""
        bucket = TokenBucket(capacity=10, refill_rate=1)
        assert bucket.capacity == 10
        assert bucket.refill_rate == 1
        assert bucket.tokens == 10  # Should start full

    def test_try_consume_success(self):
        """Test that try_consume succeeds when enough tokens are available."""
        bucket = TokenBucket(capacity=10, refill_rate=1)
        assert bucket.try_consume(5) is True  # Should succeed
        assert bucket.tokens == 5  # Should have 5 tokens left

    def test_try_consume_failure(self):
        """Test that try_consume fails when not enough tokens are available."""
        bucket = TokenBucket(capacity=10, refill_rate=1)
        assert bucket.try_consume(15) is False  # Should fail
        assert bucket.tokens == 10  # Should still have all tokens

    def test_consume_with_wait(self):
        """Test that consume waits for tokens to become available."""
        bucket = TokenBucket(capacity=10, refill_rate=10)  # 10 tokens per second
        
        # Consume all tokens
        bucket.try_consume(10)
        assert bucket.tokens == 0
        
        # Try to consume 5 more tokens, which should wait ~0.5 seconds
        start_time = time.time()
        wait_time = bucket.consume(5, wait=True)
        elapsed = time.time() - start_time
        
        # Check that we waited and got tokens
        assert wait_time > 0
        assert 0.4 < elapsed < 1.0  # Allow some margin for test execution
        assert bucket.tokens > 0  # Should have some tokens left (from refill during wait)

    def test_consume_no_wait(self):
        """Test that consume raises an error when wait=False and not enough tokens."""
        bucket = TokenBucket(capacity=10, refill_rate=1)
        
        # Consume all tokens
        bucket.try_consume(10)
        assert bucket.tokens == 0
        
        # Try to consume more tokens without waiting
        with pytest.raises(RateLimitError):
            bucket.consume(5, wait=False)

    def test_consume_max_wait(self):
        """Test that consume respects max_wait parameter."""
        bucket = TokenBucket(capacity=10, refill_rate=1)  # 1 token per second
        
        # Consume all tokens
        bucket.try_consume(10)
        assert bucket.tokens == 0
        
        # Try to consume 5 more tokens with a max_wait of 1 second
        # This should fail because it would take 5 seconds to get 5 tokens
        with pytest.raises(RateLimitError) as exc_info:
            bucket.consume(5, wait=True, max_wait=1.0)
        
        # Check that we got a retry_after in the error
        assert exc_info.value.retry_after >= 4  # Should be roughly 5 seconds, but allow some margin

    def test_refill(self):
        """Test that tokens are refilled over time."""
        bucket = TokenBucket(capacity=10, refill_rate=10)  # 10 tokens per second
        
        # Consume some tokens
        bucket.try_consume(5)
        assert bucket.tokens == 5
        
        # Wait for a bit to let tokens refill
        time.sleep(0.2)  # Should add ~2 tokens
        
        # Force a refill by checking tokens
        bucket._refill()
        assert 6.9 < bucket.tokens < 7.1  # Allow small margin for timing variations

    def test_refill_up_to_capacity(self):
        """Test that tokens are refilled only up to capacity."""
        bucket = TokenBucket(capacity=10, refill_rate=10)  # 10 tokens per second
        
        # Consume some tokens
        bucket.try_consume(5)
        assert bucket.tokens == 5
        
        # Wait long enough that refill would exceed capacity
        time.sleep(1.5)  # Would add 15 tokens at 10 per second
        
        # Force a refill by checking tokens
        bucket._refill()
        assert bucket.tokens == 10  # Should cap at capacity


class TestRateLimiter:
    """Tests for the RateLimiter class."""

    def test_initialization(self):
        """Test that RateLimiter initializes with default values."""
        limiter = RateLimiter()
        assert limiter.default_capacity == 60.0
        assert limiter.default_refill_rate == 1.0
        assert limiter.global_bucket is not None
        assert limiter.model_buckets == {}
        assert limiter.logger is not None

    def test_add_model_limit(self):
        """Test adding a model-specific rate limit."""
        limiter = RateLimiter()
        limiter.add_model_limit("gpt-4", 10.0, 0.5)
        
        assert "gpt-4" in limiter.model_buckets
        assert limiter.model_buckets["gpt-4"].capacity == 10.0
        assert limiter.model_buckets["gpt-4"].refill_rate == 0.5

    def test_get_bucket_for_model(self):
        """Test getting the token bucket for a specific model."""
        limiter = RateLimiter()
        
        # Add a model-specific bucket
        limiter.add_model_limit("gpt-4", 10.0, 0.5)
        
        # Get the bucket for that model
        bucket = limiter.get_bucket_for_model("gpt-4")
        assert bucket is limiter.model_buckets["gpt-4"]
        
        # Get the bucket for a model without a specific limit
        bucket = limiter.get_bucket_for_model("gpt-3.5-turbo")
        assert bucket is limiter.global_bucket

    def test_try_acquire(self):
        """Test acquiring tokens for a request without waiting."""
        limiter = RateLimiter()
        
        # Add a model-specific bucket with low capacity
        limiter.add_model_limit("gpt-4", 5.0, 1.0)
        
        # Try to acquire tokens for a model with no specific limit
        assert limiter.try_acquire(model="gpt-3.5-turbo", tokens=1.0) is True
        
        # Try to acquire tokens for a model with a specific limit
        assert limiter.try_acquire(model="gpt-4", tokens=3.0) is True
        
        # Try to acquire more tokens than available for the model
        assert limiter.try_acquire(model="gpt-4", tokens=3.0) is False

    def test_acquire_with_wait(self):
        """Test acquiring tokens with waiting."""
        limiter = RateLimiter()
        
        # Add a model-specific bucket with low capacity and high refill rate
        limiter.add_model_limit("gpt-4", 5.0, 5.0)  # 5 tokens per second
        
        # Acquire all tokens for the model
        assert limiter.try_acquire(model="gpt-4", tokens=5.0) is True
        
        # Try to acquire more tokens with waiting
        start_time = time.time()
        wait_time = limiter.acquire(model="gpt-4", tokens=2.0, wait=True)
        elapsed = time.time() - start_time
        
        # Check that we waited and got tokens
        assert wait_time > 0
        assert 0.2 < elapsed < 0.6  # Should wait ~0.4 seconds for 2 tokens at 5 per second

    def test_acquire_no_wait(self):
        """Test that acquire raises an error when wait=False and not enough tokens."""
        limiter = RateLimiter()
        
        # Add a model-specific bucket with low capacity
        limiter.add_model_limit("gpt-4", 5.0, 1.0)
        
        # Acquire all tokens for the model
        assert limiter.try_acquire(model="gpt-4", tokens=5.0) is True
        
        # Try to acquire more tokens without waiting
        with pytest.raises(RateLimitError):
            limiter.acquire(model="gpt-4", tokens=1.0, wait=False)

    def test_acquire_max_wait(self):
        """Test that acquire respects max_wait parameter."""
        limiter = RateLimiter()
        
        # Add a model-specific bucket with low capacity and low refill rate
        limiter.add_model_limit("gpt-4", 5.0, 0.5)  # 0.5 tokens per second
        
        # Acquire all tokens for the model
        assert limiter.try_acquire(model="gpt-4", tokens=5.0) is True
        
        # Try to acquire more tokens with a max_wait of 1 second
        # This should fail because it would take 6 seconds to get 3 tokens
        with pytest.raises(RateLimitError) as exc_info:
            limiter.acquire(model="gpt-4", tokens=3.0, wait=True, max_wait=1.0)
        
        # Check that we got a retry_after in the error
        assert exc_info.value.retry_after >= 5  # Should be roughly 6 seconds, but allow some margin

    def test_global_and_model_buckets(self):
        """Test that both global and model-specific buckets are checked."""
        limiter = RateLimiter(default_capacity=10.0, default_refill_rate=1.0)
        
        # Add a model-specific bucket with higher capacity
        limiter.add_model_limit("gpt-4", 20.0, 1.0)
        
        # Acquire 8 tokens from global bucket
        assert limiter.try_acquire(tokens=8.0) is True
        
        # Try to acquire 5 tokens for gpt-4
        # This should fail because the global bucket only has 2 tokens left
        assert limiter.try_acquire(model="gpt-4", tokens=5.0) is False
        
        # Try to acquire 1 token for gpt-4, which should work
        assert limiter.try_acquire(model="gpt-4", tokens=1.0) is True
        
        # Try to acquire 1 more token, which should fail (global bucket is empty)
        assert limiter.try_acquire(model="gpt-4", tokens=1.0) is False


class TestWithRateLimit:
    """Tests for the with_rate_limit decorator."""

    def test_rate_limit_decorator(self):
        """Test that the decorator applies rate limiting to a function."""
        # Create a rate limiter with low capacity
        limiter = RateLimiter(default_capacity=3.0, default_refill_rate=3.0)
        
        # Create a mock function
        mock_func = MagicMock(return_value="success")
        
        # Apply the decorator
        decorated = with_rate_limit(limiter, tokens=1.0)(mock_func)
        
        # Call the function multiple times
        for i in range(3):
            result = decorated()
            assert result == "success"
        
        # The next call should wait
        start_time = time.time()
        result = decorated()
        elapsed = time.time() - start_time
        
        assert result == "success"
        assert elapsed >= 0.2  # Should wait at least ~0.33 seconds

    def test_rate_limit_with_model(self):
        """Test that the decorator uses the model from function arguments."""
        # Create a rate limiter with model-specific limits
        limiter = RateLimiter()
        limiter.add_model_limit("gpt-4", 1.0, 1.0)
        
        # Create a mock function that takes a model parameter
        mock_func = MagicMock(return_value="success")
        
        # Apply the decorator with model_arg_name
        decorated = with_rate_limit(limiter, tokens=1.0, model_arg_name="model")(mock_func)
        
        # Call with different models
        result1 = decorated(model="gpt-3.5-turbo")  # Uses global bucket
        assert result1 == "success"
        
        result2 = decorated(model="gpt-4")  # Uses gpt-4 bucket
        assert result2 == "success"
        
        # The next call with gpt-4 should wait
        start_time = time.time()
        result3 = decorated(model="gpt-4")
        elapsed = time.time() - start_time
        
        assert result3 == "success"
        assert elapsed >= 0.8  # Should wait at least ~1 second

    def test_rate_limit_with_request_object(self):
        """Test that the decorator can extract model from a request object."""
        # Create a rate limiter with model-specific limits
        limiter = RateLimiter()
        limiter.add_model_limit("gpt-4", 1.0, 1.0)
        
        # Create a mock function that takes a request parameter
        mock_func = MagicMock(return_value="success")
        
        # Apply the decorator
        decorated = with_rate_limit(limiter, tokens=1.0, model_arg_name="model")(mock_func)
        
        # Create a request-like object with a model attribute
        class Request:
            def __init__(self, model):
                self.model = model
        
        # Call with different request objects
        result1 = decorated(Request("gpt-3.5-turbo"))  # Uses global bucket
        assert result1 == "success"
        
        result2 = decorated(Request("gpt-4"))  # Uses gpt-4 bucket
        assert result2 == "success"
        
        # The next call with gpt-4 should wait
        start_time = time.time()
        result3 = decorated(Request("gpt-4"))
        elapsed = time.time() - start_time
        
        assert result3 == "success"
        assert elapsed >= 0.8  # Should wait at least ~1 second