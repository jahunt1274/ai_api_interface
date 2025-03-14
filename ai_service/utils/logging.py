"""
Logging utilities for the AI service.

This module provides logging setup and utility functions for consistent
logging across the AI service.
"""

import json
import logging
import os
import sys
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict, Optional, TypeVar, Union, cast

# Type variable for logging decorators
F = TypeVar('F', bound=Callable[..., Any])


class AIServiceFormatter(logging.Formatter):
    """Custom formatter for AI service logs."""
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record with additional AI service specific fields.
        
        Args:
            record: The log record to format
            
        Returns:
            Formatted log message
        """
        # Get the standard formatted message
        formatted_msg = super().format(record)
        
        # Add extra context from record.__dict__ if it exists
        extra_context = {}
        for key, value in record.__dict__.items():
            # Skip standard LogRecord attributes and private attributes
            if key.startswith('_') or key in {
                'name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                'filename', 'module', 'exc_info', 'exc_text', 'lineno',
                'funcName', 'created', 'asctime', 'msecs', 'relativeCreated',
                'thread', 'threadName', 'processName', 'process', 'message'
            }:
                continue
                
            # Add to extra context
            extra_context[key] = value
        
        # If we have extra context, append it to the message
        if extra_context:
            # Try to convert to JSON, falling back to str representation
            try:
                extra_json = json.dumps(extra_context)
                return f"{formatted_msg} | {extra_json}"
            except (TypeError, ValueError):
                extra_str = " ".join(f"{k}={v}" for k, v in extra_context.items())
                return f"{formatted_msg} | {extra_str}"
        
        return formatted_msg


def setup_logging(
    level: Union[int, str] = logging.INFO,
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
    module_name: str = "ai_service"
) -> logging.Logger:
    """
    Set up logging for the AI service.
    
    Args:
        level: Logging level
        log_file: Optional file to log to
        log_format: Optional format string for log messages
        module_name: Name of the module/logger to set up
        
    Returns:
        Configured logger
    """
    # Convert string level to int if needed
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    
    # Create logger
    logger = logging.getLogger(module_name)
    logger.setLevel(level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Default format if not specified
    if log_format is None:
        log_format = (
            "%(asctime)s [%(levelname)s] %(name)s:%(funcName)s:%(lineno)d - %(message)s"
        )
    
    # Create formatter
    formatter = AIServiceFormatter(log_format)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def log_call(logger: Optional[logging.Logger] = None, level: int = logging.DEBUG) -> Callable[[F], F]:
    """
    Decorator to log function calls and their results.
    
    Args:
        logger: Logger to use, or None to use a logger named after the module
        level: Logging level for the message
        
    Returns:
        Decorated function with call logging
    """
    def decorator(func: F) -> F:
        # Get logger if not provided
        nonlocal logger
        if logger is None:
            logger = logging.getLogger(func.__module__)
        
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Convert args to strings, being careful with sensitive data
            arg_str = ["<self>" if i == 0 and args and hasattr(args[0], '__class__') else _sanitize_arg(arg) 
                      for i, arg in enumerate(args)]
            kwarg_str = {k: _sanitize_arg(v) for k, v in kwargs.items()}
            
            # Log function call
            logger.log(level, f"CALL {func.__name__}({', '.join(arg_str)}, {kwarg_str})")
            
            # Call function and log result or exception
            try:
                result = func(*args, **kwargs)
                logger.log(level, f"RETURN {func.__name__}: {_sanitize_arg(result)}")
                return result
            except Exception as e:
                logger.log(level, f"EXCEPTION {func.__name__}: {type(e).__name__}: {str(e)}")
                raise
            
        return cast(F, wrapper)
    
    return decorator


def _sanitize_arg(arg: Any) -> str:
    """
    Sanitize an argument for logging, handling sensitive data.
    
    Args:
        arg: The argument to sanitize
        
    Returns:
        Safe string representation of the argument
    """
    # Handle None
    if arg is None:
        return "None"
    
    # Handle sensitive parameter names
    if isinstance(arg, str) and any(keyword in arg.lower() for keyword in ["password", "secret", "key", "token"]):
        return f"<{type(arg).__name__} of length {len(arg)}>"
    
    # Handle lists, tuples, and sets
    if isinstance(arg, (list, tuple, set)):
        return f"<{type(arg).__name__} of length {len(arg)}>"
    
    # Handle dictionaries
    if isinstance(arg, dict):
        sanitized = {}
        for k, v in arg.items():
            # Sanitize keys and values with sensitive names
            if any(keyword in str(k).lower() for keyword in ["password", "secret", "key", "token"]):
                sanitized[k] = "<redacted>"
            else:
                sanitized[k] = _sanitize_arg(v)
        return str(sanitized)
    
    # Handle objects with custom string representations
    try:
        return str(arg)
    except Exception:
        return f"<{type(arg).__name__}>"


def with_context(logger: logging.Logger, **context: Any) -> Callable[[F], F]:
    """
    Decorator to add context to all log messages within a function.
    
    Args:
        logger: Logger to add context to
        **context: Key-value pairs to add to the log context
        
    Returns:
        Decorated function with context logging
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Create a custom log adapter to add context
            adapter = logging.LoggerAdapter(logger, context)
            
            # Replace the logger temporarily
            original_logger = logger
            func.__globals__['logger'] = adapter
            
            try:
                return func(*args, **kwargs)
            finally:
                # Restore original logger
                func.__globals__['logger'] = original_logger
            
        return cast(F, wrapper)
    
    return decorator