"""
Configuration utilities for the AI service.

This module provides configuration management for the AI service,
including loading configuration from environment variables and files.
"""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast

from ai_service.core.errors import ConfigurationError
from ai_service.utils.logging import setup_logging


@dataclass
class AIServiceConfig:
    """Configuration for the AI service."""
    
    # Provider configuration
    provider: str = "openai"
    api_key: Optional[str] = None
    default_model: Optional[str] = None
    organization: Optional[str] = None
    
    # API request configuration
    timeout: float = 60.0
    max_retries: int = 3
    backoff_factor: float = 2.0
    
    # Rate limiting configuration
    rate_limit_capacity: float = 60.0
    rate_limit_refill_rate: float = 1.0
    
    # Logging configuration
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    # Additional provider-specific configuration
    provider_config: Dict[str, Any] = None
    
    def __post_init__(self) -> None:
        """Initialize with defaults for None values."""
        if self.provider_config is None:
            self.provider_config = {}
    
    @classmethod
    def from_env(cls) -> "AIServiceConfig":
        """
        Create a configuration from environment variables.
        
        Environment variables:
        - AI_SERVICE_PROVIDER: Provider to use
        - AI_SERVICE_API_KEY: API key for the provider
        - AI_SERVICE_MODEL: Default model to use
        - AI_SERVICE_ORGANIZATION: Organization ID (for OpenAI)
        - AI_SERVICE_TIMEOUT: Timeout in seconds
        - AI_SERVICE_MAX_RETRIES: Maximum number of retries
        - AI_SERVICE_BACKOFF_FACTOR: Backoff factor for retries
        - AI_SERVICE_RATE_LIMIT_CAPACITY: Rate limit token bucket capacity
        - AI_SERVICE_RATE_LIMIT_REFILL_RATE: Rate limit token refill rate
        - AI_SERVICE_LOG_LEVEL: Logging level
        - AI_SERVICE_LOG_FILE: Log file path
        
        Returns:
            Configuration loaded from environment variables
        """
        # Provider configuration
        provider = os.environ.get("AI_SERVICE_PROVIDER", "openai")
        
        # Try to get API key for the specified provider
        api_key = os.environ.get(f"AI_SERVICE_API_KEY")
        if not api_key:
            # Try provider-specific API key
            api_key = os.environ.get(f"{provider.upper()}_API_KEY")
        
        # Create config with basic settings
        config = cls(
            provider=provider,
            api_key=api_key,
            default_model=os.environ.get("AI_SERVICE_MODEL"),
            organization=os.environ.get("AI_SERVICE_ORGANIZATION"),
            timeout=float(os.environ.get("AI_SERVICE_TIMEOUT", "60.0")),
            max_retries=int(os.environ.get("AI_SERVICE_MAX_RETRIES", "3")),
            backoff_factor=float(os.environ.get("AI_SERVICE_BACKOFF_FACTOR", "2.0")),
            rate_limit_capacity=float(os.environ.get("AI_SERVICE_RATE_LIMIT_CAPACITY", "60.0")),
            rate_limit_refill_rate=float(os.environ.get("AI_SERVICE_RATE_LIMIT_REFILL_RATE", "1.0")),
            log_level=os.environ.get("AI_SERVICE_LOG_LEVEL", "INFO"),
            log_file=os.environ.get("AI_SERVICE_LOG_FILE"),
        )
        
        # Add provider-specific configuration
        provider_prefix = f"{provider.upper()}_"
        for key, value in os.environ.items():
            if key.startswith(provider_prefix) and key != f"{provider.upper()}_API_KEY":
                # Convert the key to lowercase and remove the prefix
                config_key = key[len(provider_prefix):].lower()
                config.provider_config[config_key] = value
        
        return config
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "AIServiceConfig":
        """
        Create a configuration from a dictionary.
        
        Args:
            config_dict: Dictionary of configuration values
            
        Returns:
            Configuration loaded from the dictionary
        """
        # Extract provider-specific configuration
        provider = config_dict.get("provider", "openai")
        provider_config = {}
        
        for key, value in list(config_dict.items()):
            if key.startswith(f"{provider}_"):
                # Move to provider_config and remove from main dict
                provider_key = key[len(f"{provider}_"):]
                provider_config[provider_key] = value
                del config_dict[key]
        
        # Add provider_config to the dictionary
        config_dict["provider_config"] = provider_config
        
        # Create config object
        return cls(**config_dict)
    
    @classmethod
    def from_file(cls, filepath: Union[str, Path]) -> "AIServiceConfig":
        """
        Create a configuration from a file.
        
        Supports JSON, TOML, and YAML files based on the file extension.
        
        Args:
            filepath: Path to the configuration file
            
        Returns:
            Configuration loaded from the file
            
        Raises:
            ConfigurationError: If the file format is not supported or parsing fails
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise ConfigurationError(f"Configuration file not found: {filepath}")
        
        # Load based on file extension
        if filepath.suffix == ".json":
            try:
                with open(filepath, "r") as f:
                    config_dict = json.load(f)
                return cls.from_dict(config_dict)
            except json.JSONDecodeError as e:
                raise ConfigurationError(f"Error parsing JSON config: {e}")
        
        elif filepath.suffix in (".toml", ".yaml", ".yml"):
            try:
                # Import the appropriate library
                if filepath.suffix == ".toml":
                    import toml
                    with open(filepath, "r") as f:
                        config_dict = toml.load(f)
                else:  # YAML
                    import yaml
                    with open(filepath, "r") as f:
                        config_dict = yaml.safe_load(f)
                        
                return cls.from_dict(config_dict)
            except ImportError:
                raise ConfigurationError(
                    f"Required library for {filepath.suffix} not installed. "
                    f"Please install toml or pyyaml as appropriate."
                )
            except Exception as e:
                raise ConfigurationError(f"Error parsing config file: {e}")
        
        else:
            raise ConfigurationError(
                f"Unsupported config file format: {filepath.suffix}. "
                f"Supported formats: .json, .toml, .yaml, .yml"
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to a dictionary.
        
        Returns:
            Dictionary representation of the configuration
        """
        result = {
            "provider": self.provider,
            "api_key": self.api_key,
            "default_model": self.default_model,
            "organization": self.organization,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "backoff_factor": self.backoff_factor,
            "rate_limit_capacity": self.rate_limit_capacity,
            "rate_limit_refill_rate": self.rate_limit_refill_rate,
            "log_level": self.log_level,
            "log_file": self.log_file,
        }
        
        # Add provider-specific configuration with prefixed keys
        for key, value in self.provider_config.items():
            result[f"{self.provider}_{key}"] = value
        
        return result
    
    def setup_logging(self) -> logging.Logger:
        """
        Set up logging based on the configuration.
        
        Returns:
            Configured logger
        """
        return setup_logging(
            level=self.log_level,
            log_file=self.log_file,
            module_name="ai_service"
        )


def load_config(config_path: Optional[Union[str, Path]] = None) -> AIServiceConfig:
    """
    Load configuration from file and/or environment variables.
    
    Configuration is loaded in this order (later sources override earlier ones):
    1. Default values
    2. Configuration file (if provided)
    3. Environment variables
    
    Args:
        config_path: Optional path to a configuration file
        
    Returns:
        Loaded configuration
    """
    # Start with configuration from environment
    config = AIServiceConfig.from_env()
    
    # If a config file is provided, load and merge with environment config
    if config_path:
        file_config = AIServiceConfig.from_file(config_path)
        
        # Only override values that are set in the file
        for key, value in file_config.to_dict().items():
            if value is not None and key != "provider_config":
                setattr(config, key, value)
        
        # Merge provider-specific configuration
        for key, value in file_config.provider_config.items():
            config.provider_config[key] = value
    
    return config