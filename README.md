# AI Service

A flexible, provider-agnostic service for interacting with AI APIs.

## Features

- Common interface for multiple AI providers (OpenAI, etc.)
- Built-in retry and rate limiting mechanisms
- Consistent error handling
- Standardized request/response models
- Comprehensive logging and metrics
- Mock provider for testing without API calls
- Configurable via environment variables or config files

## Installation

```bash
# Basic installation
pip install -e .

# Installation with development dependencies
pip install -e ".[dev]"
```

## Basic Usage

```python
from ai_service.client import AIServiceClient
from ai_service.core.models import CompletionRequest

# Initialize client
client = AIServiceClient(provider="openai", api_key="your-api-key")

# Create a completion request
request = CompletionRequest(
    prompt="Explain quantum computing in simple terms",
    max_tokens=150
)

# Get completion
response = client.create_completion(request)
print(response.completion)
```

## Architecture

The AI Service library is designed with a modular architecture that separates concerns:

- **Core**: Interfaces, models, and error types
- **Providers**: Implementations for different AI providers
- **Utils**: Shared utilities for retry, rate limiting, and logging
- **Client**: Main client class for end-user interaction

This design allows you to easily extend the library with new providers or functionality.

## Provider Support

Currently supported providers:

- **OpenAI**: For text completions, chat completions, and embeddings
- **Mock**: For testing without making real API calls

## Advanced Features

### Chat Completions

```python
from ai_service.client import AIServiceClient
from ai_service.core.models import ChatCompletionRequest, Message, Role

client = AIServiceClient(provider="openai", api_key="your-api-key")

request = ChatCompletionRequest(
    messages=[
        Message(role=Role.SYSTEM, content="You are a helpful assistant."),
        Message(role=Role.USER, content="How do I bake a cake?")
    ],
    temperature=0.7
)

response = client.create_chat_completion(request)
print(response.message.content)
```

### Embeddings

```python
from ai_service.client import AIServiceClient
from ai_service.core.models import EmbeddingRequest

client = AIServiceClient(provider="openai", api_key="your-api-key")

request = EmbeddingRequest(
    input=["Embeddings are useful for similarity search", "Vector databases store embeddings"],
    model="text-embedding-ada-002"
)

response = client.create_embedding(request)
print(f"Generated {len(response.embeddings)} embeddings")
```

### Retry Mechanism

The library includes a configurable retry mechanism that handles transient errors:

```python
from ai_service.utils.retry import RetryStrategy
from ai_service.client import AIServiceClient

# Create a custom retry strategy
retry_strategy = RetryStrategy(
    max_attempts=5,
    base_delay=1.0,
    backoff_factor=2.0,
    jitter=0.1
)

# Initialize client with retry strategy
client = AIServiceClient(
    provider="openai",
    api_key="your-api-key",
    retry_strategy=retry_strategy
)
```

### Rate Limiting

The library provides built-in rate limiting to ensure you don't exceed API rate limits:

```python
from ai_service.utils.rate_limit import RateLimiter
from ai_service.client import AIServiceClient

# Create a rate limiter
rate_limiter = RateLimiter()

# Add model-specific limits
rate_limiter.add_model_limit("gpt-4", 10.0, 0.5)  # 10 tokens capacity, 0.5 tokens/second

# Initialize client with rate limiter
client = AIServiceClient(
    provider="openai",
    api_key="your-api-key", 
    rate_limiter=rate_limiter
)
```

### Custom Configuration

The library supports configuration via environment variables or files:

```python
from ai_service.config import load_config
from ai_service.client import AIServiceClient

# Load config from environment variables and/or config file
config = load_config(config_path="config.json")

# Initialize client with config
client = AIServiceClient(
    provider=config.provider,
    api_key=config.api_key,
    default_model=config.default_model
)
```

## Examples

The repository includes example scripts in the `examples` directory:

- `basic_usage.py`: Basic usage of different API endpoints
- `idea_categorization.py`: Example of using the AI service for categorization

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=ai_service

# Run unit tests only
pytest ai_service/tests/unit

# Run integration tests only
pytest ai_service/tests/integration
```

## Extending the Library

### Adding a New Provider

To add a new AI provider, create a new class that implements the `AIProvider` interface:

```python
from ai_service.core.interfaces import AIProvider
from ai_service.providers.base import BaseProvider

class MyNewProvider(BaseProvider):
    """Implementation of the AI provider interface for MyNewProvider."""
    
    PROVIDER_NAME = "mynewprovider"
    
    def __init__(self, api_key, **kwargs):
        super().__init__(api_key=api_key, **kwargs)
        # Initialize provider-specific client
    
    # Implement abstract methods from AIProvider
    def create_completion(self, request):
        # Implementation
        
    def create_chat_completion(self, request):
        # Implementation
        
    def create_embedding(self, request):
        # Implementation
        
    def get_model_list(self):
        # Implementation
        
    def get_token_count(self, text, model=None):
        # Implementation
```

Then register your provider with the client:

```python
from ai_service.client import AIServiceClient
from mynewprovider import MyNewProvider

# Register the provider
AIServiceClient.register_provider("mynewprovider", MyNewProvider)

# Use the provider
client = AIServiceClient(provider="mynewprovider", api_key="your-api-key")
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.