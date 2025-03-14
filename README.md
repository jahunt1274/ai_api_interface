# AI Service

A flexible, provider-agnostic service for interacting with AI APIs.

## Features

- Common interface for multiple AI providers (OpenAI, etc.)
- Built-in retry and rate limiting
- Consistent error handling
- Standardized request/response models
- Comprehensive logging and metrics

## Installation

```bash
# Basic installation
pip install -e .

# Installation with development dependencies
pip install -e ".[dev]"
```

## Usage

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