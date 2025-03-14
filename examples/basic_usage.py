#!/usr/bin/env python
"""
Basic usage example for the AI service library.

This script demonstrates how to use the AI service library to interact with
various AI providers.
"""

import os
import sys
import logging
from pathlib import Path

# Add the parent directory to the Python path to import the library
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_service.client import AIServiceClient
from ai_service.core.models import ChatCompletionRequest, CompletionRequest, Message, Role, EmbeddingRequest
from ai_service.utils.logging import setup_logging


# Set up logging
logger = setup_logging(level="INFO")


def completion_example(client: AIServiceClient) -> None:
    """Demonstrate using the completion API."""
    print("\n--- Completion Example ---")
    
    # Simple string prompt
    prompt = "Explain quantum computing in simple terms."
    print(f"Prompt: {prompt}")
    
    response = client.create_completion(prompt)
    
    print(f"Model: {response.model}")
    print(f"Response: {response.completion}")
    print(f"Token Usage: {response.usage.total_tokens} (Prompt: {response.usage.prompt_tokens}, Completion: {response.usage.completion_tokens})")


def chat_completion_example(client: AIServiceClient) -> None:
    """Demonstrate using the chat completion API."""
    print("\n--- Chat Completion Example ---")
    
    # Create a chat request with multiple messages
    request = ChatCompletionRequest(
        messages=[
            Message(role=Role.SYSTEM, content="You are a helpful assistant that speaks like a pirate."),
            Message(role=Role.USER, content="Tell me about the weather today.")
        ],
        temperature=0.7,
        max_tokens=150
    )
    
    response = client.create_chat_completion(request)
    
    print(f"Model: {response.model}")
    print(f"Response: {response.message.content}")
    print(f"Token Usage: {response.usage.total_tokens} (Prompt: {response.usage.prompt_tokens}, Completion: {response.usage.completion_tokens})")


def embedding_example(client: AIServiceClient) -> None:
    """Demonstrate using the embedding API."""
    print("\n--- Embedding Example ---")
    
    # Create an embedding request with multiple texts
    texts = [
        "This is an example sentence for embedding.",
        "Another sentence to compare similarity."
    ]
    
    request = EmbeddingRequest(
        input=texts,
        model="text-embedding-ada-002"
    )
    
    response = client.create_embedding(request)
    
    print(f"Model: {response.model}")
    print(f"Generated {len(response.embeddings)} embeddings")
    
    # Print the first few dimensions of each embedding
    for i, embedding in enumerate(response.embeddings):
        print(f"Embedding {i} (first 5 dimensions): {embedding.embedding[:5]}...")
    
    print(f"Token Usage: {response.usage.total_tokens}")


def token_counting_example(client: AIServiceClient) -> None:
    """Demonstrate the token counting functionality."""
    print("\n--- Token Counting Example ---")
    
    texts = [
        "This is a short text.",
        "This is a longer text that should contain more tokens than the previous one.",
        "This is an even longer text that discusses a complex topic like artificial intelligence and its implications for society. It should have significantly more tokens than the previous examples."
    ]
    
    for i, text in enumerate(texts):
        token_count = client.get_token_count(text)
        print(f"Text {i+1} ({len(text)} chars): {token_count} tokens")


def available_models_example(client: AIServiceClient) -> None:
    """Demonstrate listing available models."""
    print("\n--- Available Models Example ---")
    
    models = client.get_model_list()
    
    print(f"Found {len(models)} available models")
    
    # Group models by type
    gpt_models = [m for m in models if "gpt" in m.lower()]
    embedding_models = [m for m in models if "embed" in m.lower()]
    other_models = [m for m in models if "gpt" not in m.lower() and "embed" not in m.lower()]
    
    print(f"\nGPT Models ({len(gpt_models)}):")
    for model in sorted(gpt_models):
        print(f"  - {model}")
    
    print(f"\nEmbedding Models ({len(embedding_models)}):")
    for model in sorted(embedding_models):
        print(f"  - {model}")
    
    if other_models:
        print(f"\nOther Models ({len(other_models)}):")
        for model in sorted(other_models):
            print(f"  - {model}")


def mock_provider_example() -> None:
    """Demonstrate using the mock provider for testing."""
    print("\n--- Mock Provider Example ---")
    
    # Create a client with the mock provider
    client = AIServiceClient(
        provider="mock",
        response_delay=0.5,  # Half-second delay to simulate API calls
        fail_rate=0.0,  # 0% chance of random failure
        mock_responses={
            "Tell me a joke": "Why did the chicken cross the road? To get to the other side!",
            "Hello": "Hello there! How can I help you today?"
        }
    )
    
    # Use the mock client
    response = client.create_completion("Tell me a joke")
    print(f"Response to 'Tell me a joke': {response.completion}")
    
    response = client.create_completion("Hello")
    print(f"Response to 'Hello': {response.completion}")
    
    response = client.create_completion("Unknown prompt")
    print(f"Response to 'Unknown prompt': {response.completion}")


def main():
    """Run the examples."""
    # Check if OpenAI API key is available
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if api_key:
        # Create client with OpenAI provider
        client = AIServiceClient(
            provider="openai",
            api_key=api_key,
            default_model="gpt-3.5-turbo"
        )
        
        # Run examples that require a real API key
        completion_example(client)
        chat_completion_example(client)
        embedding_example(client)
        token_counting_example(client)
        available_models_example(client)
    else:
        print("OpenAI API key not found. Skipping real API examples.")
        print("Set the OPENAI_API_KEY environment variable to run all examples.")
    
    # Run the mock provider example (doesn't need a real API key)
    mock_provider_example()


if __name__ == "__main__":
    main()