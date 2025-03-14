#!/usr/bin/env python
"""
Example script for categorizing ideas using the AI service.

This script demonstrates how to use the AI service library to categorize
business ideas by industry or topic.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add the parent directory to the Python path to import the library
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_service.client import AIServiceClient
from ai_service.core.models import ChatCompletionRequest, Message, Role
from ai_service.utils.logging import setup_logging


# Set up logging
logger = setup_logging(level="INFO")


# Define available categories
IDEA_CATEGORIES = [
    "Administrative Services",
    "Agriculture and Farming",
    "Angel Investing",
    "Apps",
    "Artificial Intelligence",
    "Arts",
    "Biotechnology",
    "Climate Tech",
    "Clothing and Apparel",
    "Commerce and Shopping",
    "Community and Lifestyle",
    "Construction",
    "Consumer Electronics",
    "Consumer Goods",
    "Content and Publishing",
    "Corporate Services",
    "Data Analytics",
    "Design",
    "Education",
    "Energy",
    "Entertainment",
    "Events",
    "Financial Services",
    "Food and Beverage",
    "Gaming",
    "Government and Military",
    "Hardware",
    "Health Care",
    "Information Technology",
    "Internet Services",
    "Lending and Investments",
    "Manufacturing",
    "Media and Entertainment",
    "Mobile",
    "Music and Audio",
    "Natural Resources",
    "Navigation and Mapping",
    "Payments",
    "Platforms",
    "Privacy and Security",
    "Private Equity",
    "Professional Services",
    "Public Admin and Safety",
    "Real Estate",
    "Retail",
    "Sales and Marketing",
    "Science and Engineering",
    "Social and Non-Profit",
    "Software",
    "Sports",
    "Sustainability",
    "Transportation",
    "Travel and Tourism",
    "Venture Capital"
]


def categorize_ideas(
    client: AIServiceClient,
    ideas: List[Dict[str, Any]],
    batch_size: int = 10
) -> List[Dict[str, Any]]:
    """
    Categorize a list of ideas using the AI service.
    
    Args:
        client: AI service client
        ideas: List of idea dictionaries, each with at least "id" and "title" keys
        batch_size: Number of ideas to process in each batch
        
    Returns:
        List of ideas with added "category" field
    """
    logger.info(f"Categorizing {len(ideas)} ideas in batches of {batch_size}")
    
    categorized_ideas = []
    batches = [ideas[i:i + batch_size] for i in range(0, len(ideas), batch_size)]
    
    for batch_index, batch in enumerate(batches):
        logger.info(f"Processing batch {batch_index + 1}/{len(batches)} ({len(batch)} ideas)")
        
        # Prepare batch data for the prompt
        batch_json = json.dumps(batch, indent=2)
        
        # Create the prompt for categorization
        system_prompt = (
            "You are an expert at categorizing business ideas into industries or topics. "
            "Your task is to categorize each idea into exactly one of the given categories."
        )
        
        user_prompt = (
            f"Categorize each of the following ideas into one of these categories:\n\n"
            f"{IDEA_CATEGORIES}\n\n"
            f"For each idea, provide an object with the original '_id' and a new 'category' field "
            f"indicating the chosen category. Return your answer as a JSON array of objects.\n\n"
            f"Here are the ideas to categorize:\n{batch_json}"
        )
        
        # Create the chat request
        request = ChatCompletionRequest(
            messages=[
                Message(role=Role.SYSTEM, content=system_prompt),
                Message(role=Role.USER, content=user_prompt)
            ],
            temperature=0.0,  # Use deterministic output for categorization
            max_tokens=2000
        )
        
        # Get the categorization response
        response = client.create_chat_completion(request)
        
        # Parse the JSON response
        try:
            result_text = response.message.content
            # Clean up any markdown code block wrapping
            if result_text.startswith("```") and result_text.endswith("```"):
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]
            
            batch_results = json.loads(result_text)
            
            # Create a map of idea IDs to original ideas
            id_to_idea = {idea["_id"]: idea for idea in batch}
            
            # Update ideas with categories
            for result in batch_results:
                idea_id = result["_id"]
                category = result["category"]
                
                if idea_id in id_to_idea:
                    # Copy the original idea and add the category
                    categorized_idea = id_to_idea[idea_id].copy()
                    categorized_idea["category"] = category
                    categorized_ideas.append(categorized_idea)
            
            logger.info(
                f"Categorized {len(batch_results)} ideas in batch {batch_index + 1}. "
                f"Token usage: {response.usage.total_tokens} "
                f"(Prompt: {response.usage.prompt_tokens}, "
                f"Completion: {response.usage.completion_tokens})"
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing response: {e}")
            logger.error(f"Response content: {response.message.content}")
        except Exception as e:
            logger.error(f"Error processing batch {batch_index + 1}: {e}")
    
    logger.info(f"Categorization complete: {len(categorized_ideas)} ideas categorized")
    return categorized_ideas


def load_ideas(filepath: str) -> List[Dict[str, Any]]:
    """
    Load ideas from a JSON file.
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        List of idea dictionaries
    """
    logger.info(f"Loading ideas from {filepath}")
    
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Ensure data is a list of dictionaries
        if not isinstance(data, list):
            raise ValueError("Expected a JSON array of idea objects")
        
        # Validate and transform ideas
        ideas = []
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                logger.warning(f"Skipping item {i}: not a dictionary")
                continue
            
            # Ensure each idea has an ID and title
            if "_id" not in item and "id" not in item:
                # Generate a simple ID if none exists
                item["_id"] = f"idea-{i+1}"
            elif "id" in item and "_id" not in item:
                # Copy id to _id for consistency
                item["_id"] = item["id"]
            
            if "title" not in item:
                logger.warning(f"Skipping item {i}: no title")
                continue
            
            ideas.append(item)
        
        logger.info(f"Loaded {len(ideas)} ideas")
        return ideas
        
    except Exception as e:
        logger.error(f"Error loading ideas: {e}")
        return []


def save_results(ideas: List[Dict[str, Any]], filepath: str) -> None:
    """
    Save categorized ideas to a JSON file.
    
    Args:
        ideas: List of categorized idea dictionaries
        filepath: Path to save the results
    """
    logger.info(f"Saving {len(ideas)} categorized ideas to {filepath}")
    
    try:
        # Ensure the directory exists
        output_dir = os.path.dirname(filepath)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(ideas, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")
        
    except Exception as e:
        logger.error(f"Error saving results: {e}")


def main():
    """Parse arguments and run the categorization."""
    parser = argparse.ArgumentParser(description="Categorize business ideas using AI")
    
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to input JSON file containing ideas"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Path to output JSON file (default: categorized_ideas.json)"
    )
    
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=10,
        help="Number of ideas to process in each batch (default: 10)"
    )
    
    parser.add_argument(
        "--model", "-m",
        default="gpt-3.5-turbo",
        help="Model to use for categorization (default: gpt-3.5-turbo)"
    )
    
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock provider instead of real API (for testing)"
    )
    
    args = parser.parse_args()
    
    # Set default output path if not provided
    if not args.output:
        output_dir = os.path.dirname(args.input) or "."
        output_filename = f"categorized_{os.path.basename(args.input)}"
        args.output = os.path.join(output_dir, output_filename)
    
    # Load ideas
    ideas = load_ideas(args.input)
    
    if not ideas:
        logger.error("No valid ideas found. Exiting.")
        return 1
    
    # Create AI client
    if args.mock:
        logger.info("Using mock provider")
        client = AIServiceClient(
            provider="mock",
            default_model=args.model
        )
    else:
        # Check if API key is available
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.error(
                "OpenAI API key not found. Set the OPENAI_API_KEY environment variable "
                "or use --mock for testing."
            )
            return 1
        
        logger.info(f"Using OpenAI provider with model {args.model}")
        client = AIServiceClient(
            provider="openai",
            api_key=api_key,
            default_model=args.model
        )
    
    # Categorize ideas
    categorized_ideas = categorize_ideas(client, ideas, args.batch_size)
    
    # Save results
    if categorized_ideas:
        save_results(categorized_ideas, args.output)
    else:
        logger.error("No ideas were successfully categorized.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())