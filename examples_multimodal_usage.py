"""
Example usage of the MultiModalChatLLM provider.

This example demonstrates how to use the multimodal provider for both
text-only and image-based chat completions with Azure OpenAI.
"""

import base64
import os
from pathlib import Path

from graphrag.config.models.language_model_config import LanguageModelConfig
from graphrag.config.enums import ModelType
from graphrag.language_model.providers.custom.multimodal_provider import MultiModalChatLLM


def load_image_as_base64(image_path: str) -> str:
    """Load an image file and convert it to base64 string.
    
    Args:
        image_path: Path to the image file.
        
    Returns:
        Base64 encoded string of the image.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def example_text_chat():
    """Example: Text-only chat completion."""
    print("=== Text-Only Chat Example ===\n")
    
    # Create configuration
    config = LanguageModelConfig(
        type=ModelType.AzureOpenAIChat,
        api_base="https://mustang-llm-base.openai.azure.com/",
        api_version="2024-12-01-preview",
        deployment_name="gpt-4.1-mini",
        model="gpt-4.1-mini",
        max_tokens=4000,
        temperature=0.7,
    )
    
    # Initialize provider
    llm = MultiModalChatLLM(name="multimodal_chat", config=config)
    
    # Simple text chat
    response = llm.chat(
        prompt="What are the main benefits of using GraphRAG?",
        is_image=False
    )
    
    print(f"Response: {response.output.content}\n")
    print(f"Full response keys: {response.output.full_response.keys()}\n")


def example_image_chat():
    """Example: Image-based chat completion."""
    print("=== Image-Based Chat Example ===\n")
    
    # Create configuration
    config = LanguageModelConfig(
        type=ModelType.AzureOpenAIChat,
        api_base="https://mustang-llm-base.openai.azure.com/",
        api_version="2024-12-01-preview",
        deployment_name="gpt-4.1-mini",
        model="gpt-4.1-mini",
        max_tokens=4000,
        temperature=0.7,
    )
    
    # Initialize provider
    llm = MultiModalChatLLM(name="multimodal_chat", config=config)
    
    # Load an image (example path - replace with actual image)
    image_path = "path/to/your/image.jpg"
    
    if not Path(image_path).exists():
        print(f"Image not found at {image_path}")
        print("Skipping image chat example.\n")
        return
    
    # Convert image to base64
    image_base64 = load_image_as_base64(image_path)
    
    # Chat with image
    response = llm.chat(
        prompt=image_base64,
        is_image=True
    )
    
    print(f"Response: {response.output.content}\n")


def example_multimodal_conversation():
    """Example: Multi-turn conversation with images."""
    print("=== Multi-Turn Conversation Example ===\n")
    
    # Create configuration
    config = LanguageModelConfig(
        type=ModelType.AzureOpenAIChat,
        api_base="https://mustang-llm-base.openai.azure.com/",
        api_version="2024-12-01-preview",
        deployment_name="gpt-4.1-mini",
        model="gpt-4.1-mini",
        max_tokens=4000,
        temperature=0.7,
    )
    
    # Initialize provider
    llm = MultiModalChatLLM(name="multimodal_chat", config=config)
    
    # First turn: text only
    response1 = llm.chat(
        prompt="Hello! I'm going to show you an image in a moment.",
        is_image=False
    )
    
    print(f"Turn 1 - User: Hello! I'm going to show you an image in a moment.")
    print(f"Turn 1 - Assistant: {response1.output.content}\n")
    
    # Get history from first response
    history = response1.history
    
    # Second turn: with image (if available)
    image_path = "path/to/your/image.jpg"
    
    if Path(image_path).exists():
        image_base64 = load_image_as_base64(image_path)
        
        response2 = llm.chat(
            prompt=image_base64,
            history=history,
            is_image=True
        )
        
        print(f"Turn 2 - User: [Image provided]")
        print(f"Turn 2 - Assistant: {response2.output.content}\n")
        
        # Third turn: follow-up question
        history = response2.history
        
        response3 = llm.chat(
            prompt="Can you describe what you see in more detail?",
            history=history,
            is_image=False
        )
        
        print(f"Turn 3 - User: Can you describe what you see in more detail?")
        print(f"Turn 3 - Assistant: {response3.output.content}\n")
    else:
        print(f"Image not found at {image_path}")
        print("Skipping image turns.\n")


async def example_async_chat():
    """Example: Async text chat completion."""
    print("=== Async Chat Example ===\n")
    
    # Create configuration
    config = LanguageModelConfig(
        type=ModelType.AzureOpenAIChat,
        api_base="https://mustang-llm-base.openai.azure.com/",
        api_version="2024-12-01-preview",
        deployment_name="gpt-4.1-mini",
        model="gpt-4.1-mini",
        max_tokens=4000,
        temperature=0.7,
    )
    
    # Initialize provider
    llm = MultiModalChatLLM(name="multimodal_chat", config=config)
    
    # Async chat
    response = await llm.achat(
        prompt="Explain what makes a good knowledge graph in 2 sentences.",
        is_image=False
    )
    
    print(f"Response: {response.output.content}\n")


if __name__ == "__main__":
    # Make sure GRAPHRAG_API_KEY is set
    if not os.environ.get("GRAPHRAG_API_KEY"):
        print("Error: GRAPHRAG_API_KEY environment variable is not set!")
        print("Please set it before running this example.")
        exit(1)
    
    # Run examples
    try:
        example_text_chat()
        example_image_chat()
        example_multimodal_conversation()
        
        # Run async example
        import asyncio
        asyncio.run(example_async_chat())
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()
