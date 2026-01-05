# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing multimodal model provider definitions."""

from __future__ import annotations
import weave

import inspect
import json
import os
from typing import TYPE_CHECKING, Any, cast

from openai import AsyncAzureOpenAI, AzureOpenAI
from pydantic import BaseModel

from graphrag.language_model.response.base import (
    BaseModelOutput,
    BaseModelResponse,
    ModelResponse,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Generator

    from graphrag.cache.pipeline_cache import PipelineCache
    from graphrag.config.models.language_model_config import LanguageModelConfig


class MultiModalChatLLM:
    """A multi-modal chat LLM provider for Azure OpenAI with vision capabilities."""

    def __init__(
        self,
        name: str,
        config: LanguageModelConfig,
        cache: PipelineCache | None = None,
        **kwargs: Any,
    ):
        """Initialize the MultiModalChatLLM provider.
        
        Args:
            name: The name of the model instance.
            config: The language model configuration containing Azure OpenAI settings.
            **kwargs: Additional keyword arguments.
        """
        self.config = config
        self.name = name

        # Get API key from environment variable
        self.api_key = os.environ.get("GRAPHRAG_API_KEY")
        if not self.api_key:
            raise ValueError("GRAPHRAG_API_KEY environment variable is not set")

        # Validate required Azure OpenAI settings
        if not config.api_base:
            raise ValueError("api_base is required for Azure OpenAI")
        if not config.api_version:
            raise ValueError("api_version is required for Azure OpenAI")
        if not config.deployment_name:
            raise ValueError("deployment_name is required for Azure OpenAI")

        # Initialize Azure OpenAI clients
        self.async_client = AsyncAzureOpenAI(
            api_key=self.api_key,
            api_version=config.api_version,
            azure_endpoint=config.api_base,
        )

        self.sync_client = AzureOpenAI(
            api_key=self.api_key,
            api_version=config.api_version,
            azure_endpoint=config.api_base,
        )

    def _get_kwargs(self, **kwargs: Any) -> dict[str, Any]:
        """Get model arguments supported by Azure OpenAI.
        
        Args:
            **kwargs: Keyword arguments including json and json_model parameters.
            
        Returns:
            Dictionary of processed arguments for the API call.
        """
        new_args = {}
        
        # If using JSON, set up response_format
        if kwargs.get("json"):
            new_args["response_format"] = {"type": "json_object"}
            
            if (
                "json_model" in kwargs
                and inspect.isclass(kwargs["json_model"])
                and issubclass(kwargs["json_model"], BaseModel)
            ):
                # Use Pydantic model for structured output
                new_args["response_format"] = kwargs["json_model"]
        
        # Pass through other relevant parameters
        for key in ["max_tokens", "temperature", "top_p", "frequency_penalty", "presence_penalty"]:
            if key in kwargs:
                new_args[key] = kwargs[key]
                
        return new_args

    def _build_message_content(self, text: str, is_image: bool = False) -> list[dict[str, Any]] | str:
        """Build message content for text or image.
        
        Args:
            text: The text content or base64 encoded image string.
            is_image: Whether the text is a base64 encoded image.
            
        Returns:
            Message content in the appropriate format for Azure OpenAI.
        """
        if not is_image:
            # Simple text message
            return text

        # Multimodal message with image
        return [
            {
                "type": "text",
                "text": "Analyze the following image. Describe all the entities that are present in the image. For all entities, also describe the relationships that are present between entities. Do not make up entities or relationships. Make sure that the entities and relationships you find in the image are actually there. Finally, describe each entity and relationship in detail. Do not provide a summary, only a simple bullet pointed list of the entities and relationships, with their descriptions.",
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{text}"
                }
            }
        ]

    def _build_messages(
        self, 
        prompt: str | dict[str, Any], 
        history: list[dict[str, Any]] | None = None,
        is_image: bool = False
    ) -> list[dict[str, Any]]:
        """Build the messages array for the API request.
        
        Args:
            prompt: The current prompt/message (can be str or dict).
            history: Optional conversation history in OpenAI format.
            is_image: Whether the prompt contains a base64 encoded image.
            
        Returns:
            List of message dictionaries for the API.
        """
        messages = []

        # Add history if provided
        if history:
            messages.extend(history)

        # Add current message (convert to str if needed)
        prompt_str = prompt if isinstance(prompt, str) else str(prompt)
        content = self._build_message_content(prompt_str, is_image)

        print("Built message content:", content[0:100],"...")  # --- IGNORE ---
        messages.append({
            "role": "user",
            "content": content
        })

        return messages

    @weave.op
    async def achat(
        self,
        prompt: str | dict[str, Any],
        history: list[dict[str, Any]] | None = None,
        is_image: bool = False,
        **kwargs,
    ) -> ModelResponse:
        """Chat with the Azure OpenAI model (async).
        
        Args:
            prompt: The prompt text or base64 encoded image string.
            history: Optional conversation history.
            is_image: If True, prompt is treated as base64 encoded image.
            **kwargs: Additional parameters for the API call (supports json, json_model).
            
        Returns:
            ModelResponse with the chat completion.
        """
        messages = self._build_messages(prompt, history, is_image)
        processed_kwargs = self._get_kwargs(**kwargs)

        try:
            # Build API parameters
            api_params = {
                "model": self.config.deployment_name,
                "messages": messages,
                "max_tokens": processed_kwargs.get("max_tokens", self.config.max_tokens or 4000),
                "temperature": processed_kwargs.get("temperature", self.config.temperature or 0.7),
                "top_p": processed_kwargs.get("top_p", self.config.top_p),
                "frequency_penalty": processed_kwargs.get("frequency_penalty", self.config.frequency_penalty),
                "presence_penalty": processed_kwargs.get("presence_penalty", self.config.presence_penalty),
            }
            
            # Check if we're using structured outputs with a Pydantic model
            use_parse = False
            if "response_format" in processed_kwargs:
                if inspect.isclass(processed_kwargs["response_format"]) and issubclass(
                    processed_kwargs["response_format"], BaseModel
                ):
                    # Use .parse() for Pydantic models
                    use_parse = True
                else:
                    # Use .create() with JSON object format
                    api_params["response_format"] = processed_kwargs["response_format"]
            
            # Make API request using appropriate method
            if use_parse:
                response = await self.async_client.beta.chat.completions.parse(
                    **api_params,
                    response_format=processed_kwargs["response_format"]
                )
            else:
                response = await self.async_client.chat.completions.create(**api_params)

            # Get content and parsed response
            message = response.choices[0].message
            content = message.content or ""
            parsed_response: BaseModel | None = None
            
            # If using structured outputs with parse(), get the parsed object
            if use_parse and hasattr(message, "parsed") and message.parsed is not None:
                parsed_response = message.parsed
            elif "response_format" in processed_kwargs and not use_parse:
                # Manual JSON parsing for JSON object format
                try:
                    parsed_dict: dict[str, Any] = json.loads(content or "{}")
                    parsed_response = parsed_dict  # type: ignore
                except json.JSONDecodeError:
                    parsed_response = None

            print("----")  # --- IGNORE ---
            print("----")  # --- IGNORE ---
            print("----")  # --- IGNORE ---
            print(content)
            print("----")  # --- IGNORE ---
            print("----")  # --- IGNORE ---
            print("----")  # --- IGNORE ---

            # Build conversation history
            updated_history = messages.copy() if messages else []
            updated_history.append({
                "role": "assistant",
                "content": content
            })

            return BaseModelResponse(
                output=BaseModelOutput(
                    content=content,
                    full_response=response.model_dump() if hasattr(response, 'model_dump') else {},
                ),
                parsed_response=parsed_response,
                history=updated_history,
            )

        except Exception as e:
            error_msg = f"Error calling Azure OpenAI API: {str(e)}"
            raise RuntimeError(error_msg) from e

    def chat(
        self,
        prompt: str,
        history: list[dict[str, Any]] | None = None,
        is_image: bool = False,
        **kwargs,
    ) -> ModelResponse:
        """Chat with the Azure OpenAI model (sync).
        
        Args:
            prompt: The prompt text or base64 encoded image string.
            history: Optional conversation history.
            is_image: If True, prompt is treated as base64 encoded image.
            **kwargs: Additional parameters for the API call (supports json, json_model).
            
        Returns:
            ModelResponse with the chat completion.
        """
        messages = self._build_messages(prompt, history, is_image)
        processed_kwargs = self._get_kwargs(**kwargs)

        try:
            # Build API parameters
            api_params = {
                "model": self.config.deployment_name,
                "messages": messages,
                "max_tokens": processed_kwargs.get("max_tokens", self.config.max_tokens or 4000),
                "temperature": processed_kwargs.get("temperature", self.config.temperature or 0.7),
                "top_p": processed_kwargs.get("top_p", self.config.top_p),
                "frequency_penalty": processed_kwargs.get("frequency_penalty", self.config.frequency_penalty),
                "presence_penalty": processed_kwargs.get("presence_penalty", self.config.presence_penalty),
            }
            
            # Check if we're using structured outputs with a Pydantic model
            use_parse = False
            if "response_format" in processed_kwargs:
                if inspect.isclass(processed_kwargs["response_format"]) and issubclass(
                    processed_kwargs["response_format"], BaseModel
                ):
                    # Use .parse() for Pydantic models
                    use_parse = True
                else:
                    # Use .create() with JSON object format
                    api_params["response_format"] = processed_kwargs["response_format"]
            
            # Make API request using appropriate method
            if use_parse:
                response = self.sync_client.beta.chat.completions.parse(
                    **api_params,
                    response_format=processed_kwargs["response_format"]
                )
            else:
                response = self.sync_client.chat.completions.create(**api_params)

            # Get content and parsed response
            message = response.choices[0].message
            content = message.content or ""
            parsed_response: BaseModel | None = None
            
            # If using structured outputs with parse(), get the parsed object
            if use_parse and hasattr(message, "parsed") and message.parsed is not None:
                parsed_response = message.parsed
            elif "response_format" in processed_kwargs and not use_parse:
                # Manual JSON parsing for JSON object format
                try:
                    parsed_dict: dict[str, Any] = json.loads(content or "{}")
                    parsed_response = parsed_dict  # type: ignore
                except json.JSONDecodeError:
                    parsed_response = None

            # Build conversation history
            updated_history = messages.copy() if messages else []
            updated_history.append({
                "role": "assistant",
                "content": content
            })

            return BaseModelResponse(
                output=BaseModelOutput(
                    content=content,
                    full_response=response.model_dump() if hasattr(response, 'model_dump') else {},
                ),
                parsed_response=parsed_response,
                history=updated_history,
            )

        except Exception as e:
            error_msg = f"Error calling Azure OpenAI API: {str(e)}"
            raise RuntimeError(error_msg) from e

    async def achat_stream(
        self,
        prompt: str,
        history: list | None = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """Streaming is not implemented for this provider."""
        raise NotImplementedError("Streaming is not supported for MultiModalChatLLM")

    def chat_stream(
        self,
        prompt: str,
        history: list | None = None,
        **kwargs,
    ) -> Generator[str, None]:
        """Streaming is not implemented for this provider."""
        raise NotImplementedError("Streaming is not supported for MultiModalChatLLM")


class MockEmbeddingLLM:
    """A mock embedding LLM provider."""

    def __init__(self, **kwargs: Any):
        from graphrag.config.enums import ModelType
        
        self.config = LanguageModelConfig(
            type=ModelType.MockEmbedding, model="text-embedding-ada-002", api_key="mock"
        )

    def embed_batch(self, text_list: list[str], **kwargs: Any) -> list[list[float]]:
        """Generate an embedding for the input text."""
        if isinstance(text_list, str):
            return [[1.0, 1.0, 1.0]]
        return [[1.0, 1.0, 1.0] for _ in text_list]

    def embed(self, text: str, **kwargs: Any) -> list[float]:
        """Generate an embedding for the input text."""
        return [1.0, 1.0, 1.0]

    async def aembed(self, text: str, **kwargs: Any) -> list[float]:
        """Generate an embedding for the input text."""
        return [1.0, 1.0, 1.0]

    async def aembed_batch(
        self, text_list: list[str], **kwargs: Any
    ) -> list[list[float]]:
        """Generate an embedding for the input text."""
        if isinstance(text_list, str):
            return [[1.0, 1.0, 1.0]]
        return [[1.0, 1.0, 1.0] for _ in text_list]
