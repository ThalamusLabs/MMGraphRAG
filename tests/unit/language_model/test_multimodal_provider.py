"""
Tests for the MultiModalChatLLM provider.
"""

import base64
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphrag.config.enums import ModelType
from graphrag.config.models.language_model_config import LanguageModelConfig
from graphrag.language_model.providers.custom.multimodal_provider import (
    MultiModalChatLLM,
)


@pytest.fixture
def config():
    """Create a test configuration."""
    return LanguageModelConfig(
        type=ModelType.AzureOpenAIChat,
        api_base="https://test.openai.azure.com/",
        api_version="2024-12-01-preview",
        deployment_name="gpt-4.1-mini",
        model="gpt-4.1-mini",
        max_tokens=4000,
        temperature=0.7,
    )


@pytest.fixture
def mock_env():
    """Mock environment variable."""
    with patch.dict(os.environ, {"GRAPHRAG_API_KEY": "test-api-key"}):
        yield


def test_initialization_success(config, mock_env):
    """Test successful initialization."""
    llm = MultiModalChatLLM(name="test", config=config)
    assert llm.name == "test"
    assert llm.config == config
    assert llm.api_key == "test-api-key"


def test_initialization_missing_api_key(config):
    """Test initialization fails without API key."""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="GRAPHRAG_API_KEY"):
            MultiModalChatLLM(name="test", config=config)


def test_initialization_missing_api_base(mock_env):
    """Test initialization fails without api_base."""
    config = LanguageModelConfig(
        type=ModelType.AzureOpenAIChat,
        api_base=None,
        api_version="2024-12-01-preview",
        deployment_name="gpt-4.1-mini",
    )
    with pytest.raises(ValueError, match="api_base is required"):
        MultiModalChatLLM(name="test", config=config)


def test_build_message_content_text(config, mock_env):
    """Test building text message content."""
    llm = MultiModalChatLLM(name="test", config=config)
    content = llm._build_message_content("Hello world", is_image=False)
    assert content == "Hello world"


def test_build_message_content_image(config, mock_env):
    """Test building image message content."""
    llm = MultiModalChatLLM(name="test", config=config)
    test_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAACklEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg=="
    content = llm._build_message_content(test_base64, is_image=True)
    
    assert isinstance(content, list)
    assert len(content) == 1
    assert content[0]["type"] == "image_url"
    assert content[0]["image_url"]["url"] == f"data:image/jpeg;base64,{test_base64}"


def test_build_messages_no_history(config, mock_env):
    """Test building messages without history."""
    llm = MultiModalChatLLM(name="test", config=config)
    messages = llm._build_messages("Hello", history=None, is_image=False)
    
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Hello"


def test_build_messages_with_history(config, mock_env):
    """Test building messages with history."""
    llm = MultiModalChatLLM(name="test", config=config)
    history = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    messages = llm._build_messages("How are you?", history=history, is_image=False)
    
    assert len(messages) == 3
    assert messages[0] == history[0]
    assert messages[1] == history[1]
    assert messages[2]["role"] == "user"
    assert messages[2]["content"] == "How are you?"


@pytest.mark.asyncio
async def test_achat_text(config, mock_env):
    """Test async chat with text."""
    llm = MultiModalChatLLM(name="test", config=config)
    
    # Mock the async client
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Test response"
    mock_response.model_dump.return_value = {"id": "test-id"}
    
    with patch.object(
        llm.async_client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_response
        
        response = await llm.achat("Hello", is_image=False)
        
        assert response.output.content == "Test response"
        assert len(response.history) == 2
        assert response.history[0]["role"] == "user"
        assert response.history[1]["role"] == "assistant"
        
        # Verify API call
        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4.1-mini"
        assert len(call_kwargs["messages"]) == 1


@pytest.mark.asyncio
async def test_achat_image(config, mock_env):
    """Test async chat with image."""
    llm = MultiModalChatLLM(name="test", config=config)
    
    # Mock the async client
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "I see a test image"
    mock_response.model_dump.return_value = {"id": "test-id"}
    
    test_base64 = "test_image_base64_string"
    
    with patch.object(
        llm.async_client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_response
        
        response = await llm.achat(test_base64, is_image=True)
        
        assert response.output.content == "I see a test image"
        
        # Verify API call with image format
        call_kwargs = mock_create.call_args.kwargs
        messages = call_kwargs["messages"]
        assert len(messages) == 1
        assert isinstance(messages[0]["content"], list)
        assert messages[0]["content"][0]["type"] == "image_url"


def test_chat_sync(config, mock_env):
    """Test synchronous chat."""
    llm = MultiModalChatLLM(name="test", config=config)
    
    # Mock the sync client
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Sync response"
    mock_response.model_dump.return_value = {"id": "test-id"}
    
    with patch.object(llm.sync_client.chat.completions, "create") as mock_create:
        mock_create.return_value = mock_response
        
        response = llm.chat("Hello", is_image=False)
        
        assert response.output.content == "Sync response"
        mock_create.assert_called_once()


@pytest.mark.asyncio
async def test_achat_with_history(config, mock_env):
    """Test async chat with conversation history."""
    llm = MultiModalChatLLM(name="test", config=config)
    
    # Mock the async client
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Response with history"
    mock_response.model_dump.return_value = {"id": "test-id"}
    
    history = [
        {"role": "user", "content": "First message"},
        {"role": "assistant", "content": "First response"},
    ]
    
    with patch.object(
        llm.async_client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_response
        
        response = await llm.achat("Second message", history=history, is_image=False)
        
        # Verify history is included in API call
        call_kwargs = mock_create.call_args.kwargs
        messages = call_kwargs["messages"]
        assert len(messages) == 3
        assert messages[0] == history[0]
        assert messages[1] == history[1]
        assert messages[2]["content"] == "Second message"
        
        # Verify updated history in response
        assert len(response.history) == 4


@pytest.mark.asyncio
async def test_achat_error_handling(config, mock_env):
    """Test error handling in async chat."""
    llm = MultiModalChatLLM(name="test", config=config)
    
    with patch.object(
        llm.async_client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.side_effect = Exception("API Error")
        
        with pytest.raises(RuntimeError, match="Error calling Azure OpenAI API"):
            await llm.achat("Hello", is_image=False)


def test_chat_stream_not_implemented(config, mock_env):
    """Test that streaming is not implemented."""
    llm = MultiModalChatLLM(name="test", config=config)
    
    with pytest.raises(NotImplementedError, match="Streaming is not supported"):
        list(llm.chat_stream("Hello"))


@pytest.mark.asyncio
async def test_achat_stream_not_implemented(config, mock_env):
    """Test that async streaming is not implemented."""
    llm = MultiModalChatLLM(name="test", config=config)
    
    with pytest.raises(NotImplementedError, match="Streaming is not supported"):
        stream = await llm.achat_stream("Hello")
        async for _ in stream:
            pass


def test_custom_parameters(config, mock_env):
    """Test passing custom parameters to API."""
    llm = MultiModalChatLLM(name="test", config=config)
    
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Custom params response"
    mock_response.model_dump.return_value = {"id": "test-id"}
    
    with patch.object(llm.sync_client.chat.completions, "create") as mock_create:
        mock_create.return_value = mock_response
        
        response = llm.chat(
            "Hello", 
            is_image=False,
            max_tokens=1000,
            temperature=0.5
        )
        
        # Verify custom params were passed
        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["max_tokens"] == 1000
        assert call_kwargs["temperature"] == 0.5
