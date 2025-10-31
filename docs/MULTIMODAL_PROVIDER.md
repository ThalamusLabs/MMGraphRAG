# MultiModalChatLLM Provider

## Overview

The `MultiModalChatLLM` provider enables Azure OpenAI chat completions with vision capabilities, supporting both text-only and image-based interactions. It's designed for GraphRAG workflows that need to process visual content alongside text.

## Features

- ✅ **Text-only chat completions** - Standard conversational AI
- ✅ **Image-based completions** - Process base64-encoded JPEG images
- ✅ **Multimodal conversation history** - Maintain context across text and image turns
- ✅ **Async & Sync support** - Both `chat()` and `achat()` methods
- ✅ **Azure OpenAI integration** - Uses official OpenAI Python SDK
- ✅ **Environment variable support** - API key from `GRAPHRAG_API_KEY`

## Configuration

### Required Settings

The provider requires the following configuration parameters:

```python
from graphrag.config.models.language_model_config import LanguageModelConfig
from graphrag.config.enums import ModelType

config = LanguageModelConfig(
    type=ModelType.AzureOpenAIChat,
    api_base="https://your-instance.openai.azure.com/",
    api_version="2024-12-01-preview",
    deployment_name="gpt-4.1-mini",  # Must be a vision-capable model
    model="gpt-4.1-mini",
    max_tokens=4000,
    temperature=0.7,
)
```

### Environment Variables

Set the `GRAPHRAG_API_KEY` environment variable:

```bash
export GRAPHRAG_API_KEY="your-azure-openai-api-key"
```

Or in your `.env` file:

```
GRAPHRAG_API_KEY=your-azure-openai-api-key
```

### YAML Configuration (settings.yaml)

```yaml
models:
  multimodal_chat_model:
    type: azure_openai_chat
    api_base: https://your-instance.openai.azure.com/
    api_version: 2024-12-01-preview
    auth_type: api_key
    api_key: ${GRAPHRAG_API_KEY}
    model: gpt-4.1-mini
    deployment_name: gpt-4.1-mini
    model_supports_json: true
    concurrent_requests: 25
    max_tokens: 4000
    temperature: 0.7
```

## Usage

### Basic Text Chat

```python
from graphrag.language_model.providers.custom.multimodal_provider import MultiModalChatLLM

# Initialize
llm = MultiModalChatLLM(name="my_chat", config=config)

# Simple text completion
response = llm.chat(
    prompt="What is GraphRAG?",
    is_image=False
)

print(response.output.content)
```

### Image-Based Chat

```python
import base64

# Load and encode image
with open("image.jpg", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode("utf-8")

# Send image for analysis
response = llm.chat(
    prompt=image_base64,
    is_image=True  # Important: set this flag
)

print(response.output.content)
```

### Multi-Turn Conversation

```python
# First turn: text
response1 = llm.chat(
    prompt="I have a diagram to show you.",
    is_image=False
)

# Second turn: image with history
response2 = llm.chat(
    prompt=image_base64,
    history=response1.history,
    is_image=True
)

# Third turn: follow-up question
response3 = llm.chat(
    prompt="What are the main components?",
    history=response2.history,
    is_image=False
)
```

### Async Usage

```python
import asyncio

async def analyze_image():
    response = await llm.achat(
        prompt=image_base64,
        is_image=True
    )
    return response.output.content

# Run async
result = asyncio.run(analyze_image())
```

## API Reference

### Constructor

```python
MultiModalChatLLM(name: str, config: LanguageModelConfig, **kwargs)
```

**Parameters:**
- `name` (str): Identifier for this model instance
- `config` (LanguageModelConfig): Configuration object with Azure OpenAI settings
- `**kwargs`: Additional keyword arguments (currently unused)

**Raises:**
- `ValueError`: If `GRAPHRAG_API_KEY` is not set
- `ValueError`: If required config fields are missing

### Methods

#### `chat()`

Synchronous chat completion.

```python
def chat(
    prompt: str,
    history: list[dict[str, Any]] | None = None,
    is_image: bool = False,
    **kwargs
) -> ModelResponse
```

**Parameters:**
- `prompt` (str): Text content or base64-encoded image string
- `history` (list[dict], optional): Conversation history in OpenAI format
- `is_image` (bool): If `True`, treats prompt as base64 image
- `**kwargs`: Override config settings (e.g., `max_tokens`, `temperature`)

**Returns:**
- `ModelResponse`: Response object with `output.content` and `history`

#### `achat()`

Asynchronous chat completion (same parameters as `chat()`).

```python
async def achat(
    prompt: str,
    history: list[dict[str, Any]] | None = None,
    is_image: bool = False,
    **kwargs
) -> ModelResponse
```

### Response Structure

```python
response = llm.chat(...)

# Access response content
content = response.output.content  # str

# Access full API response
full_response = response.output.full_response  # dict

# Get updated conversation history
history = response.history  # list[dict]
```

## Image Format Requirements

- **Encoding**: Base64 string (without data URL prefix)
- **Format**: JPEG images (automatically prefixed with `data:image/jpeg;base64,`)
- **Size**: Respect Azure OpenAI token/size limits
- **Quality**: Higher quality images provide better analysis

### Image Encoding Example

```python
import base64
from PIL import Image
from io import BytesIO

def encode_image(image_path: str, max_size: tuple = (2048, 2048)) -> str:
    """Load, resize if needed, and encode image to base64."""
    with Image.open(image_path) as img:
        # Resize if too large
        if img.width > max_size[0] or img.height > max_size[1]:
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        # Convert to JPEG and encode
        buffer = BytesIO()
        img.convert("RGB").save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

# Use it
image_base64 = encode_image("my_image.png")
response = llm.chat(prompt=image_base64, is_image=True)
```

## Error Handling

```python
try:
    response = llm.chat(prompt=image_base64, is_image=True)
except RuntimeError as e:
    print(f"API Error: {e}")
except ValueError as e:
    print(f"Configuration Error: {e}")
```

## Best Practices

1. **Environment Variables**: Always use `GRAPHRAG_API_KEY` env var instead of hardcoding
2. **Image Size**: Resize large images before encoding to reduce token usage
3. **History Management**: Pass `history` for multi-turn conversations
4. **Error Handling**: Wrap API calls in try-except blocks
5. **Async for Scale**: Use `achat()` for concurrent requests
6. **Token Limits**: Monitor `max_tokens` for long responses
7. **Rate Limiting**: Configure `concurrent_requests` in config

## Integration with GraphRAG

### Using in Custom Workflows

```python
from graphrag.config.models.language_model_config import LanguageModelConfig
from graphrag.language_model.providers.custom.multimodal_provider import MultiModalChatLLM

# Load from settings.yaml
config = LanguageModelConfig(...)

# Create provider
llm = MultiModalChatLLM(name="image_analyzer", config=config)

# Use in workflow
def process_document_with_images(text: str, images: list[str]) -> dict:
    """Process document combining text and image analysis."""
    results = []
    
    # Analyze each image
    for img_base64 in images:
        response = llm.chat(
            prompt=img_base64,
            is_image=True
        )
        results.append(response.output.content)
    
    return {
        "text": text,
        "image_analyses": results
    }
```

## Troubleshooting

### Common Issues

1. **Import Error: "openai" not found**
   - Solution: Install OpenAI SDK: `pip install openai>=1.68.0`

2. **ValueError: GRAPHRAG_API_KEY not set**
   - Solution: Export or add to `.env` file

3. **API Error: 400 Bad Request**
   - Check deployment name matches your Azure resource
   - Verify API version is correct
   - Ensure model supports vision (e.g., gpt-4o, gpt-4-vision)

4. **API Error: 401 Unauthorized**
   - Verify API key is correct
   - Check key has access to the deployment

5. **Token Limit Exceeded**
   - Reduce image size before encoding
   - Lower `max_tokens` setting
   - Split large requests

## Limitations

- ❌ Streaming not supported (use `chat()` or `achat()` only)
- ❌ Only JPEG images supported (PNG/other formats need conversion)
- ❌ Single image per message (multiple images require separate turns)
- ⚠️ Token usage higher for images (consult Azure OpenAI pricing)

## Future Enhancements

Potential improvements for future versions:

- [ ] Support multiple images per message
- [ ] Auto-detect image format (PNG, WebP, etc.)
- [ ] Streaming support for vision models
- [ ] Built-in image compression/optimization
- [ ] Batch image processing
- [ ] Vision model fallback chain

## License

Copyright (c) 2025 Microsoft Corporation. Licensed under the MIT License.
