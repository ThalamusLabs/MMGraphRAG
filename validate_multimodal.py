import os
import base64

from graphrag.config.models.language_model_config import LanguageModelConfig
from graphrag.config.enums import ModelType
from graphrag.language_model.providers.custom.multimodal_provider import (
    MultiModalChatLLM,
)

# Initialize - pass the API key from environment to the config
config = LanguageModelConfig(
    type=ModelType.MultiModal,
    api_key=os.environ.get("GRAPHRAG_API_KEY"),  # Pass API key to config
    api_base="https://mustang-llm-base.openai.azure.com/",
    api_version="2024-12-01-preview",
    deployment_name="gpt-4.1-mini",
    model="gpt-4.1-mini",
    max_tokens=4000,
)

llm = MultiModalChatLLM(name="multimodal", config=config)

# Text chat
response = llm.chat(prompt="Extract entities from text: Dickens went walking with his small dog named Alice who were lost in Wonderland. How could they come here they thought. Blah blah blah.", is_image=False)
print(response.output.content)

# Image chat
with open("./christmas/input/bus.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode("utf-8")
response = llm.chat(prompt=image_b64, is_image=True)
print("----")
print(response.output.content)
