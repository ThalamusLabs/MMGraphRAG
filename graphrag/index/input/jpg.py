# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing load method definition."""

import base64
import logging
from io import BytesIO
from pathlib import Path

import pandas as pd
from PIL import Image

from graphrag.config.models.input_config import InputConfig
from graphrag.index.input.util import load_files
from graphrag.index.utils.hashing import gen_sha512_hash
from graphrag.storage.pipeline_storage import PipelineStorage

logger = logging.getLogger(__name__)


async def load_jpg(
    config: InputConfig,
    storage: PipelineStorage,
) -> pd.DataFrame:
    """Load JPG image inputs from a directory."""
    logger.info("Loading jpg files from %s", config.storage.base_dir)
    print(f"Loading jpg files from {config.storage.base_dir}")
    async def load_file(path: str, group: dict | None = None) -> pd.DataFrame:
        if group is None:
            group = {}
        
        # Load image as bytes
        image_bytes = await storage.get(path, as_bytes=True)
        
        # Extract image metadata using PIL
        try:
            image = Image.open(BytesIO(image_bytes))
            width, height = image.size
            format_type = image.format
            mode = image.mode
            
            # Get EXIF data if available
            exif_data = {}
            if hasattr(image, '_getexif') and image._getexif() is not None:
                exif_data = image._getexif()
        except Exception as e:
            logger.warning("Could not extract image metadata from %s: %s", path, e)
            width = height = format_type = mode = None
            exif_data = {}
        
        # Convert image to base64 for storage/processing
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # Create structured data similar to other loaders
        new_item = {
            **group,
            "text": image_base64,  # Store base64 encoded image data
            "image_width": width,
            "image_height": height,
            "image_format": format_type,
            "image_mode": mode,
            "image_size_bytes": len(image_bytes),
            "exif_data": str(exif_data) if exif_data else "",
        }
        
        new_item["id"] = gen_sha512_hash(new_item, ["text", "image_width", "image_height"])
        new_item["title"] = str(Path(path).name)
        new_item["creation_date"] = await storage.get_creation_date(path)
        
        return pd.DataFrame([new_item])

    return await load_files(load_file, config, storage)