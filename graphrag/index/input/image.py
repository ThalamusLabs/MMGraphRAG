# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Image input loader.

This mirrors the simple pattern used by ``text.py`` but loads images using Pillow.

Returned DataFrame columns:
  - id: stable hash based on basic image attributes and path
  - title: filename
  - creation_date: from storage backend
  - width, height, mode, format: image metadata
  - text: synthetic textual description so downstream text-based steps (chunking, etc.) don't break
  - image: the in‑memory ``PIL.Image.Image`` object (kept for any later custom processing)
  - any group metadata columns propagated by ``load_files``

Keeping it intentionally simple; callers can override ``file_pattern`` to match the set of
image extensions they want (e.g. ``.*\\.(png|jpg|jpeg)$``) until broader multi-extension
support is formalized.
"""

from __future__ import annotations

import logging
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

from graphrag.config.models.input_config import InputConfig
from graphrag.index.input.util import load_files
from graphrag.index.utils.hashing import gen_sha512_hash
from graphrag.storage.pipeline_storage import PipelineStorage

logger = logging.getLogger(__name__)


async def load_image(
    config: InputConfig,
    storage: PipelineStorage,
) -> pd.DataFrame:
    """Load image inputs from a directory.

    The caller is responsible for setting an appropriate ``file_pattern`` in ``config``
    (e.g. ``.*\\.(png|jpg|jpeg|gif)$``). We purposely avoid adding complex multi-extension
    logic here to keep the loader minimal.
    """
    logger.info("Loading image files from %s", config.storage.base_dir)

    async def load_file(path: str, group: dict | None = None) -> pd.DataFrame:  # type: ignore[override]
        if group is None:
            group = {}


        logger.debug("Loading image %s", path)

        # Get raw bytes then open via Pillow
        raw_bytes: bytes = await storage.get(path, as_bytes=True)  # type: ignore[assignment]
        try:
            with Image.open(BytesIO(raw_bytes)) as img:  # ensure resources closed
                img.load()  # force load so we can safely keep a copy
                image: Image.Image = img.copy()
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to open image %s: %s", path, e)
            raise

        width, height = image.size
        fmt = image.format or ""  # format may be None
        mode = image.mode

        # Synthetic text description so downstream text-based stages have something.
        synthetic_text = f"Image {Path(path).name} {width}x{height} mode={mode} format={fmt}"

        # Convert PIL Image to bytes for PyArrow compatibility
        # Store as PNG to preserve quality and support all modes
        image_buffer = BytesIO()
        image.save(image_buffer, format="PNG")
        image_bytes = image_buffer.getvalue()

        new_item: dict[str, Any] = {
            **group,
            "image": image_bytes,  # Store as bytes instead of PIL Image
            "width": width,
            "height": height,
            "mode": mode,
            "format": fmt,
            "text": synthetic_text,
        }

        # Stable id from basic attributes + path (exclude the PIL object itself)
        hash_fields = {
            "path": path,
            "width": width,
            "height": height,
            "mode": mode,
            "format": fmt,
        }
        new_item["id"] = gen_sha512_hash(hash_fields, hash_fields.keys())
        new_item["title"] = str(Path(path).name)
        new_item["creation_date"] = await storage.get_creation_date(path)

        return pd.DataFrame([new_item])

    return await load_files(load_file, config, storage)


