# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing run_workflow method definition."""

import weave
import json
import logging
from typing import Any, cast

import pandas as pd

from graphrag.callbacks.workflow_callbacks import WorkflowCallbacks
from graphrag.config.models.chunking_config import ChunkStrategyType
from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.index.operations.chunk_text.chunk_text import chunk_text
from graphrag.index.operations.chunk_text.strategies import get_encoding_fn
from graphrag.index.typing.context import PipelineRunContext
from graphrag.index.typing.workflow import WorkflowFunctionOutput
from graphrag.index.utils.hashing import gen_sha512_hash
from graphrag.utils.storage import load_table_from_storage, write_table_to_storage
from PIL import Image
import io
import base64
logger = logging.getLogger(__name__)

async def run_workflow(
    config: GraphRagConfig,
    context: PipelineRunContext,
) -> WorkflowFunctionOutput:
    """All the steps to transform base text_units."""
    logger.info("Workflow started: create_base_text_units")
    documents = await load_table_from_storage("documents", context.output_storage)

    chunks = config.chunks

    # print(documents.columns)
    if  documents["doc_type"].iloc[0] == "jpeg":

        output = create_base_image_units(documents=documents)

        print("Created base image units:", len(output))
        for _, row in output.head(2).iterrows():
            print(f"ID: {row['id']}, Text length: {len(row['text'])}, n_tokens: {row['n_tokens']}, doc_type: {row['doc_type']}\nText extract:\n{row['text'][:50]}...\n")

    else:
        output = create_base_text_units(
            documents,
            context.callbacks,
            chunks.group_by_columns,
            chunks.size,
            chunks.overlap,
            chunks.encoding_model,
            strategy=chunks.strategy,
            prepend_metadata=chunks.prepend_metadata,
            chunk_size_includes_metadata=chunks.chunk_size_includes_metadata,
        )

    await write_table_to_storage(output, "text_units", context.output_storage)

    logger.info("Workflow completed: create_base_text_units")
    return WorkflowFunctionOutput(result=output)

import pandas as pd


@weave.op
def create_base_image_units(
        documents: pd.DataFrame,
) -> pd.DataFrame:
    
    def chunker(base64_image: str, n_chunks: int = 16) -> list[str]:
        image_bytes = base64.b64decode(base64_image)
        img = Image.open(io.BytesIO(image_bytes))

        width, height = img.size
        
        # Calculate grid size (e.g., 4 chunks = 2x2, 9 chunks = 3x3, 16 chunks = 4x4)
        grid_size = int(n_chunks ** 0.5)
        chunk_width = width // grid_size
        chunk_height = height // grid_size

        result = []
        for row in range(grid_size):
            for col in range(grid_size):
                left = col * chunk_width
                top = row * chunk_height
                right = left + chunk_width
                bottom = top + chunk_height
                
                chunk = img.crop((left, top, right, bottom))
                
                # Convert RGBA to RGB if needed
                if chunk.mode == 'RGBA':
                    chunk = chunk.convert('RGB')
                
                # Convert chunk to bytes
                chunk_bytes = io.BytesIO()
                chunk.save(chunk_bytes, format='JPEG')
                chunk_bytes = chunk_bytes.getvalue()
                
                result.append(base64.b64encode(chunk_bytes).decode('utf-8'))

        return result
    
    chunks = chunker(base64_image=documents["text"].iloc[0])

    # Create 4 rows, one for each chunk
    print(len(chunks))
    output_data = []
    for i, chunk in enumerate(chunks):
        output_data.append({
            "id": f"{documents['id'].iloc[0]}_{i}",
            "text": chunk,
            "document_ids": documents["id"].iloc[0],
            "n_tokens": len(chunk),
            "doc_type": documents["doc_type"].iloc[0]
        })

    output = pd.DataFrame(output_data)

    return output
    


@weave.op
def create_base_text_units(
    documents: pd.DataFrame,
    callbacks: WorkflowCallbacks,
    group_by_columns: list[str],
    size: int,
    overlap: int,
    encoding_model: str,
    strategy: ChunkStrategyType,
    prepend_metadata: bool = False,
    chunk_size_includes_metadata: bool = False,
) -> pd.DataFrame:
    """All the steps to transform base text_units."""
    # print("Creating base text units with chunk size:", size)
    sort = documents.sort_values(by=["id"], ascending=[True])

    sort["text_with_ids"] = list(
        zip(*[sort[col] for col in ["id", "text"]], strict=True)
    )

    agg_dict = {"text_with_ids": list}
    if "metadata" in documents:
        agg_dict["metadata"] = "first"  # type: ignore

    aggregated = (
        (
            sort.groupby(group_by_columns, sort=False)
            if len(group_by_columns) > 0
            else sort.groupby(lambda _x: True)
        )
        .agg(agg_dict)
        .reset_index()
    )
    aggregated.rename(columns={"text_with_ids": "texts"}, inplace=True)

    def chunker(row: pd.Series) -> Any:
        line_delimiter = ".\n"
        metadata_str = ""
        metadata_tokens = 0

        if prepend_metadata and "metadata" in row:
            metadata = row["metadata"]
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            if isinstance(metadata, dict):
                metadata_str = (
                    line_delimiter.join(f"{k}: {v}" for k, v in metadata.items())
                    + line_delimiter
                )

            if chunk_size_includes_metadata:
                encode, _ = get_encoding_fn(encoding_model)
                metadata_tokens = len(encode(metadata_str))
                if metadata_tokens >= size:
                    message = "Metadata tokens exceeds the maximum tokens per chunk. Please increase the tokens per chunk."
                    raise ValueError(message)

        chunked = chunk_text(
            pd.DataFrame([row]).reset_index(drop=True),
            column="texts",
            size=size - metadata_tokens,
            overlap=overlap,
            encoding_model=encoding_model,
            strategy=strategy,
            callbacks=callbacks,
        )[0]

        if prepend_metadata:
            for index, chunk in enumerate(chunked):
                if isinstance(chunk, str):
                    chunked[index] = metadata_str + chunk
                else:
                    chunked[index] = (
                        (chunk[0], metadata_str + chunk[1], chunk[2]) if chunk else None
                    )

        row["chunks"] = chunked
        return row

    # Track progress of row-wise apply operation
    total_rows = len(aggregated)
    logger.info("Starting chunking process for %d documents", total_rows)

    def chunker_with_logging(row: pd.Series, row_index: int) -> Any:
        """Add logging to chunker execution."""
        result = chunker(row)
        logger.info("chunker progress:  %d/%d", row_index + 1, total_rows)
        return result

    aggregated = aggregated.apply(
        lambda row: chunker_with_logging(row, row.name), axis=1
    )

    aggregated = cast("pd.DataFrame", aggregated[[*group_by_columns, "chunks"]])
    aggregated = aggregated.explode("chunks")
    aggregated.rename(
        columns={
            "chunks": "chunk",
        },
        inplace=True,
    )
    aggregated["id"] = aggregated.apply(
        lambda row: gen_sha512_hash(row, ["chunk"]), axis=1
    )
    aggregated[["document_ids", "chunk", "n_tokens"]] = pd.DataFrame(
        aggregated["chunk"].tolist(), index=aggregated.index
    )
    # rename for downstream consumption
    aggregated.rename(columns={"chunk": "text"}, inplace=True)

    return cast(
        "pd.DataFrame", aggregated[aggregated["text"].notna()].reset_index(drop=True)
    )
