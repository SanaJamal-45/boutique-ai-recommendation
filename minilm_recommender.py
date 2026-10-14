
import os
from pathlib import Path
from functools import lru_cache

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


MODEL_DIR = Path(
    os.environ.get(
        "MINILM_MODEL_DIR",
        str(Path.home() / "minilm-feasibility/model")
    )
)


@lru_cache(maxsize=1)
def get_runtime():
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1

    session = ort.InferenceSession(
        str(MODEL_DIR / "onnx/model.onnx"),
        sess_options=options,
        providers=["CPUExecutionProvider"]
    )

    tokenizer = Tokenizer.from_file(
        str(MODEL_DIR / "tokenizer.json")
    )
    tokenizer.enable_truncation(max_length=128)
    tokenizer.enable_padding(
        pad_id=0,
        pad_token="[PAD]"
    )

    return session, tokenizer


def generate_embeddings(texts):
    session, tokenizer = get_runtime()

    encoded = tokenizer.encode_batch(texts)

    mask = np.array(
        [item.attention_mask for item in encoded],
        dtype=np.int64
    )

    inputs = {
        "input_ids": np.array(
            [item.ids for item in encoded],
            dtype=np.int64
        ),
        "attention_mask": mask,
        "token_type_ids": np.array(
            [item.type_ids for item in encoded],
            dtype=np.int64
        )
    }

    output = session.run(
        ["last_hidden_state"], inputs
    )[0]

    expanded_mask = mask[..., None].astype(np.float32)

    embeddings = (output * expanded_mask).sum(axis=1)
    embeddings /= np.maximum(
        expanded_mask.sum(axis=1), 1e-9
    )

    norms = np.linalg.norm(
        embeddings, axis=1, keepdims=True
    )
    embeddings /= np.maximum(norms, 1e-9)

    return embeddings


def product_text(row):
    product_id, name, description, categories = row
    return f"{name}. {description}. {' '.join(categories)}"


@lru_cache(maxsize=2)
def get_catalog_embeddings(product_rows):
    texts = [product_text(row) for row in product_rows]
    return generate_embeddings(texts)


def recommend_products(products, selected_product_ids, limit=5):
    product_rows = tuple(
        (
            p.id,
            p.name,
            p.description,
            tuple(p.categories)
        )
        for p in products
    )

    if not product_rows or limit <= 0:
        return []

    # Cached until product descriptions or categories change
    catalog_embeddings = get_catalog_embeddings(product_rows)

    selected_ids = set(selected_product_ids)

    selected_rows = [
        row for row in product_rows
        if row[0] in selected_ids
    ]

    # Execute neural-network inference for each request
    if selected_rows:
        query_texts = [
            product_text(row) for row in selected_rows
        ]
    else:
        query_texts = ["everyday lifestyle shopping products"]

    query_embeddings = generate_embeddings(query_texts)

    query_embedding = query_embeddings.mean(axis=0)
    query_embedding /= max(
        np.linalg.norm(query_embedding), 1e-9
    )

    scores = catalog_embeddings @ query_embedding

    candidates = [
        (row[0], float(scores[i]))
        for i, row in enumerate(product_rows)
        if row[0] not in selected_ids
    ]

    candidates.sort(key=lambda x: (-x[1], x[0]))

    return [product_id for product_id, _ in candidates[:limit]]
