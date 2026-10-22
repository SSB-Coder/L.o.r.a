"""Exact CPU retrieval without loading an optional native FAISS DLL.

NumPy targets small local/session indexes. FAISS is an explicit opt-in for
larger collections after its native runtime has been validated.
"""

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

from lora.core.paths import project_path


@lru_cache(maxsize=1)
def _configured_backend() -> str:
    path = project_path('config/lora.yaml')
    if not path.is_file():
        return 'numpy'
    data = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    return str(data.get('grounding', {}).get('vector_store', 'numpy')).lower()


def backend_name() -> str:
    name = os.getenv('LORA_VECTOR_BACKEND', _configured_backend()).lower()
    if name not in {'numpy', 'faiss'}:
        raise ValueError(f'Unsupported vector backend: {name}')
    return name


class NumpyFlatIndex:
    def __init__(self, dimension: int):
        self.d = dimension
        self.vectors = np.empty((0, dimension), dtype=np.float32)

    @property
    def ntotal(self) -> int:
        return len(self.vectors)

    def add(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.d or not np.all(np.isfinite(values)):
            raise ValueError("Invalid embedding shape or non-finite vector")
        self.vectors = np.concatenate((self.vectors, values))

    def search(self, queries: np.ndarray, top_k: int):
        scores = np.asarray(queries, dtype=np.float32) @ self.vectors.T
        order = np.argsort(-scores, axis=1, kind="stable")[:, :max(0, top_k)]
        return np.take_along_axis(scores, order, axis=1), order


def make_index(dimension: int):
    if backend_name() == "faiss":
        import faiss
        return faiss.IndexFlatIP(dimension)
    return NumpyFlatIndex(dimension)


def save_index(index, path: str | Path) -> None:
    if isinstance(index, NumpyFlatIndex):
        with Path(path).open('wb') as file:
            np.save(file, index.vectors, allow_pickle=False)
    else:
        import faiss
        faiss.write_index(index, str(path))


def load_index(path: str | Path):
    try:
        vectors = np.load(path, allow_pickle=False)
    except ValueError as exc:
        if backend_name() != "faiss":
            raise ValueError("Legacy FAISS index preserved. Enable LORA_VECTOR_BACKEND=faiss with a working FAISS runtime, or re-index the original files.") from exc
        import faiss
        return faiss.read_index(str(path))
    if vectors.ndim != 2:
        raise ValueError("Invalid persisted vector matrix")
    index = NumpyFlatIndex(vectors.shape[1])
    index.add(vectors)
    return index
