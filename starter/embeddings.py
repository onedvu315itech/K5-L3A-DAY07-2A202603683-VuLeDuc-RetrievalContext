
"""Choose a real embedding provider (API or local) and implement this contract."""
# [ADDED FOR OPENAI] Standard library and HTTP imports for OpenAI embeddings
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Protocol
import requests
from sentence_transformers import SentenceTransformer


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in the same order."""
        ...

    def embed_query(self, question: str) -> list[float]:
        """Return a query vector compatible with the stored document vectors."""
        ...


class MyEmbedder:
    """Local embedding adapter using Sentence Transformers."""

    def __init__(self, model_name: str) -> None:
        if not model_name:
            raise ValueError("model_name cannot be empty")

        self.model_name = model_name

        # Load the actual pretrained embedding model
        self.model = SentenceTransformer(model_name)

        # Get the model's actual embedding dimensions
        self.dimensions = self.model.get_embedding_dimension()

        if self.dimensions is None:
            raise ValueError("Cannot determine embedding dimensions")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate one normalized embedding vector per document."""

        if not texts:
            return []

        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("All document texts must be non-empty strings")

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        return [
            [float(value) for value in vector]
            for vector in embeddings
        ]

    def embed_query(self, question: str) -> list[float]:
        """Generate a normalized query embedding."""

        if not question.strip():
            raise ValueError("Question cannot be empty")

        embedding = self.model.encode(
            question,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        return [float(value) for value in embedding]


# [ADDED FOR OPENAI] New embedding adapter using OpenAI API (OPENAI_API_KEY)
class OpenAIEmbedder:
    """OpenAI API embedding adapter using OPENAI_API_KEY."""

    DEFAULT_MODEL = "text-embedding-3-small"
    DEFAULT_DIMENSIONS = 1536
    BATCH_SIZE = 200
    MAX_RETRIES = 6

    def __init__(
        self,
        model_name: str | None = None,
        api_key: str | None = None,
        dimensions: int | None = None,
        base_url: str | None = None,
    ) -> None:
        # [ADDED FOR OPENAI] Resolve API key from parameter or environment variables
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY is required to use OpenAIEmbedder"
            )

        # [ADDED FOR OPENAI] Resolve OpenAI embedding model name (fallback to text-embedding-3-small)
        if not model_name or model_name.startswith("sentence-transformers/"):
            self.model_name = (
                os.getenv("OPENAI_EMBEDDING_MODEL") or self.DEFAULT_MODEL
            )
        else:
            self.model_name = model_name

        # [ADDED FOR OPENAI] Configure output vector dimensionality (default 1536, <= 2000 for pgvector HNSW)
        env_dims = os.getenv("OPENAI_EMBEDDING_DIMENSIONS")
        self.dimensions = (
            dimensions
            if dimensions is not None
            else (int(env_dims) if env_dims else self.DEFAULT_DIMENSIONS)
        )

        if self.dimensions <= 0:
            raise ValueError("dimensions must be a positive integer")

        self.base_url = (
            base_url
            or os.getenv("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        ).rstrip("/")

        # [ADDED FOR OPENAI] Initialize in-memory query cache and persistent disk cache
        self._query_cache: dict[str, list[float]] = {}
        self._cache_path = (
            Path(__file__).resolve().parents[1]
            / "results"
            / ".openai_embedding_cache.json"
        )
        self._disk_cache: dict[str, list[float]] = self._load_disk_cache()

    def _load_disk_cache(self) -> dict[str, list[float]]:
        """Load cached embeddings from disk to avoid redundant API calls."""
        if self._cache_path.exists():
            try:
                with self._cache_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return data
            except Exception:
                pass
        return {}

    def _save_disk_cache(self) -> None:
        """Persist embeddings cache to disk after each batch."""
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self._cache_path.open("w", encoding="utf-8") as f:
                json.dump(self._disk_cache, f)
        except Exception:
            pass

    def _cache_key(self, text: str) -> str:
        """Create a deterministic cache key for (model, dimensions, text)."""
        raw = f"{self.model_name}:{self.dimensions}:{text}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _request_embeddings(self, inputs: list[str]) -> list[list[float]]:
        """Call OpenAI /v1/embeddings endpoint with automatic retry on 429 rate limits."""
        url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: dict = {
            "model": self.model_name,
            "input": inputs,
        }
        # text-embedding-3-* models support the dimensions parameter natively
        if self.model_name.startswith("text-embedding-3"):
            payload["dimensions"] = self.dimensions

        for attempt in range(1, self.MAX_RETRIES + 1):
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=60,
            )

            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                wait_seconds = float(retry_after) if retry_after else (5.0 * attempt)
                print(
                    f"[OpenAI] Rate limit (429) reached. "
                    f"Waiting {wait_seconds:.1f}s before retry {attempt}/{self.MAX_RETRIES}..."
                )
                time.sleep(wait_seconds)
                continue

            if not response.ok:
                raise RuntimeError(
                    f"OpenAI Embeddings API error ({response.status_code}): {response.text}"
                )

            body = response.json()
            data = body.get("data", [])
            if len(data) != len(inputs):
                raise RuntimeError(
                    f"OpenAI returned {len(data)} embeddings for {len(inputs)} inputs"
                )

            # Ensure embeddings are ordered by index
            data_sorted = sorted(data, key=lambda item: item["index"])
            return [
                self._normalize([float(v) for v in item["embedding"]])
                for item in data_sorted
            ]

        raise RuntimeError("Exceeded maximum retries for OpenAI embeddings due to rate limits")

    @staticmethod
    def _normalize(vector: list[float]) -> list[float]:
        """L2-normalize an embedding vector for cosine similarity."""
        norm = math.sqrt(sum(float(v) * float(v) for v in vector))
        if norm == 0.0:
            return [float(v) for v in vector]
        return [float(v) / norm for v in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate one normalized OpenAI embedding vector per document chunk."""

        if not texts:
            return []

        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("All document texts must be non-empty strings")

        results: list[list[float] | None] = [None] * len(texts)
        missing_indices: list[int] = []

        # [ADDED FOR OPENAI] Check disk cache first so already-embedded chunks cost 0 tokens
        for idx, text in enumerate(texts):
            key = self._cache_key(text)
            if key in self._disk_cache:
                results[idx] = self._disk_cache[key]
            else:
                missing_indices.append(idx)

        if not missing_indices:
            print(f"[OpenAI] Loaded all {len(texts)} document embeddings from local cache.")
            return [vec for vec in results if vec is not None]

        print(
            f"[OpenAI] Embedding {len(missing_indices)}/{len(texts)} chunks via OpenAI API "
            f"({len(texts) - len(missing_indices)} cached)..."
        )

        # [ADDED FOR OPENAI] Batch requests with automatic retry and cache persistence
        for start in range(0, len(missing_indices), self.BATCH_SIZE):
            batch_indices = missing_indices[start:start + self.BATCH_SIZE]
            batch_texts = [texts[i] for i in batch_indices]

            vectors = self._request_embeddings(batch_texts)

            for idx, vec in zip(batch_indices, vectors):
                results[idx] = vec
                self._disk_cache[self._cache_key(texts[idx])] = vec

            self._save_disk_cache()
            done_count = min(start + self.BATCH_SIZE, len(missing_indices))
            print(f"[OpenAI] Embedded {done_count}/{len(missing_indices)} chunks")

        return [vec for vec in results if vec is not None]

    def embed_query(self, question: str) -> list[float]:
        """Generate a normalized OpenAI query embedding."""

        if not question.strip():
            raise ValueError("Question cannot be empty")

        # [ADDED FOR OPENAI] Reuse cached query vector across strategies A, B, and C
        if question in self._query_cache:
            return self._query_cache[question]

        key = self._cache_key(question)
        if key in self._disk_cache:
            vector = self._disk_cache[key]
            self._query_cache[question] = vector
            return vector

        vector = self._request_embeddings([question])[0]
        self._query_cache[question] = vector
        self._disk_cache[key] = vector
        self._save_disk_cache()
        return vector

