
"""Store and search vectors using PostgreSQL + pgvector."""

from typing import Any
import math

import psycopg
from pgvector.psycopg import register_vector
from psycopg.types.json import Jsonb

try:
    from .models import Chunk
except ImportError:
    from models import Chunk


class PgVectorStore:
    TABLE_NAME = "rag_chunks"

    def __init__(self, database_url: str, dimensions: int) -> None:
        """Retain connection settings and validate embedding dimensions."""

        if not database_url or not database_url.strip():
            raise ValueError("database_url cannot be empty")

        if type(dimensions) is not int or dimensions <= 0:
            raise ValueError("dimensions must be a positive integer")

        self.database_url = database_url
        self.dimensions = dimensions

    def _connect(self):
        """Create a PostgreSQL connection and register pgvector support."""

        conn = psycopg.connect(self.database_url)
        register_vector(conn)
        return conn

    def create_schema(self) -> None:
        """Create the vector extension, table, and similarity index."""

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "CREATE EXTENSION IF NOT EXISTS vector"
                )

                # [ADDED FOR OPENAI] If table already exists with a different vector dimension
                # (for example, switching between 384-dim local MyEmbedder and 1536-dim OpenAIEmbedder),
                # recreate the table so the embedding column matches self.dimensions.
                cur.execute(
                    """
                    SELECT atttypmod
                    FROM pg_attribute
                    WHERE attrelid = to_regclass(%s)
                      AND attname = 'embedding'
                      AND NOT attisdropped
                    """,
                    (self.TABLE_NAME,)
                )
                existing_dim = cur.fetchone()
                if existing_dim and existing_dim[0] != -1 and existing_dim[0] != self.dimensions:
                    print(
                        f"[Store] Vector dimension changed ({existing_dim[0]} -> {self.dimensions}). "
                        f"Recreating table {self.TABLE_NAME}..."
                    )
                    cur.execute(f"DROP TABLE IF EXISTS {self.TABLE_NAME}")

                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self.TABLE_NAME} (
                        strategy TEXT NOT NULL,
                        chunk_id TEXT NOT NULL,
                        document_id TEXT NOT NULL,
                        source TEXT NOT NULL,
                        page INTEGER NOT NULL,
                        text TEXT NOT NULL,
                        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                        embedding VECTOR({self.dimensions}) NOT NULL,

                        PRIMARY KEY (strategy, chunk_id)
                    )
                    """
                )

                cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS
                    rag_chunks_embedding_hnsw_idx
                    ON {self.TABLE_NAME}
                    USING hnsw (embedding vector_cosine_ops)
                    """
                )

    def upsert(
        self,
        strategy: str,
        chunks: list[Chunk],
        vectors: list[list[float]]
    ) -> None:
        """Replace existing document chunks and insert their embeddings."""

        if not strategy or not strategy.strip():
            raise ValueError("strategy cannot be empty")

        if len(chunks) != len(vectors):
            raise ValueError(
                "Number of chunks must match number of vectors"
            )

        if not chunks:
            return

        # Validate chunk IDs before writing.
        chunk_ids = [chunk.chunk_id for chunk in chunks]

        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("Duplicate chunk IDs detected")

        # Collect document IDs represented in this ingestion batch.
        document_ids = list({
            chunk.document_id for chunk in chunks
        })

        rows = []

        for chunk, vector in zip(chunks, vectors):
            if len(vector) != self.dimensions:
                raise ValueError(
                    f"Expected vector dimension {self.dimensions}, "
                    f"received {len(vector)} for {chunk.chunk_id}"
                )

            if not all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
                for value in vector
            ):
                raise ValueError(
                    f"Invalid vector values for chunk {chunk.chunk_id}"
                )

            rows.append((
                strategy,
                chunk.chunk_id,
                chunk.document_id,
                chunk.source,
                chunk.page,
                chunk.text,
                Jsonb(chunk.metadata),
                vector
            ))

        # Delete old chunks and insert replacements in one transaction.
        # This removes obsolete chunks when a document changes.
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    DELETE FROM {self.TABLE_NAME}
                    WHERE strategy = %s
                    AND document_id = ANY(%s)
                    """,
                    (strategy, document_ids)
                )

                cur.executemany(
                    f"""
                    INSERT INTO {self.TABLE_NAME} (
                        strategy,
                        chunk_id,
                        document_id,
                        source,
                        page,
                        text,
                        metadata,
                        embedding
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    rows
                )

    def search(
        self,
        strategy: str,
        query_vector: list[float],
        top_k: int = 3,
        filters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Search chunks by cosine similarity and optional metadata filters."""

        if not strategy or not strategy.strip():
            raise ValueError("strategy cannot be empty")

        if type(top_k) is not int or top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        if len(query_vector) != self.dimensions:
            raise ValueError(
                f"Expected query vector dimension {self.dimensions}, "
                f"received {len(query_vector)}"
            )

        if not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            for value in query_vector
        ):
            raise ValueError("Query vector contains invalid values")

        # Convert Python list into pgvector textual representation.
        vector_param = "[" + ",".join(
            str(float(value)) for value in query_vector
        ) + "]"

        sql = f"""
            SELECT
                chunk_id,
                document_id,
                source,
                page,
                text,
                metadata,
                1 - (embedding <=> %s::vector) AS score
            FROM {self.TABLE_NAME}
            WHERE strategy = %s
        """

        params: list[Any] = [vector_param, strategy]

        # JSONB containment: all supplied metadata key/value pairs
        # must match the stored metadata.
        if filters:
            filters = filters.copy()

            # Filter using dedicated database columns
            if "document_id" in filters:
                sql += " AND document_id = %s"
                params.append(filters.pop("document_id"))

            if "source" in filters:
                sql += " AND source = %s"
                params.append(filters.pop("source"))

            if "page" in filters:
                sql += " AND page = %s"
                params.append(filters.pop("page"))

            # Apply remaining filters to JSONB metadata
            if filters:
                sql += " AND metadata @> %s"
                params.append(Jsonb(filters))

        sql += """
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """

        params.extend([vector_param, top_k])

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

        # Suppress per-row verbose printing during batch benchmark runs
        return [
            {
                "chunk_id": row[0],
                "document_id": row[1],
                "source": row[2],
                "page": row[3],
                "text": row[4],
                "metadata": row[5],
                "score": float(row[6])
            }
            for row in rows
        ]
