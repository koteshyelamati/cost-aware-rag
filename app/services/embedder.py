import asyncio

from google import genai
from google.genai import types

from app.config import cfg
from app.utils.logger import get_logger

logger = get_logger(__name__)

_BATCH_LIMIT = 100
_EMBED_MODEL = "gemini-embedding-001"
_EMBED_DIMS = 768
_BACKOFF = (1, 2, 4)


class EmbeddingError(RuntimeError):
    pass


class Embedder:
    """Embed text batches via Gemini text-embedding-004."""

    def __init__(self) -> None:
        self._client = genai.Client(api_key=cfg.GEMINI_API_KEY)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one 768-dim vector per input text, batched at 100."""
        all_vectors: list[list[float]] = []
        for i in range(0, len(texts), _BATCH_LIMIT):
            batch = texts[i : i + _BATCH_LIMIT]
            vectors = await self._embed_batch_with_retry(batch, batch_start=i)
            all_vectors.extend(vectors)
        return all_vectors

    async def _embed_batch_with_retry(
        self, batch: list[str], *, batch_start: int
    ) -> list[list[float]]:
        last_exc: Exception | None = None
        for attempt, delay in enumerate(_BACKOFF, start=1):
            try:
                return await asyncio.to_thread(self._call_api, batch)
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "embedding batch failed, will retry",
                    extra={"attempt": attempt, "batch_start": batch_start, "error": str(exc)},
                )
                await asyncio.sleep(delay)

        raise EmbeddingError(
            f"failed to embed batch starting at index {batch_start} after {len(_BACKOFF)} attempts: {last_exc}"
        ) from last_exc

    def _call_api(self, batch: list[str]) -> list[list[float]]:
        result = self._client.models.embed_content(
            model=_EMBED_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(output_dimensionality=_EMBED_DIMS),
        )
        return [e.values for e in result.embeddings]
