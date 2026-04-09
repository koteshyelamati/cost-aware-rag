import asyncio

from google import genai
from google.genai import types

from app.config import cfg
from app.models.schemas import GeneratorResponse, RetrievedChunk
from app.utils.logger import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = (
    "Answer ONLY using the provided context chunks. "
    "Cite sources as [chunk_0], [chunk_2] etc. "
    "If context is insufficient return exactly: "
    "I could not find relevant information in the provided documents."
)
_INSUFFICIENT = "I could not find relevant information in the provided documents."

_client = genai.Client(api_key=cfg.GEMINI_API_KEY)


def _build_context(chunks: list[RetrievedChunk]) -> str:
    return "\n".join(f"[chunk_{i}] {c.text}" for i, c in enumerate(chunks))


def _extract_cited_ids(answer: str, chunk_count: int) -> list[str]:
    return [f"chunk_{i}" for i in range(chunk_count) if f"[chunk_{i}]" in answer]


def _call_gemini(model_name: str, prompt: str) -> types.GenerateContentResponse:
    return _client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(system_instruction=_SYSTEM_PROMPT),
    )


async def generate(
    query: str,
    chunks: list[RetrievedChunk],
    model_name: str,
) -> GeneratorResponse:
    """Generate a grounded answer from retrieved chunks using the specified Gemini model."""
    if not chunks:
        return GeneratorResponse(
            answer=_INSUFFICIENT,
            cited_chunk_ids=[],
            tokens_in=0,
            tokens_out=0,
        )

    context = _build_context(chunks)
    prompt = f"{context}\n\nQuestion: {query}"

    response = await asyncio.to_thread(_call_gemini, model_name, prompt)
    answer = response.text.strip()

    usage = response.usage_metadata
    tokens_in: int = usage.prompt_token_count if usage else 0
    tokens_out: int = usage.candidates_token_count if usage else 0

    cited = _extract_cited_ids(answer, len(chunks))

    logger.info(
        "generation complete",
        extra={
            "model": model_name,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cited_count": len(cited),
        },
    )
    return GeneratorResponse(
        answer=answer,
        cited_chunk_ids=cited,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
    )
