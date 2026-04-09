import tiktoken

from app.models.schemas import Chunk

_ENCODING = tiktoken.get_encoding("cl100k_base")
_CHUNK_SIZE = 512
_OVERLAP = 50


def chunk(text: str, doc_id: str) -> list[Chunk]:
    """Split text into overlapping token windows and return Chunk list."""
    tokens = _ENCODING.encode(text)
    chunks: list[Chunk] = []
    start = 0
    index = 0

    while start < len(tokens):
        end = start + _CHUNK_SIZE
        window = tokens[start:end]
        chunks.append(
            Chunk(
                text=_ENCODING.decode(window),
                index=index,
                token_count=len(window),
                doc_id=doc_id,
            )
        )
        index += 1
        next_start = end - _OVERLAP
        # guard against infinite loop on very short token sequences
        start = next_start if next_start > start else start + 1

    return chunks
