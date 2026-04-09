import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile

from app.dependencies import get_db, get_embedder
from app.models.schemas import IngestResponse
from app.services.chunker import chunk
from app.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)

_ALLOWED_EXTENSIONS = {".txt", ".md", ".pdf"}


async def _ingest_background(
    raw: bytes,
    doc_id: str,
    source_file: str,
    db,
    embedder,
) -> None:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        logger.error(f"failed to decode file {source_file}: {exc}")
        return

    chunks = chunk(text, doc_id)
    texts = [c.text for c in chunks]
    embeddings = await embedder.embed(texts)

    docs = [
        {
            "text": c.text,
            "embedding": embeddings[i],
            "doc_id": c.doc_id,
            "chunk_index": c.index,
            "source_file": source_file,
            "created_at": datetime.now(timezone.utc),
        }
        for i, c in enumerate(chunks)
    ]

    from app.config import cfg

    await db[cfg.COLLECTION_NAME].insert_many(docs)
    logger.info(
        "ingest complete",
        extra={"doc_id": doc_id, "chunks": len(docs), "source_file": source_file},
    )


@router.post("/ingest", response_model=IngestResponse, status_code=202)
async def ingest(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    db=Depends(get_db),
    embedder=Depends(get_embedder),
) -> IngestResponse:
    """Accept a document upload, validate extension, and queue async ingestion."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"unsupported file type '{suffix}': allowed {sorted(_ALLOWED_EXTENSIONS)}",
        )

    request_id = str(uuid.uuid4())
    doc_id = str(uuid.uuid4())
    raw = await file.read()

    background_tasks.add_task(
        _ingest_background,
        raw=raw,
        doc_id=doc_id,
        source_file=file.filename or "unknown",
        db=db,
        embedder=embedder,
    )

    logger.info(
        "ingest queued",
        extra={"request_id": request_id, "doc_id": doc_id, "source_file": file.filename},
    )
    return IngestResponse(request_id=request_id, doc_id=doc_id, status="processing")
