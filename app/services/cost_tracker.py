from datetime import datetime, timezone

from motor.motor_asyncio import AsyncIOMotorDatabase

from app.config import cfg
from app.models.schemas import CostMetadata
from app.utils.logger import get_logger

logger = get_logger(__name__)

COST_PER_1K_TOKENS: dict[str, dict[str, float]] = {
    "gemini-1.5-flash": {"in": 0.000075, "out": 0.0003},
    "gemini-1.5-pro":   {"in": 0.00125,  "out": 0.005},
}


def calculate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    """Return estimated USD cost for a single inference call."""
    rates = COST_PER_1K_TOKENS.get(model)
    if rates is None:
        logger.warning("unknown model for cost calculation", extra={"model": model})
        return 0.0
    return (tokens_in / 1000) * rates["in"] + (tokens_out / 1000) * rates["out"]


async def write_cost(cost_metadata: CostMetadata, db: AsyncIOMotorDatabase) -> None:
    """Persist cost record to MongoDB; intended to run as a fire-and-forget task."""
    doc = {
        **cost_metadata.model_dump(),
        "recorded_at": datetime.now(timezone.utc),
    }
    try:
        await db[cfg.COSTS_COLLECTION].insert_one(doc)
        logger.info(
            "cost record written",
            extra={
                "model": cost_metadata.model_used,
                "cost_usd": cost_metadata.estimated_cost_usd,
            },
        )
    except Exception as exc:
        # non-fatal: cost tracking must never break the query path
        logger.error(
            f"failed to write cost record: {exc}",
            extra={"model": cost_metadata.model_used},
        )
