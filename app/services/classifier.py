import re

import tiktoken

from app.config import cfg
from app.models.schemas import ComplexityResult
from app.utils.logger import get_logger

logger = get_logger(__name__)

_ENC = tiktoken.get_encoding("cl100k_base")
_TOKEN_THRESHOLD = 80
_COMPLEX_KEYWORDS = frozenset(
    [
        "compare",
        "analyze",
        "trade-off",
        "tradeoff",
        "difference between",
        "pros and cons",
        "explain why",
        "architecture",
        "design",
    ]
)


def classify(query: str) -> ComplexityResult:
    """Score query complexity via heuristics; no LLM involved."""
    score = 0.0
    signals: list[str] = []
    q_lower = query.lower()

    token_count = len(_ENC.encode(query))
    if token_count > _TOKEN_THRESHOLD:
        score += 0.3
        signals.append(f"token_count={token_count} > {_TOKEN_THRESHOLD}")

    matched = [kw for kw in _COMPLEX_KEYWORDS if kw in q_lower]
    if matched:
        score += 0.4
        signals.append(f"keywords={matched}")

    # multi-part: connective "and" plus at least one question mark
    if re.search(r"\band\b", q_lower) and "?" in query:
        score += 0.3
        signals.append("multi_part_question")

    score = min(score, 1.0)
    tier = "complex" if score >= cfg.COMPLEXITY_THRESHOLD else "simple"

    logger.info(
        "query classified",
        extra={"tier": tier, "score": round(score, 3), "signals": signals},
    )
    return ComplexityResult(score=score, tier=tier, signals=signals)
