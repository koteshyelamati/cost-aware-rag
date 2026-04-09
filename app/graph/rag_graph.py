from typing import Any, TypedDict

from langgraph.graph import StateGraph, END

from app.models.schemas import ComplexityResult, CostMetadata, RetrievedChunk
from app.services import classifier, retriever, generator, cost_tracker
from app.utils.logger import get_logger

logger = get_logger(__name__)


class RAGState(TypedDict):
    query: str
    query_embedding: list[float]
    complexity: ComplexityResult | None
    retrieved_chunks: list[RetrievedChunk]
    answer: str
    cost_metadata: CostMetadata | None
    cache_hit: bool


def classify_node(state: RAGState) -> RAGState:
    """Route query to simple or complex tier via heuristic scoring."""
    result = classifier.classify(state["query"])
    return {**state, "complexity": result}


async def retrieve_node(state: RAGState, *, db, collection_name: str) -> RAGState:
    """Fetch top-k chunks from MongoDB vector index."""
    chunks = await retriever.retrieve(
        embedding=state["query_embedding"],
        top_k=5,
        db=db,
        collection_name=collection_name,
    )
    return {**state, "retrieved_chunks": chunks}


async def generate_node(state: RAGState) -> RAGState:
    """Call Gemini with retrieved context and populate answer + cost_metadata."""
    from app.config import cfg

    complexity = state["complexity"]
    model_name = cfg.COMPLEX_MODEL if (complexity and complexity.tier == "complex") else cfg.SIMPLE_MODEL

    gen_response = await generator.generate(
        query=state["query"],
        chunks=state["retrieved_chunks"],
        model_name=model_name,
    )

    cost_usd = cost_tracker.calculate_cost(
        model=model_name,
        tokens_in=gen_response.tokens_in,
        tokens_out=gen_response.tokens_out,
    )

    cost_meta = CostMetadata(
        model_used=model_name,
        tokens_in=gen_response.tokens_in,
        tokens_out=gen_response.tokens_out,
        estimated_cost_usd=cost_usd,
        cache_hit=False,
        latency_ms=0.0,  # caller patches this with wall-clock time
    )

    return {**state, "answer": gen_response.answer, "cost_metadata": cost_meta}


def build_graph(db, collection_name: str) -> Any:
    """Compile the RAG StateGraph with injected Motor db handle."""

    async def _retrieve(state: RAGState) -> RAGState:
        return await retrieve_node(state, db=db, collection_name=collection_name)

    graph: StateGraph = StateGraph(RAGState)
    graph.add_node("classify", classify_node)
    graph.add_node("retrieve", _retrieve)
    graph.add_node("generate", generate_node)

    graph.set_entry_point("classify")
    graph.add_edge("classify", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()
