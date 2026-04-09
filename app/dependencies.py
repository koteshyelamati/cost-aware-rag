from fastapi import Header, HTTPException, Request

from app.config import cfg


def get_db(request: Request):
    """Return the Motor database bound to app state at startup."""
    return request.app.state.db


def get_redis(request: Request):
    """Return the redis.asyncio client bound to app state at startup."""
    return request.app.state.redis


def get_embedder(request: Request):
    """Return the Embedder singleton bound to app state at startup."""
    return request.app.state.embedder


async def verify_api_key(x_api_key: str = Header(alias="X-API-Key")) -> None:
    """Reject requests whose X-API-Key header does not match cfg.API_KEY."""
    if x_api_key != cfg.API_KEY:
        raise HTTPException(status_code=401, detail="invalid API key")
