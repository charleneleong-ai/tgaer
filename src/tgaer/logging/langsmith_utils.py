from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional

from langsmith import Client


_client: Optional[Client] = None


def get_client() -> Client:
    global _client
    if _client is None:
        _client = Client()
    return _client


@contextmanager
def traced_run(
    name: str,
    *,
    run_type: str = "chain",
    metadata: Optional[Dict[str, Any]] = None,
) -> Generator[None, None, None]:
    client = get_client()
    run = client.create_run(
        name=name,
        run_type=run_type,
        extra=metadata or {},
    )
    try:
        yield
    finally:
        client.update_run(run.id, end_time=None)
