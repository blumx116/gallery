from contextlib import contextmanager
from typing import Iterator

biqguery_context_state: list[str] = []


@contextmanager 
def bq_context(name: str) -> Iterator[None]:
    # NOTE: very much not concurrency safe
    biqguery_context_state.append(name)
    yield None
    biqguery_context_state.pop()


def bq_name(name: str) -> str:




