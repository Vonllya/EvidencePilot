"""Small in-memory cache used by retrieval implementations."""

from typing import Generic, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class MemoryCache(dict[K, V], Generic[K, V]):
    """Typed process-local cache with explicit lookup semantics."""

    def get_cached(self, key: K) -> V | None:
        return self.get(key)
