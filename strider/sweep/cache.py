"""
Persistent disk cache for thermodynamic computations.

Uses sqlite3 (stdlib) with pickle serialization. WAL mode enables
safe concurrent reads/writes across workers.
"""

from __future__ import annotations
import hashlib
import os
import pickle
import sqlite3
import time
from pathlib import Path
from typing import Any


class DiskCache:
    """
    Cross-session persistent cache keyed by SHA-256 hash of inputs.

    Parameters
    ----------
    path        : Path to the sqlite3 database file
    max_size_mb : Maximum database size before LRU eviction
    ttl_days    : Time-to-live in days (None = never expire)
    """

    _CREATE = """
    CREATE TABLE IF NOT EXISTS cache (
        key      TEXT PRIMARY KEY,
        value    BLOB NOT NULL,
        accessed REAL NOT NULL,
        created  REAL NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_accessed ON cache(accessed);
    """

    def __init__(
        self,
        path: str | Path = "~/.strider/cache.db",
        max_size_mb: float = 500.0,
        ttl_days: float | None = None,
    ) -> None:
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.ttl_seconds = ttl_days * 86400 if ttl_days is not None else None
        self._hits = 0
        self._misses = 0

        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(self._CREATE)
        self._conn.commit()

        if self.ttl_seconds is not None:
            self._evict_expired()

    def get(self, key: str) -> Any | None:
        """Return the cached value for key, or None on a miss. Updates access timestamp."""
        row = self._conn.execute(
            "SELECT value FROM cache WHERE key=?", (key,)
        ).fetchone()
        if row is None:
            self._misses += 1
            return None
        self._conn.execute(
            "UPDATE cache SET accessed=? WHERE key=?", (time.time(), key)
        )
        self._conn.commit()
        self._hits += 1
        return pickle.loads(row[0])

    def set(self, key: str, value: Any) -> None:
        """Persist value under key. Triggers LRU eviction if database exceeds max_size_mb."""
        now = time.time()
        blob = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        self._conn.execute(
            "INSERT OR REPLACE INTO cache(key,value,accessed,created) VALUES(?,?,?,?)",
            (key, blob, now, now),
        )
        self._conn.commit()
        self._maybe_evict()

    def stats(self) -> dict:
        """Return a dict with hits, misses, hit_rate, entries, and size_mb."""
        size_bytes = os.path.getsize(self.path) if self.path.exists() else 0
        count = self._conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total else 0.0,
            "entries": count,
            "size_mb": size_bytes / 1024 / 1024,
        }

    def clear(self) -> None:
        """Delete all entries from the cache database."""
        self._conn.execute("DELETE FROM cache")
        self._conn.commit()

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()

    def __enter__(self): return self
    def __exit__(self, *_): self.close()

    # ─── internals ───────────────────────────────────────────────────────────

    def _maybe_evict(self) -> None:
        """Evict the oldest 20% of entries when the database exceeds max_size_bytes."""
        size = os.path.getsize(self.path) if self.path.exists() else 0
        if size > self.max_size_bytes:
            # Delete 20% of oldest entries
            count = self._conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
            n_evict = max(1, count // 5)
            self._conn.execute(
                "DELETE FROM cache WHERE key IN "
                "(SELECT key FROM cache ORDER BY accessed ASC LIMIT ?)",
                (n_evict,),
            )
            self._conn.commit()
            self._conn.execute("VACUUM")
            self._conn.commit()

    def _evict_expired(self) -> None:
        """Remove all entries older than ttl_seconds from the database."""
        cutoff = time.time() - self.ttl_seconds
        self._conn.execute("DELETE FROM cache WHERE created < ?", (cutoff,))
        self._conn.commit()

    @staticmethod
    def make_key(*args) -> str:
        """Build a SHA-256 cache key from an arbitrary sequence of arguments."""
        raw = "|".join(str(a) for a in args)
        return hashlib.sha256(raw.encode()).hexdigest()
