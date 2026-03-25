"""Cache Manager - SQLite caching, retry logic, and checkpoint saving"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Optional, Dict, Any
import sqlite3
import hashlib
import json
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
import threading
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════
# LLM CACHE MANAGER
# ════════════════════════════════════════════════

class LLMCacheManager:
    """
    SQLite-based cache for LLM responses

    Features:
    - Persistent SQLite storage (survives VS Code restarts)
    - Automatic TTL expiration
    - Thread-safe operations
    - LRU-style eviction when full
    """

    def __init__(
        self,
        db_path: str = "data/llm_cache.db",
        default_ttl_days: int = 7,
        max_entries: int = 10000
    ):
        self.db_path = db_path
        self.default_ttl = timedelta(days=default_ttl_days)
        self.max_entries = max_entries
        self._lock = threading.Lock()

        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._cleanup_expired()

        self.stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'invalidations': 0
        }

        logger.info(f"Cache initialized: {db_path} (TTL: {default_ttl_days}d)")

    def _init_db(self):
        """Initialize SQLite database schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS llm_cache (
                    key TEXT PRIMARY KEY,
                    prompt_hash TEXT NOT NULL,
                    response_data TEXT NOT NULL,
                    model TEXT NOT NULL,
                    prompt_version TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT,
                    access_count INTEGER DEFAULT 0,
                    last_accessed TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_expires
                ON llm_cache(expires_at)
            """)
            conn.commit()

    def _generate_key(
        self,
        prompt: str,
        model: str,
        prompt_version: str
    ) -> str:
        """Generate deterministic cache key"""
        composite = f"{prompt.strip().lower()}||{model}||{prompt_version}"
        return hashlib.sha256(composite.encode()).hexdigest()[:32]

    def get(
        self,
        prompt: str,
        model: str,
        prompt_version: str
    ) -> Optional[Dict]:
        """Get cached response if available and not expired"""
        key = self._generate_key(prompt, model, prompt_version)

        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT response_data, expires_at, access_count "
                    "FROM llm_cache WHERE key = ?",
                    (key,)
                )
                row = cursor.fetchone()

                if not row:
                    self.stats['misses'] += 1
                    return None

                response_json, expires_at, access_count = row

                # Check expiration
                if expires_at:
                    if datetime.now() > datetime.fromisoformat(expires_at):
                        conn.execute(
                            "DELETE FROM llm_cache WHERE key = ?", (key,)
                        )
                        conn.commit()
                        self.stats['misses'] += 1
                        return None

                # Update access tracking
                conn.execute(
                    "UPDATE llm_cache SET access_count = ?, "
                    "last_accessed = ? WHERE key = ?",
                    (access_count + 1, datetime.now().isoformat(), key)
                )
                conn.commit()
                self.stats['hits'] += 1

                try:
                    return json.loads(response_json)
                except json.JSONDecodeError:
                    logger.error(f"Corrupted cache entry: {key}")
                    return None

    def set(
        self,
        prompt: str,
        model: str,
        prompt_version: str,
        response_data: Dict,
        ttl_days: Optional[int] = None
    ):
        """Cache a response"""
        key = self._generate_key(prompt, model, prompt_version)
        now = datetime.now()
        expires = now + (
            timedelta(days=ttl_days) if ttl_days else self.default_ttl
        )

        entry = {
            'key': key,
            'prompt_hash': hashlib.md5(prompt.encode()).hexdigest()[:16],
            'response_data': json.dumps(response_data),
            'model': model,
            'prompt_version': prompt_version,
            'created_at': now.isoformat(),
            'expires_at': expires.isoformat(),
            'access_count': 0,
            'last_accessed': now.isoformat()
        }

        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO llm_cache
                    (key, prompt_hash, response_data, model, prompt_version,
                     created_at, expires_at, access_count, last_accessed)
                    VALUES
                    (:key, :prompt_hash, :response_data, :model,
                     :prompt_version, :created_at, :expires_at,
                     :access_count, :last_accessed)
                """, entry)
                conn.commit()
                self.stats['sets'] += 1

        self._enforce_size_limit()

    def _enforce_size_limit(self):
        """Remove oldest entries if cache is full"""
        with sqlite3.connect(self.db_path) as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM llm_cache"
            ).fetchone()[0]

            if count > self.max_entries:
                to_remove = count - self.max_entries
                conn.execute("""
                    DELETE FROM llm_cache WHERE key IN (
                        SELECT key FROM llm_cache
                        ORDER BY last_accessed ASC LIMIT ?
                    )
                """, (to_remove,))
                conn.commit()
                logger.info(f"Trimmed {to_remove} old cache entries")

    def _cleanup_expired(self):
        """Remove expired entries on startup"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM llm_cache WHERE expires_at < ?",
                (datetime.now().isoformat(),)
            )
            conn.commit()
            if cursor.rowcount > 0:
                logger.info(f"Cleaned {cursor.rowcount} expired entries")

    def invalidate(
        self,
        prompt_version: Optional[str] = None,
        model: Optional[str] = None
    ):
        """Invalidate cache entries by version or model"""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                if prompt_version and model:
                    conn.execute(
                        "DELETE FROM llm_cache WHERE "
                        "prompt_version = ? AND model = ?",
                        (prompt_version, model)
                    )
                elif prompt_version:
                    conn.execute(
                        "DELETE FROM llm_cache WHERE prompt_version = ?",
                        (prompt_version,)
                    )
                elif model:
                    conn.execute(
                        "DELETE FROM llm_cache WHERE model = ?",
                        (model,)
                    )
                else:
                    conn.execute("DELETE FROM llm_cache")
                conn.commit()
                self.stats['invalidations'] += 1

    def get_stats(self) -> Dict:
        """Get cache statistics"""
        total = self.stats['hits'] + self.stats['misses']
        hit_rate = (
            self.stats['hits'] / total * 100 if total > 0 else 0
        )

        with sqlite3.connect(self.db_path) as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM llm_cache"
            ).fetchone()[0]

        return {
            **self.stats,
            'hit_rate_percent': round(hit_rate, 2),
            'total_entries': count,
            'db_path': self.db_path
        }


# ════════════════════════════════════════════════
# CHECKPOINT MANAGER (Change 2 - NEW)
# Handles Colab disconnection recovery
# ════════════════════════════════════════════════

class CheckpointManager:
    """
    Saves batch processing progress to handle Colab disconnections

    When Colab disconnects mid-batch, progress is saved.
    On restart, processing resumes from last checkpoint.

    Usage:
        checkpoint = CheckpointManager()
        completed_ids, results = checkpoint.load()
        # ... process items ...
        checkpoint.save(completed_ids, results)  # every N items
        checkpoint.clear()  # when fully done
    """

    def __init__(
        self,
        checkpoint_path: str = "outputs/checkpoints/week5.json"
    ):
        self.path = Path(checkpoint_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, completed_ids: list, results: list):
        """
        Save current progress to disk

        Args:
            completed_ids: List of regulation_chunk_ids processed
            results: List of explanation dicts generated so far
        """
        with open(self.path, 'w') as f:
            json.dump({
                'completed_ids': completed_ids,
                'results': results,
                'saved_at': datetime.now().isoformat(),
                'count': len(completed_ids)
            }, f)
        logger.info(f"💾 Checkpoint saved: {len(completed_ids)} items")

    def load(self) -> tuple:
        """
        Load existing checkpoint if available

        Returns:
            (completed_ids, results)
            Both empty lists if no checkpoint exists
        """
        if not self.path.exists():
            logger.info("No checkpoint found, starting fresh")
            return [], []

        with open(self.path) as f:
            data = json.load(f)

        print(f" Checkpoint found: {data['count']} items already done")
        print(f"   Saved at: {data['saved_at']}")
        print(f"   Resuming from where we left off...")

        return data['completed_ids'], data['results']

    def clear(self):
        """Remove checkpoint after successful completion"""
        if self.path.exists():
            self.path.unlink()
            logger.info(" Checkpoint cleared - batch complete")

    def exists(self) -> bool:
        """Check if a checkpoint file exists"""
        return self.path.exists()


# ════════════════════════════════════════════════
# INTELLIGENT RETRY MANAGER
# ════════════════════════════════════════════════

class IntelligentRetryManager:
    """
    Manages LLM calls with caching, retry, and circuit breaker

    Features:
    - Cache-first lookup
    - Exponential backoff retry
    - Parse failure regeneration
    - Circuit breaker pattern
    """

    def __init__(
        self,
        cache_manager: LLMCacheManager,
        max_retries: int = 3,
        base_delay: float = 1.0
    ):
        self.cache = cache_manager
        self.max_retries = max_retries
        self.base_delay = base_delay

        self.retry_stats = {
            'total_calls': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'retries_needed': 0,
            'regenerations': 0,
            'failures': 0
        }

        self.circuit_breaker = {
            'failures': 0,
            'last_failure': None,
            'is_open': False,
            'threshold': 5,
            'timeout_seconds': 60
        }

    def execute(
        self,
        prompt: str,
        model: str,
        prompt_version: str,
        llm_client,
        parser,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1
    ) -> Dict:
        """
        Execute LLM call with caching and retry

        Returns:
            Dict with success, data, from_cache, attempts, error
        """
        self.retry_stats['total_calls'] += 1

        # Check cache first
        cached = self.cache.get(prompt, model, prompt_version)
        if cached:
            self.retry_stats['cache_hits'] += 1
            return {
                'success': True, 'data': cached,
                'from_cache': True, 'attempts': 0, 'error': None
            }

        self.retry_stats['cache_misses'] += 1

        # Check circuit breaker
        if self._is_circuit_open():
            return {
                'success': False, 'data': None,
                'from_cache': False, 'attempts': 0,
                'error': 'Circuit breaker open - too many failures'
            }

        last_error = None

        for attempt in range(self.max_retries):
            try:
                response = llm_client.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature + (attempt * 0.05),
                    max_tokens=500
                )

                parse_result = parser.parse(response.text)

                if parse_result.success:
                    self.cache.set(
                        prompt, model, prompt_version, parse_result.data
                    )
                    self._reset_circuit()
                    return {
                        'success': True, 'data': parse_result.data,
                        'from_cache': False,
                        'attempts': attempt + 1, 'error': None
                    }
                else:
                    last_error = f"Parse failed: {parse_result.error}"
                    self.retry_stats['regenerations'] += 1

                    if attempt < self.max_retries - 1:
                        wait = self.base_delay * (2 ** attempt)
                        logger.warning(
                            f"Parse failed, retrying in {wait}s..."
                        )
                        time.sleep(wait)
                    else:
                        self._record_failure()
                        return {
                            'success': False,
                            'data': {
                                'summary': 'LLM generation failed',
                                'recommendation': 'Manual review required',
                                'risk_level': 'medium',
                                'key_differences': ['Parse failed'],
                                'confidence': 'low'
                            },
                            'from_cache': False,
                            'attempts': attempt + 1,
                            'error': last_error
                        }

            except Exception as e:
                last_error = str(e)
                self._record_failure()

                if attempt < self.max_retries - 1:
                    wait = self.base_delay * (2 ** attempt)
                    logger.warning(f"Error: {e}, retrying in {wait}s...")
                    time.sleep(wait)
                    self.retry_stats['retries_needed'] += 1
                else:
                    self.retry_stats['failures'] += 1
                    return {
                        'success': False, 'data': None,
                        'from_cache': False,
                        'attempts': attempt + 1, 'error': last_error
                    }

        return {
            'success': False, 'data': None,
            'from_cache': False,
            'attempts': self.max_retries,
            'error': 'Max retries exceeded'
        }

    def _is_circuit_open(self) -> bool:
        if not self.circuit_breaker['is_open']:
            return False
        if self.circuit_breaker['last_failure']:
            elapsed = (
                datetime.now() - self.circuit_breaker['last_failure']
            ).total_seconds()
            if elapsed > self.circuit_breaker['timeout_seconds']:
                self.circuit_breaker['is_open'] = False
                self.circuit_breaker['failures'] = 0
                return False
        return True

    def _record_failure(self):
        self.circuit_breaker['failures'] += 1
        self.circuit_breaker['last_failure'] = datetime.now()
        if (self.circuit_breaker['failures'] >=
                self.circuit_breaker['threshold']):
            self.circuit_breaker['is_open'] = True
            logger.error("Circuit breaker opened")

    def _reset_circuit(self):
        self.circuit_breaker['failures'] = 0
        self.circuit_breaker['last_failure'] = None
        self.circuit_breaker['is_open'] = False

    def get_stats(self) -> Dict:
        return {
            **self.retry_stats,
            'circuit_breaker': {
                'is_open': self.circuit_breaker['is_open'],
                'recent_failures': self.circuit_breaker['failures']
            }
        }


# Test
# Test
if __name__ == "__main__":
    print("=" * 60)
    print("Cache Manager, Checkpoint & Retry Test")
    print("=" * 60)

    print("\n1. Testing LLM cache...")
    cache = LLMCacheManager(
        db_path="outputs/test_cache.db",
        default_ttl_days=7
    )

    test_response = {
        'summary': 'Policy covers encryption but lacks key management',
        'risk_level': 'medium',
        'confidence': 'high'
    }

    cache.set("test prompt", "llama3-8b", "v2", test_response)
    result = cache.get("test prompt", "llama3-8b", "v2")
    miss = cache.get("test prompt", "llama3-8b", "v1")

    print(f"   Cache set/get: {'PASS' if result else 'FAIL'}")
    print(f"   Cache miss (wrong version): {'PASS' if miss is None else 'FAIL'}")

    stats = cache.get_stats()
    print(f"   Stats: {stats}")

    print("\n2. Testing CheckpointManager...")
    cp = CheckpointManager("outputs/checkpoints/test_groq.json")
    cp.save(['reg_001', 'reg_002', 'reg_003'],
            [{'id': 'a'}, {'id': 'b'}, {'id': 'c'}])
    ids, results = cp.load()
    print(f"   Checkpoint save/load: {'PASS' if len(ids) == 3 else 'FAIL'}")
    print(f"   IDs recovered: {ids}")
    cp.clear()
    print(f"   Checkpoint cleared: {'PASS' if not cp.exists() else 'FAIL'}")

    print("\n3. Testing retry manager with mock...")

    # Mock LLM that fails once then succeeds
    # Uses LLMResponse structure matching groq_client.py
    class MockGroqClient:
        calls = 0

        def generate(
            self,
            prompt,
            system_prompt=None,
            temperature=0.1,
            max_tokens=500
        ):
            self.calls += 1
            if self.calls == 1:
                raise Exception("Temporary connection error")

            # Return mock response matching GroqLLMClient structure
            # No external import needed - define inline
            class MockResponse:
                def __init__(self):
                    self.text = (
                        '{"summary": "Test gap found",'
                        '"recommendation": "Update policy",'
                        '"risk_level": "low",'
                        '"key_differences": ["Missing item"],'
                        '"confidence": "high"}'
                    )
                    self.model = "llama3-8b-8192"
                    self.prompt_tokens = 50
                    self.completion_tokens = 30
                    self.total_tokens = 80
                    self.latency_ms = 200.0
                    self.timestamp = datetime.now().isoformat()

            return MockResponse()

    # Mock parser that parses JSON directly
    class MockParser:
        def parse(self, text):
            from explanation.response_parser import ParseResult
            try:
                data = json.loads(text)
                return ParseResult(
                    success=True,
                    data=data,
                    method='direct',
                    error=None,
                    raw_text=text,
                    parse_time_ms=1.0,
                    attempts=1
                )
            except Exception as e:
                return ParseResult(
                    success=False,
                    data=None,
                    method='failed',
                    error=str(e),
                    raw_text=text,
                    parse_time_ms=1.0,
                    attempts=1
                )

    retry = IntelligentRetryManager(cache)
    mock_llm = MockGroqClient()
    mock_parser = MockParser()

    # First call - should fail once then succeed
    result = retry.execute(
        "new test prompt",
        "llama3-8b",
        "v2",
        mock_llm,
        mock_parser
    )
    print(
        f"   Success after retry: "
        f"{'PASS' if result['success'] else 'FAIL'}"
    )
    print(f"   Attempts needed: {result['attempts']}")

    # Second call - should hit cache
    result2 = retry.execute(
        "new test prompt",
        "llama3-8b",
        "v2",
        mock_llm,
        mock_parser
    )
    print(
        f"   Second call cached: "
        f"{'PASS' if result2['from_cache'] else 'FAIL'}"
    )

    retry_stats = retry.get_stats()
    print(f"   Cache hits: {retry_stats['cache_hits']}")

    # Cleanup - close connections before deleting
    import os
    import gc

    # Clear cache object to release SQLite connection
    del cache
    del retry
    gc.collect()

    # Now safe to delete
    import time
    time.sleep(0.5)  # Small wait for Windows file release

    for db_file in ["outputs/test_cache.db"]:
        if os.path.exists(db_file):
            try:
                os.remove(db_file)
                print(f"\n   Cleanup: PASS")
            except PermissionError:
                print(f"\n   Cleanup: file will be removed on next run")

    print("\n" + "=" * 60)
    print("=" * 60)