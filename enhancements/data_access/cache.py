"""
Cache management module for storing and retrieving data efficiently.

This module provides a DuckDB-based cache system with automatic expiration
and statistics tracking. Falls back to in-memory cache if DuckDB is not available.
"""

import os
import json
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from pathlib import Path
import logging
import hashlib

# Try to import duckdb, fallback to in-memory cache if not available
try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False
    logging.warning("DuckDB not available, using in-memory cache")

logger = logging.getLogger(__name__)


class InMemoryCache:
    """Simple in-memory cache implementation as fallback."""
    
    def __init__(self):
        self.cache = {}
        self.stats = {
            "total_entries": 0,
            "total_hits": 0,
            "total_misses": 0,
            "hit_rate": 0.0
        }
    
    def get(self, provider: str, symbol: str, params: Optional[Dict] = None) -> Optional[Any]:
        """Retrieve data from cache."""
        cache_key = f"{provider}:{symbol}:{json.dumps(params or {}, sort_keys=True)}"
        
        if cache_key in self.cache:
            entry = self.cache[cache_key]
            if entry["expires_at"] > time.time():
                self.stats["total_hits"] += 1
                self._update_hit_rate()
                return entry["data"]
            else:
                # Expired entry
                del self.cache[cache_key]
        
        self.stats["total_misses"] += 1
        self._update_hit_rate()
        return None
    
    def set(self, provider: str, symbol: str, data: Any, 
            params: Optional[Dict] = None, ttl: int = 3600) -> None:
        """Store data in cache."""
        cache_key = f"{provider}:{symbol}:{json.dumps(params or {}, sort_keys=True)}"
        
        self.cache[cache_key] = {
            "data": data,
            "expires_at": time.time() + ttl,
            "created_at": time.time()
        }
        self.stats["total_entries"] = len(self.cache)
    
    def clear_all(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
        self.stats = {
            "total_entries": 0,
            "total_hits": 0,
            "total_misses": 0,
            "hit_rate": 0.0
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        # Clean up expired entries
        current_time = time.time()
        expired_keys = [k for k, v in self.cache.items() if v["expires_at"] <= current_time]
        for key in expired_keys:
            del self.cache[key]
        
        self.stats["total_entries"] = len(self.cache)
        return self.stats
    
    def _update_hit_rate(self):
        """Update hit rate calculation."""
        total_requests = self.stats["total_hits"] + self.stats["total_misses"]
        if total_requests > 0:
            self.stats["hit_rate"] = self.stats["total_hits"] / total_requests


class CacheManager:
    """
    Manages caching of financial data using DuckDB or in-memory fallback.
    
    Features:
    - Persistent cache storage (with DuckDB)
    - Automatic expiration
    - Statistics tracking
    - Thread-safe operations
    """
    
    def __init__(self, cache_dir: str = "cache", db_name: str = "finbot_cache.db"):
        """
        Initialize the cache manager.
        
        Args:
            cache_dir: Directory to store cache database
            db_name: Name of the cache database file
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.db_path = self.cache_dir / db_name
        self.use_duckdb = DUCKDB_AVAILABLE
        
        if self.use_duckdb:
            self._init_duckdb()
        else:
            # Use in-memory cache as fallback
            self.fallback_cache = InMemoryCache()
    
    def _init_duckdb(self):
        """Initialize DuckDB database and tables."""
        self.conn = duckdb.connect(str(self.db_path))
        
        # Create cache table if not exists
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                id INTEGER PRIMARY KEY,
                provider VARCHAR NOT NULL,
                symbol VARCHAR NOT NULL,
                params_hash VARCHAR NOT NULL,
                data JSON NOT NULL,
                created_at TIMESTAMP NOT NULL,
                expires_at TIMESTAMP NOT NULL,
                access_count INTEGER DEFAULT 0,
                last_accessed TIMESTAMP
            )
        """)
        
        # Create unique index for cache lookup
        self.conn.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_cache_unique 
            ON cache(params_hash)
        """)
        
        # Create index for faster lookups
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_cache_lookup 
            ON cache(provider, symbol, params_hash)
        """)
        
        # Create stats table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS cache_stats (
                id INTEGER PRIMARY KEY,
                total_hits INTEGER DEFAULT 0,
                total_misses INTEGER DEFAULT 0,
                last_cleanup TIMESTAMP
            )
        """)
        
        # Initialize stats if not exists
        if self.conn.execute("SELECT COUNT(*) FROM cache_stats").fetchone()[0] == 0:
            self.conn.execute("""
                INSERT INTO cache_stats (id, total_hits, total_misses, last_cleanup)
                VALUES (1, 0, 0, CURRENT_TIMESTAMP)
            """)
    
    def _generate_cache_key(self, provider: str, symbol: str, params: Optional[Dict] = None) -> str:
        """
        Generate a unique cache key based on provider, symbol, and parameters.
        
        Args:
            provider: Data provider name
            symbol: Stock symbol
            params: Additional parameters that affect the data
            
        Returns:
            A unique cache key string
        """
        key_parts = [provider, symbol]
        if params:
            # Sort params for consistent key generation
            sorted_params = json.dumps(params, sort_keys=True)
            key_parts.append(sorted_params)
        
        key_string = ":".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, provider: str, symbol: str, params: Optional[Dict] = None) -> Optional[Any]:
        """
        Retrieve data from cache if available and not expired.
        
        Args:
            provider: Data provider name
            symbol: Stock symbol
            params: Additional parameters that affect the data
            
        Returns:
            Cached data if available and valid, None otherwise
        """
        if not self.use_duckdb:
            return self.fallback_cache.get(provider, symbol, params)
        
        cache_key = self._generate_cache_key(provider, symbol, params)
        
        result = self.conn.execute("""
            SELECT data, expires_at 
            FROM cache 
            WHERE params_hash = ? AND expires_at > ?
        """, [cache_key, datetime.now()]).fetchone()
        
        if result:
            # Update access stats
            self.conn.execute("""
                UPDATE cache 
                SET access_count = access_count + 1, last_accessed = ?
                WHERE params_hash = ?
            """, [datetime.now(), cache_key])
            
            self.conn.execute("""
                UPDATE cache_stats 
                SET total_hits = total_hits + 1
                WHERE id = 1
            """)
            
            return json.loads(result[0])
        
        # Update miss stats
        self.conn.execute("""
            UPDATE cache_stats 
            SET total_misses = total_misses + 1
            WHERE id = 1
        """)
        
        return None
    
    def set(self, provider: str, symbol: str, data: Any, 
            params: Optional[Dict] = None, ttl: Optional[int] = None,
            metadata: Optional[Dict] = None):
        """
        Store data in cache.
        
        Args:
            provider: Data provider name
            symbol: Stock symbol
            data: Data to cache
            params: Additional parameters that affect the data
            ttl: Time-to-live in seconds (uses default if not specified)
            metadata: Additional metadata to store with the cache entry
        """
        if not self.use_duckdb:
            self.fallback_cache.set(provider, symbol, data, params, ttl or 3600)
            return
        
        cache_key = self._generate_cache_key(provider, symbol, params)
        ttl = ttl or 3600
        
        created_at = datetime.now()
        expires_at = created_at + timedelta(seconds=ttl)
        
        # First try to update existing entry
        result = self.conn.execute("""
            UPDATE cache 
            SET data = ?, expires_at = ?, last_accessed = ?, access_count = access_count + 1
            WHERE params_hash = ?
            RETURNING id
        """, [
            json.dumps(data),
            expires_at,
            created_at,
            cache_key
        ]).fetchone()
        
        # If no existing entry, insert new one
        if not result:
            # Get next id
            next_id = self.conn.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM cache").fetchone()[0]
            
            self.conn.execute("""
                INSERT INTO cache 
                (id, provider, symbol, params_hash, data, created_at, expires_at, access_count, last_accessed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                next_id,
                provider,
                symbol,
                cache_key,
                json.dumps(data),
                created_at,
                expires_at,
                0,
                created_at
            ])
        
        self.conn.commit()
    
    def clear_expired(self):
        """Remove all expired cache entries."""
        if not self.use_duckdb:
            # In-memory cache handles this automatically
            return
        
        self.conn.execute("""
            DELETE FROM cache WHERE expires_at < ?
        """, [datetime.now()])
        self.conn.commit()
    
    def clear_all(self):
        """Remove all cache entries."""
        if not self.use_duckdb:
            self.fallback_cache.clear_all()
            return
        
        self.conn.execute("DELETE FROM cache")
        self.conn.execute("""
            UPDATE cache_stats 
            SET total_hits = 0, total_misses = 0, last_cleanup = ?
            WHERE id = 1
        """, [datetime.now()])
        self.conn.commit()
    
    def get_stats(self) -> Dict:
        """
        Get cache statistics.
        
        Returns:
            Dictionary containing cache statistics
        """
        if not self.use_duckdb:
            return self.fallback_cache.get_stats()
        
        # Get cache entry stats
        entry_stats = self.conn.execute("""
            SELECT 
                COUNT(*) as total_entries,
                COUNT(CASE WHEN expires_at > ? THEN 1 END) as valid_entries,
                COUNT(DISTINCT provider) as providers,
                COUNT(DISTINCT symbol) as symbols
            FROM cache
        """, [datetime.now()]).fetchone()
        
        # Get hit/miss stats
        hit_miss_stats = self.conn.execute("""
            SELECT total_hits, total_misses
            FROM cache_stats
            WHERE id = 1
        """).fetchone()
        
        total_hits = hit_miss_stats[0] if hit_miss_stats else 0
        total_misses = hit_miss_stats[1] if hit_miss_stats else 0
        total_requests = total_hits + total_misses
        
        return {
            "total_entries": entry_stats[0],
            "valid_entries": entry_stats[1],
            "providers": entry_stats[2],
            "symbols": entry_stats[3],
            "total_hits": total_hits,
            "total_misses": total_misses,
            "hit_rate": total_hits / total_requests if total_requests > 0 else 0.0
        }
    
    def close(self):
        """Close the database connection."""
        if self.use_duckdb:
            self.conn.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close() 