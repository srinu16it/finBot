"""
Tests for the cache module.

These tests demonstrate how to test enhancement modules
while maintaining >80% coverage.
"""

import pytest
import os
import tempfile
from datetime import datetime, timedelta

from enhancements.data_access.cache import CacheManager


class TestCacheManager:
    """Test suite for CacheManager."""
    
    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            yield tmp.name
        # Cleanup
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)
    
    @pytest.fixture
    def cache(self, temp_db):
        """Create a CacheManager instance with temporary database."""
        return CacheManager(db_path=temp_db)
    
    def test_cache_initialization(self, cache):
        """Test that cache initializes correctly."""
        assert cache is not None
        assert cache.default_ttl == 3600
    
    def test_cache_set_and_get(self, cache):
        """Test basic cache set and get operations."""
        # Set data
        test_data = {"price": 100.50, "volume": 1000000}
        cache.set("test_provider", "AAPL", test_data)
        
        # Get data
        retrieved = cache.get("test_provider", "AAPL")
        assert retrieved == test_data
    
    def test_cache_expiration(self, cache):
        """Test that expired cache entries return None."""
        # Set data with very short TTL
        test_data = {"price": 100.50}
        cache.set("test_provider", "AAPL", test_data, ttl=0)
        
        # Should return None as it's already expired
        retrieved = cache.get("test_provider", "AAPL")
        assert retrieved is None
    
    def test_cache_with_params(self, cache):
        """Test cache with different parameters."""
        # Set data with params
        params1 = {"period": "1mo", "interval": "1d"}
        params2 = {"period": "1y", "interval": "1d"}
        
        data1 = {"count": 30}
        data2 = {"count": 365}
        
        cache.set("provider", "AAPL", data1, params=params1)
        cache.set("provider", "AAPL", data2, params=params2)
        
        # Should retrieve different data based on params
        assert cache.get("provider", "AAPL", params=params1) == data1
        assert cache.get("provider", "AAPL", params=params2) == data2
    
    def test_cache_stats(self, cache):
        """Test cache statistics."""
        # Add some entries
        cache.set("provider1", "AAPL", {"test": 1})
        cache.set("provider1", "GOOGL", {"test": 2})
        cache.set("provider2", "AAPL", {"test": 3})
        
        stats = cache.get_stats()
        assert stats["total_entries"] == 3
        assert stats["valid_entries"] == 3
        assert stats["providers"] == 2
        assert stats["symbols"] == 2
    
    def test_clear_expired(self, cache):
        """Test clearing expired entries."""
        # Add one valid and one expired entry
        cache.set("provider", "AAPL", {"valid": True}, ttl=3600)
        cache.set("provider", "GOOGL", {"valid": False}, ttl=0)
        
        # Clear expired
        cache.clear_expired()
        
        # Only valid entry should remain
        assert cache.get("provider", "AAPL") is not None
        assert cache.get("provider", "GOOGL") is None
    
    def test_clear_all(self, cache):
        """Test clearing all entries."""
        # Add entries
        cache.set("provider", "AAPL", {"test": 1})
        cache.set("provider", "GOOGL", {"test": 2})
        
        # Clear all
        cache.clear_all()
        
        # No entries should remain
        stats = cache.get_stats()
        assert stats["total_entries"] == 0
    
    def test_context_manager(self, temp_db):
        """Test using CacheManager as context manager."""
        with CacheManager(db_path=temp_db) as cache:
            cache.set("provider", "AAPL", {"test": 1})
            assert cache.get("provider", "AAPL") is not None
        
        # Connection should be closed after context exit
        # Create new instance to verify data persists
        cache2 = CacheManager(db_path=temp_db)
        assert cache2.get("provider", "AAPL") is not None
        cache2.close() 