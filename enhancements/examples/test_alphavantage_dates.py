#!/usr/bin/env python3
"""
Test AlphaVantage date issues for UNH.
"""

import os
import sys
from pathlib import Path
import pandas as pd

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

# Handle both API key naming conventions
if "ALPHA_VANTAGE_API_KEY" in os.environ and "ALPHAVANTAGE_API_KEY" not in os.environ:
    os.environ["ALPHAVANTAGE_API_KEY"] = os.environ["ALPHA_VANTAGE_API_KEY"]

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from enhancements.data_providers.alpha_provider import AlphaVantageProvider
from enhancements.data_access.cache import CacheManager


def test_alphavantage_dates():
    """Test AlphaVantage date issues."""
    print("Testing AlphaVantage Date Issues for UNH")
    print("=" * 60)
    
    # Initialize provider
    cache_manager = CacheManager()
    try:
        provider = AlphaVantageProvider(cache_manager)
        print("✅ AlphaVantage provider initialized")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return
    
    # Test daily data with full output
    print("\n📊 Testing Daily Data (Full):")
    df_full = provider.get_daily("UNH", outputsize="full", use_cache=False)
    
    if df_full is not None:
        print(f"  Total rows: {len(df_full)}")
        print(f"  Date range: {df_full.index.min()} to {df_full.index.max()}")
        print(f"  Latest close: ${df_full['Close'].iloc[-1]:.2f}")
        
        # Show first and last 5 dates
        print("\n  First 5 dates:")
        for date in df_full.index[:5]:
            print(f"    {date}: ${df_full.loc[date, 'Close']:.2f}")
        
        print("\n  Last 5 dates:")
        for date in df_full.index[-5:]:
            print(f"    {date}: ${df_full.loc[date, 'Close']:.2f}")
        
        # Check for year 2000 data
        year_2000 = df_full[df_full.index.year == 2000]
        if not year_2000.empty:
            print(f"\n⚠️ Found {len(year_2000)} rows from year 2000")
            print(f"  2000 price range: ${year_2000['Close'].min():.2f} - ${year_2000['Close'].max():.2f}")
    else:
        print("  Failed to get full data")
    
    # Test daily data with compact output
    print("\n📊 Testing Daily Data (Compact):")
    df_compact = provider.get_daily("UNH", outputsize="compact", use_cache=False)
    
    if df_compact is not None:
        print(f"  Total rows: {len(df_compact)}")
        print(f"  Date range: {df_compact.index.min()} to {df_compact.index.max()}")
        print(f"  Latest close: ${df_compact['Close'].iloc[-1]:.2f}")
    else:
        print("  Failed to get compact data")


if __name__ == "__main__":
    test_alphavantage_dates() 