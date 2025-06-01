#!/usr/bin/env python3
"""
Test AlphaVantage 4-hour data implementation
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
from dotenv import load_dotenv
env_path = project_root / '.env'
if env_path.exists():
    load_dotenv(env_path)

# Handle both API key naming conventions
if "ALPHA_VANTAGE_API_KEY" in os.environ and "ALPHAVANTAGE_API_KEY" not in os.environ:
    os.environ["ALPHAVANTAGE_API_KEY"] = os.environ["ALPHA_VANTAGE_API_KEY"]

from enhancements.data_providers.alpha_provider import AlphaVantageProvider
from enhancements.data_providers.yahoo_provider import YahooProvider
from enhancements.data_access.cache import CacheManager


def test_alpha_4h():
    """Test AlphaVantage 4-hour data functionality."""
    symbol = "AAPL"
    cache_manager = CacheManager()
    
    print(f"Testing 4-hour data for {symbol}")
    print("=" * 60)
    
    # Test AlphaVantage
    if "ALPHAVANTAGE_API_KEY" in os.environ:
        print("\n1. Testing AlphaVantage 4-hour data:")
        print("-" * 40)
        
        alpha_provider = AlphaVantageProvider(cache_manager)
        
        # First test if we can get 60-minute data
        print("   Fetching 60-minute data...")
        df_60min = alpha_provider.get_intraday(symbol, interval="60min", outputsize="full")
        
        if df_60min is not None:
            print(f"   ✅ 60-minute data: {len(df_60min)} rows")
            print(f"   Date range: {df_60min.index[0]} to {df_60min.index[-1]}")
            
            # Now test 4-hour data
            print("\n   Fetching 4-hour data...")
            df_4h = alpha_provider.get_4hour_data(symbol, outputsize="full")
            
            if df_4h is not None:
                print(f"   ✅ 4-hour data created: {len(df_4h)} rows")
                
                if len(df_4h) > 0:
                    if 'Date' in df_4h.columns:
                        print(f"   Date range: {df_4h['Date'].iloc[0]} to {df_4h['Date'].iloc[-1]}")
                    elif isinstance(df_4h.index, pd.DatetimeIndex):
                        print(f"   Date range: {df_4h.index[0]} to {df_4h.index[-1]}")
                    
                    # Show sample data
                    print("\n   Last 5 4-hour bars:")
                    print(df_4h.tail(5))
                else:
                    print("   ⚠️  4-hour DataFrame is empty - check hour filtering logic")
                    print("   Debugging info:")
                    print(f"   60-min data hours present: {sorted(df_60min.index.hour.unique())}")
                    print(f"   60-min data date range: {df_60min.index.min()} to {df_60min.index.max()}")
            else:
                print("   ❌ Failed to create 4-hour data")
        else:
            print("   ❌ Failed to fetch 60-minute data (needed for 4-hour resampling)")
    else:
        print("\n❌ ALPHAVANTAGE_API_KEY not set - skipping AlphaVantage test")
    
    # Compare with Yahoo for reference
    print("\n2. Yahoo 4-hour data (for comparison):")
    print("-" * 40)
    
    yahoo_provider = YahooProvider(cache_manager)
    df_yahoo_4h = yahoo_provider.get_4hour_data(symbol, period="2mo")
    
    if df_yahoo_4h is not None:
        print(f"   ✅ Yahoo 4-hour data: {len(df_yahoo_4h)} rows")
        if 'Date' in df_yahoo_4h.columns:
            print(f"   Date range: {df_yahoo_4h['Date'].iloc[0]} to {df_yahoo_4h['Date'].iloc[-1]}")
        
        # Show sample data
        print("\n   Last 5 4-hour bars:")
        print(df_yahoo_4h.tail(5))
    else:
        print("   ❌ Failed to fetch Yahoo 4-hour data")
    
    print("\n" + "=" * 60)
    print("Test completed!")


if __name__ == "__main__":
    test_alpha_4h() 