#!/usr/bin/env python3
"""
Simple example script to test AlphaVantage data provider.

Usage:
    ./venv_test/bin/python enhancements/examples/test_alphavantage.py
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from enhancements.data_providers.alpha_provider import AlphaVantageProvider
from enhancements.data_access.cache import CacheManager


def main():
    """Test AlphaVantage data provider."""
    
    # Check for API key
    if "ALPHAVANTAGE_API_KEY" not in os.environ:
        print("❌ Please set the ALPHAVANTAGE_API_KEY environment variable!")
        print("Get your free API key at: https://www.alphavantage.co/support/#api-key")
        print("\nExample:")
        print("export ALPHAVANTAGE_API_KEY='your_api_key_here'")
        return
    
    # Initialize provider with cache
    cache_manager = CacheManager(cache_dir="enhancements/cache")
    provider = AlphaVantageProvider(cache_manager)
    
    # Test different data types
    symbol = "AAPL"
    
    print(f"\n📊 Testing AlphaVantage Data Provider for {symbol}")
    print("=" * 50)
    
    # 1. Get daily data
    print("\n1. Fetching daily data...")
    daily_df = provider.get_daily(symbol, outputsize="compact")
    if daily_df is not None:
        print(f"✅ Daily data: {len(daily_df)} rows")
        print(f"Latest close: ${daily_df['Close'].iloc[0]:.2f}")
        print(f"Date range: {daily_df.index[-1].date()} to {daily_df.index[0].date()}")
    else:
        print("❌ Failed to fetch daily data")
    
    # 2. Get real-time quote
    print("\n2. Fetching real-time quote...")
    quote = provider.get_quote(symbol)
    if quote:
        print(f"✅ Current price: ${quote['price']:.2f}")
        print(f"   Change: {quote['change']:.2f} ({quote['change_percent']})")
        print(f"   Volume: {quote['volume']:,}")
    else:
        print("❌ Failed to fetch quote")
    
    # 3. Get technical indicator (RSI)
    print("\n3. Fetching RSI indicator...")
    rsi_df = provider.get_technical_indicator(
        symbol, 
        indicator="RSI",
        interval="daily",
        time_period=14,
        series_type="close"
    )
    if rsi_df is not None:
        print(f"✅ RSI data: {len(rsi_df)} rows")
        print(f"Latest RSI: {rsi_df.iloc[0].values[0]:.2f}")
    else:
        print("❌ Failed to fetch RSI data")
    
    # 4. Check cache stats
    print("\n4. Cache Statistics:")
    stats = cache_manager.get_stats()
    print(f"   Total entries: {stats.get('total_entries', 0)}")
    print(f"   Valid entries: {stats.get('valid_entries', 0)}")
    print(f"   Hit rate: {stats.get('hit_rate', 0):.2%}")
    
    print("\n✅ Test completed!")
    print("\nNote: Due to rate limits (5 calls/minute for free tier),")
    print("subsequent runs will use cached data when available.")


if __name__ == "__main__":
    main() 