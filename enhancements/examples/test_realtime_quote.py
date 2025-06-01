#!/usr/bin/env python3
"""
Test script for AlphaVantage real-time quotes.

Usage:
    ./venv_test/bin/python enhancements/examples/test_realtime_quote.py
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from enhancements.data_providers.alpha_provider import AlphaVantageProvider
from enhancements.data_access.cache import CacheManager


def test_realtime_quote(symbol: str = "AAPL"):
    """Test real-time quote functionality."""
    
    # Check for API key
    if "ALPHAVANTAGE_API_KEY" not in os.environ:
        print("❌ Please set ALPHAVANTAGE_API_KEY environment variable")
        return
    
    print(f"\n🔍 Testing Real-Time Quote for {symbol}")
    print("=" * 50)
    
    # Initialize provider
    cache_manager = CacheManager()
    provider = AlphaVantageProvider(cache_manager)
    
    # Fetch real-time quote
    print(f"\nFetching real-time quote...")
    quote = provider.get_quote(symbol)
    
    if quote:
        print(f"\n✅ Real-Time Quote Retrieved Successfully!")
        print(f"\nSymbol: {quote.get('symbol', symbol)}")
        print(f"Price: ${quote.get('price', 0):.2f}")
        print(f"Open: ${quote.get('open', 0):.2f}")
        print(f"High: ${quote.get('high', 0):.2f}")
        print(f"Low: ${quote.get('low', 0):.2f}")
        print(f"Volume: {quote.get('volume', 0):,}")
        print(f"Previous Close: ${quote.get('previous_close', 0):.2f}")
        print(f"Change: ${quote.get('change', 0):.2f}")
        print(f"Change %: {quote.get('change_percent', 0)}%")
        print(f"Latest Trading Day: {quote.get('latest_trading_day', 'N/A')}")
        print(f"Timestamp: {quote.get('timestamp', 'N/A')}")
        
        # Compare with daily data
        print(f"\n📊 Comparing with Daily Data...")
        daily_df = provider.get_daily(symbol, outputsize="compact")
        
        if daily_df is not None and not daily_df.empty:
            last_close = daily_df['Close'].iloc[0]
            print(f"Last Daily Close: ${last_close:.2f}")
            diff = abs(quote.get('price', 0) - last_close)
            print(f"Difference: ${diff:.2f}")
            
            if diff > 0.01:
                print("✅ Real-time price differs from last close - data is fresh!")
            else:
                print("ℹ️  Real-time price matches last close")
    else:
        print("❌ Failed to retrieve quote")
        
    # Test caching
    print(f"\n💾 Testing Cache...")
    cached_quote = provider.get_quote(symbol)
    if cached_quote:
        print("✅ Cache working - retrieved from cache")


def main():
    """Main function."""
    # Test multiple symbols
    symbols = ["AAPL", "MSFT", "GOOGL"]
    
    for symbol in symbols:
        test_realtime_quote(symbol)
        print("\n" + "-" * 50)


if __name__ == "__main__":
    main() 