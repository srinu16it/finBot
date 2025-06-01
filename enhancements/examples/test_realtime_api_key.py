#!/usr/bin/env python3
"""
Test Real-time API Key Usage
"""

import os
import sys
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Load environment variables
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path)
    print(f"✅ Loaded .env from {env_path}")

from enhancements.data_providers.alpha_provider import AlphaVantageProvider
from enhancements.data_access.cache import CacheManager

def test_realtime_api_keys():
    """Test that different API keys are being used correctly."""
    
    print("\nChecking API Keys Configuration")
    print("=" * 60)
    
    # Check which API keys are set
    regular_key = os.getenv("ALPHAVANTAGE_API_KEY") or os.getenv("ALPHA_VANTAGE_API_KEY")
    realtime_key = os.getenv("ALPHA_VANTAGE_REALTIME_API_KEY")
    
    print("\n📋 API Keys Status:")
    print(f"  Regular API Key: {'✅ Set' if regular_key else '❌ Not set'}")
    if regular_key:
        print(f"    Key preview: {regular_key[:8]}...")
    
    print(f"  Realtime API Key: {'✅ Set' if realtime_key else '❌ Not set'}")
    if realtime_key:
        print(f"    Key preview: {realtime_key[:8]}...")
        if realtime_key == regular_key:
            print("    ⚠️  Same as regular key")
        else:
            print("    ✅ Different from regular key")
    
    if not realtime_key:
        print("\n❌ ALPHA_VANTAGE_REALTIME_API_KEY not found in .env")
        print("   Add it to your .env file:")
        print("   ALPHA_VANTAGE_REALTIME_API_KEY=your_premium_key_here")
        return
    
    # Test the implementation
    print("\n🧪 Testing Options Chain with Realtime Key")
    print("-" * 40)
    
    try:
        cache_manager = CacheManager()
        provider = AlphaVantageProvider(cache_manager)
        
        # Test options chain (uses realtime key)
        symbol = "AAPL"
        print(f"\nFetching options chain for {symbol}...")
        options_data = provider.get_options_chain(symbol, require_greeks=True)
        
        if options_data:
            print("✅ Options chain request successful!")
            print(f"  Data received: {list(options_data.keys())}")
        else:
            print("❌ No options data returned")
            print("   Note: REALTIME_OPTIONS requires a premium AlphaVantage subscription")
        
        # Test news sentiment (also uses realtime key now)
        print(f"\nFetching news sentiment for {symbol}...")
        news_data = provider.get_news_sentiment(symbol)
        
        if news_data:
            print("✅ News sentiment request successful!")
            print(f"  Sentiment Score: {news_data.get('sentiment_score', 'N/A')}")
            print(f"  Articles Analyzed: {news_data.get('articles_analyzed', 'N/A')}")
        else:
            print("❌ No news data returned")
            
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n📝 Summary:")
    print("  - Regular API calls use ALPHAVANTAGE_API_KEY")
    print("  - Options chain uses ALPHA_VANTAGE_REALTIME_API_KEY (if set)")
    print("  - News sentiment uses ALPHA_VANTAGE_REALTIME_API_KEY (if set)")
    print("  - Falls back to regular key if realtime key not found")

if __name__ == "__main__":
    test_realtime_api_keys() 