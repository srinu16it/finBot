#!/usr/bin/env python3
"""
Test Premium API Key for Options Chain
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

def test_premium_options():
    """Test if ALPHA_VANTAGE_PREMIUM_API_KEY works for options data."""
    
    print("\nTesting Premium API Key for Options Chain")
    print("=" * 60)
    
    # Check all possible API keys
    regular_key = os.getenv("ALPHAVANTAGE_API_KEY") or os.getenv("ALPHA_VANTAGE_API_KEY")
    realtime_key = os.getenv("ALPHA_VANTAGE_REALTIME_API_KEY")
    premium_key = os.getenv("ALPHA_VANTAGE_PREMIUM_API_KEY")
    
    print("\n📋 API Keys Found:")
    print(f"  Regular: {'✅' if regular_key else '❌'} {regular_key[:8] + '...' if regular_key else 'Not set'}")
    print(f"  Realtime: {'✅' if realtime_key else '❌'} {realtime_key[:8] + '...' if realtime_key else 'Not set'}")
    print(f"  Premium: {'✅' if premium_key else '❌'} {premium_key[:8] + '...' if premium_key else 'Not set'}")
    
    if not premium_key:
        print("\n❌ ALPHA_VANTAGE_PREMIUM_API_KEY not found in .env")
        return
    
    print(f"\n🔑 Using Premium Key: {premium_key[:8]}...")
    
    # Test with different symbols
    symbols = ["AAPL", "SPY", "TSLA"]
    
    for symbol in symbols:
        print(f"\n🧪 Testing {symbol}...")
        print("-" * 40)
        
        try:
            cache_manager = CacheManager()
            provider = AlphaVantageProvider(cache_manager)
            
            # Fetch options chain
            options_data = provider.get_options_chain(symbol, require_greeks=True)
            
            if options_data:
                print(f"✅ SUCCESS! Options data received for {symbol}:")
                print(f"  Underlying Price: ${options_data.get('underlying_price', 'N/A')}")
                print(f"  ATM IV: {options_data.get('atm_iv', 'N/A')}%")
                print(f"  Call IV: {options_data.get('call_iv', 'N/A')}%")
                print(f"  Put IV: {options_data.get('put_iv', 'N/A')}%")
                print(f"  IV Skew: {options_data.get('iv_skew', 'N/A')}%")
                print(f"  Options Count: {options_data.get('options_count', 'N/A')}")
                
                # This means the premium key is working!
                print("\n🎉 Premium API Key is working!")
                break
            else:
                print(f"❌ No options data for {symbol}")
                
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Final summary
    print("\n" + "=" * 60)
    print("📝 Summary:")
    if any([regular_key, realtime_key, premium_key]):
        print("  API keys are configured.")
        print("  If no options data returned, you may need:")
        print("  - A premium AlphaVantage subscription")
        print("  - Different API endpoint access")
        print("  - To check rate limits")
    else:
        print("  No API keys found in .env file")

if __name__ == "__main__":
    test_premium_options() 