#!/usr/bin/env python3
"""
Test Real-time Options Chain with IV Implementation
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from enhancements.data_providers.alpha_provider import AlphaVantageProvider
from enhancements.data_access.cache import CacheManager

def test_options_chain():
    """Test if options chain with IV is implemented and working."""
    
    print("Testing Real-time Options Chain Implementation")
    print("=" * 60)
    
    # Check if API key is set
    if "ALPHAVANTAGE_API_KEY" not in os.environ and "ALPHA_VANTAGE_API_KEY" not in os.environ:
        print("❌ ALPHAVANTAGE_API_KEY not set")
        print("   Set it in your .env file or export it:")
        print("   export ALPHAVANTAGE_API_KEY=your_key_here")
        return
    
    try:
        # Initialize provider
        cache_manager = CacheManager()
        provider = AlphaVantageProvider(cache_manager)
        
        # Check if method exists
        if hasattr(provider, 'get_options_chain'):
            print("✅ get_options_chain method exists")
            
            # Test with a symbol
            symbol = "AAPL"
            print(f"\nFetching options chain for {symbol}...")
            
            # Call the method
            options_data = provider.get_options_chain(symbol, require_greeks=True)
            
            if options_data:
                print("\n✅ Options chain data received:")
                print(f"  Underlying Price: ${options_data.get('underlying_price', 'N/A')}")
                print(f"  ATM IV: {options_data.get('atm_iv', 'N/A')}%")
                print(f"  Call IV: {options_data.get('call_iv', 'N/A')}%")
                print(f"  Put IV: {options_data.get('put_iv', 'N/A')}%")
                print(f"  IV Skew: {options_data.get('iv_skew', 'N/A')}%")
                print(f"  Options Count: {options_data.get('options_count', 'N/A')}")
                
                # Check if we got actual IV data
                if options_data.get('atm_iv'):
                    print("\n🎉 SUCCESS: Real-time IV data is working!")
                else:
                    print("\n⚠️  No IV data returned - check API response")
            else:
                print("\n❌ No options data returned")
                print("   This could be due to:")
                print("   - Invalid API key")
                print("   - Rate limit reached")
                print("   - Options data not available for symbol")
                print("   - Need premium AlphaVantage subscription")
        else:
            print("❌ get_options_chain method NOT found")
            print("   The implementation may not be loaded correctly")
            
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_options_chain() 