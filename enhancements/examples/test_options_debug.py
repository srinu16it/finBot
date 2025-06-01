#!/usr/bin/env python3
"""
Debug test for Options Chain API
"""

import os
import sys
import logging
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Enable debug logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Load environment variables
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path)
    print(f"✅ Loaded .env from {env_path}")

from enhancements.data_providers.alpha_provider import AlphaVantageProvider
from enhancements.data_access.cache import CacheManager
import requests

def test_options_api_directly():
    """Test the API directly to see raw response."""
    
    print("\n🔍 Testing Options API Directly")
    print("=" * 60)
    
    # Get the premium key
    premium_key = os.getenv("ALPHA_VANTAGE_PREMIUM_API_KEY")
    if not premium_key:
        print("❌ No ALPHA_VANTAGE_PREMIUM_API_KEY found")
        return
        
    print(f"Using key: {premium_key[:8]}...")
    
    # Test direct API call
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "REALTIME_OPTIONS",
        "symbol": "AAPL",
        "require_greeks": "true",
        "apikey": premium_key
    }
    
    print("\n📡 Making direct API call...")
    print(f"URL: {url}")
    print(f"Params: {params}")
    
    try:
        response = requests.get(url, params=params)
        print(f"\nResponse Status: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        data = response.json()
        print(f"\nResponse Keys: {list(data.keys())}")
        
        # Print full response (limited)
        import json
        print("\nFull Response (first 500 chars):")
        print(json.dumps(data, indent=2)[:500])
        
        # Check specific fields
        if "Error Message" in data:
            print(f"\n❌ API Error: {data['Error Message']}")
        elif "Note" in data:
            print(f"\n⚠️ API Note: {data['Note']}")
        elif "Information" in data:
            print(f"\nℹ️ API Info: {data['Information']}")
        elif "options" in data:
            print(f"\n✅ Options data found! Count: {len(data['options'])}")
        else:
            print(f"\n🤔 Unexpected response structure")
            
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Also test through our provider
    print("\n\n🧪 Testing through AlphaVantageProvider...")
    print("-" * 60)
    
    try:
        cache_manager = CacheManager()
        provider = AlphaVantageProvider(cache_manager)
        
        options_data = provider.get_options_chain("AAPL", require_greeks=True)
        
        if options_data:
            print("✅ Options data retrieved through provider!")
            print(f"Keys: {list(options_data.keys())}")
        else:
            print("❌ No options data through provider")
            
    except Exception as e:
        print(f"❌ Provider error: {str(e)}")

if __name__ == "__main__":
    test_options_api_directly() 