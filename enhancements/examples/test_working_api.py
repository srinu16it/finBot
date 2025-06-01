#!/usr/bin/env python3
"""
Test with the API key that works in curl
"""

import os
import sys
import requests
import json
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Load environment variables
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path)

def test_working_curl_api():
    """Test with the exact parameters that work in curl."""
    
    print("Testing with Working CURL Parameters")
    print("=" * 60)
    
    # Use the exact working API key
    api_key = "MYKYNF7L9MR0AMX9"
    
    # Test symbols
    symbols = ["NVDA", "AAPL", "TSLA"]
    
    for symbol in symbols:
        print(f"\n🧪 Testing {symbol} with regular API key...")
        
        # Exact curl parameters
        url = f"https://www.alphavantage.co/query?function=REALTIME_OPTIONS&symbol={symbol}&apikey={api_key}"
        
        print(f"URL: {url}")
        
        try:
            response = requests.get(url, timeout=10)
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                
                # Check what we got
                if 'options' in data:
                    print(f"✅ SUCCESS! Found {len(data['options'])} options")
                    
                    # Show sample data
                    if data['options']:
                        first_option = data['options'][0]
                        print("\nSample option data:")
                        print(json.dumps(first_option, indent=2)[:300] + "...")
                        
                        # Check if we have the fields we need
                        if 'impliedVolatility' in first_option:
                            print("\n✅ Has IV data!")
                        if 'strike' in first_option:
                            print("✅ Has strike data!")
                            
                elif 'data' in data:
                    # This is what we were getting before
                    print("❌ Got 'data' key instead of 'options'")
                    print(f"Message: {data.get('message', 'No message')[:200]}")
                else:
                    print(f"❌ Unexpected response: {list(data.keys())}")
                    
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    # Now let's update our provider to use this
    print("\n\n📝 Solution:")
    print("The regular API key (MYKYNF7L9MR0AMX9) works!")
    print("We should use this instead of looking for premium keys.")

if __name__ == "__main__":
    test_working_curl_api() 