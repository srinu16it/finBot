#!/usr/bin/env python3
"""
Test API with Greeks to get IV
"""

import requests
import json

def test_with_greeks():
    """Test if adding require_greeks gives us IV data."""
    
    print("Testing REALTIME_OPTIONS with Greeks")
    print("=" * 60)
    
    api_key = "MYKYNF7L9MR0AMX9"
    symbol = "NVDA"
    
    # Test without greeks first
    print("\n1️⃣ WITHOUT Greeks:")
    url = f"https://www.alphavantage.co/query?function=REALTIME_OPTIONS&symbol={symbol}&apikey={api_key}"
    response = requests.get(url)
    data = response.json()
    
    if 'data' in data and len(data['data']) > 0:
        print(f"Fields: {list(data['data'][0].keys())}")
    
    # Test WITH greeks
    print("\n2️⃣ WITH Greeks (require_greeks=true):")
    url = f"https://www.alphavantage.co/query?function=REALTIME_OPTIONS&symbol={symbol}&require_greeks=true&apikey={api_key}"
    response = requests.get(url)
    data = response.json()
    
    if 'data' in data and len(data['data']) > 0:
        first_option = data['data'][0]
        print(f"Fields: {list(first_option.keys())}")
        
        # Look for IV and Greeks
        print("\n🔍 Greeks and IV fields found:")
        for key in first_option.keys():
            if key not in ['contractID', 'symbol', 'expiration', 'strike', 'type', 'last', 'mark', 'bid', 'ask', 'volume', 'open_interest', 'date', 'bid_size', 'ask_size']:
                print(f"  {key}: {first_option[key]}")
        
        # Check if we have implied_volatility
        if 'implied_volatility' in first_option:
            print(f"\n✅ SUCCESS! Found IV: {first_option['implied_volatility']}")
            
            # Show a few more examples
            print("\nSample IV values:")
            for i in range(min(5, len(data['data']))):
                option = data['data'][i]
                print(f"  {option['type']} ${option['strike']}: IV = {option.get('implied_volatility', 'N/A')}")

if __name__ == "__main__":
    test_with_greeks() 