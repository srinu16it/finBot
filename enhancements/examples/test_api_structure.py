#!/usr/bin/env python3
"""
Test to see actual API response structure
"""

import os
import sys
import requests
import json
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def test_api_structure():
    """See the actual structure of the API response."""
    
    print("Checking API Response Structure")
    print("=" * 60)
    
    # Use the working API key
    api_key = "MYKYNF7L9MR0AMX9"
    symbol = "NVDA"
    
    url = f"https://www.alphavantage.co/query?function=REALTIME_OPTIONS&symbol={symbol}&apikey={api_key}"
    
    print(f"Testing {symbol}...")
    print(f"URL: {url}\n")
    
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        
        # Show top-level keys
        print(f"Top-level keys: {list(data.keys())}")
        
        # Show the structure
        if 'data' in data:
            print(f"\n'data' key contains: {type(data['data'])}")
            
            if isinstance(data['data'], list) and len(data['data']) > 0:
                print(f"Number of items in data: {len(data['data'])}")
                print("\nFirst item structure:")
                first_item = data['data'][0]
                print(json.dumps(first_item, indent=2))
                
                # Check for IV fields
                print("\n🔍 Checking for IV fields:")
                for key in first_item.keys():
                    if 'iv' in key.lower() or 'implied' in key.lower() or 'volatility' in key.lower():
                        print(f"  Found: {key} = {first_item[key]}")
                        
            elif isinstance(data['data'], dict):
                print("\nData structure:")
                print(json.dumps(data['data'], indent=2)[:500])
                
        # Check meta information
        if 'meta' in data:
            print(f"\nMeta information: {data['meta']}")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_api_structure() 