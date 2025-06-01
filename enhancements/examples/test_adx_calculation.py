#!/usr/bin/env python3
"""
Test ADX calculation to ensure it's working correctly.

Usage:
    ./venv_test/bin/python enhancements/examples/test_adx_calculation.py
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import yfinance as yf

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from advanced_options_analyzer import AdvancedOptionsAnalyzer


def test_adx_calculation():
    """Test ADX calculation with known data."""
    print("Testing ADX Calculation")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = AdvancedOptionsAnalyzer()
    
    # Test with real data
    symbols = ["AAPL", "SPY", "TSLA"]
    
    for symbol in symbols:
        print(f"\n📊 Testing {symbol}")
        
        # Fetch data
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="6mo", interval="1d")
        
        if df.empty:
            print(f"Failed to fetch data for {symbol}")
            continue
        
        # Calculate ADX
        adx = analyzer.calculate_adx(df)
        
        # Show last 10 ADX values
        print(f"\nLast 10 ADX values:")
        last_10 = adx.tail(10)
        for date, value in last_10.items():
            print(f"{date.strftime('%Y-%m-%d')}: {value:.2f}")
        
        # Show statistics
        valid_adx = adx[adx > 0]
        if len(valid_adx) > 0:
            print(f"\nADX Statistics:")
            print(f"Mean ADX: {valid_adx.mean():.2f}")
            print(f"Max ADX: {valid_adx.max():.2f}")
            print(f"Min ADX: {valid_adx.min():.2f}")
            print(f"Current ADX: {adx.iloc[-1]:.2f}")
            
            # Check trend strength
            current_adx = adx.iloc[-1]
            if current_adx < 20:
                print("Trend Strength: WEAK (ADX < 20)")
            elif current_adx < 25:
                print("Trend Strength: MODERATE (20 ≤ ADX < 25)")
            elif current_adx < 50:
                print("Trend Strength: STRONG (25 ≤ ADX < 50)")
            else:
                print("Trend Strength: VERY STRONG (ADX ≥ 50)")
        else:
            print("No valid ADX values calculated")
    
    # Try with TA-Lib if available for comparison
    try:
        import talib
        print("\n" + "="*60)
        print("Comparing with TA-Lib (if available)")
        
        for symbol in symbols[:1]:  # Just test one symbol
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="6mo", interval="1d")
            
            if not df.empty:
                # Calculate with TA-Lib
                talib_adx = talib.ADX(df['High'].values, df['Low'].values, df['Close'].values, timeperiod=14)
                
                # Calculate with our method
                our_adx = analyzer.calculate_adx(df)
                
                print(f"\n{symbol} - Last 5 values comparison:")
                print("Date         | Our ADX | TA-Lib ADX | Difference")
                print("-" * 50)
                
                for i in range(-5, 0):
                    date = df.index[i]
                    our_val = our_adx.iloc[i]
                    talib_val = talib_adx[i] if not np.isnan(talib_adx[i]) else 0
                    diff = abs(our_val - talib_val)
                    print(f"{date.strftime('%Y-%m-%d')} | {our_val:7.2f} | {talib_val:10.2f} | {diff:10.2f}")
                    
    except ImportError:
        print("\nTA-Lib not available for comparison")


def main():
    """Run ADX calculation tests."""
    test_adx_calculation()


if __name__ == "__main__":
    main() 