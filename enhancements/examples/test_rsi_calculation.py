#!/usr/bin/env python3
"""
Test script to verify RSI calculation is working correctly.

Usage:
    ./venv_test/bin/python enhancements/examples/test_rsi_calculation.py
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from enhancements.data_providers.yahoo_provider import YahooProvider
from enhancements.data_providers.alpha_provider import AlphaVantageProvider
from enhancements.data_access.cache import CacheManager


def calculate_rsi(data, period=14):
    """Calculate RSI with proper handling of edge cases."""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    # Avoid division by zero
    rs = gain / loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Fill initial NaN values with 50 (neutral)
    rsi = rsi.fillna(50)
    
    return rsi


def test_rsi_yahoo(symbol: str = "AAPL"):
    """Test RSI calculation with Yahoo Finance data."""
    print(f"\n📊 Testing RSI with Yahoo Finance for {symbol}")
    print("=" * 50)
    
    cache_manager = CacheManager()
    provider = YahooProvider(cache_manager)
    
    # Get data
    df = provider.get_ohlcv(symbol, period="3mo", interval="1d")
    
    if df is not None:
        if 'Date' in df.columns:
            df.set_index('Date', inplace=True)
        
        # Calculate RSI
        df['RSI'] = calculate_rsi(df['Close'])
        
        print(f"\nData shape: {df.shape}")
        print(f"Date range: {df.index[0]} to {df.index[-1]}")
        
        # Show last 10 RSI values
        print("\nLast 10 RSI values:")
        print(df[['Close', 'RSI']].tail(10))
        
        # RSI statistics
        rsi_stats = df['RSI'].describe()
        print(f"\nRSI Statistics:")
        print(rsi_stats)
        
        # Current RSI
        current_rsi = df['RSI'].iloc[-1]
        print(f"\nCurrent RSI: {current_rsi:.2f}")
        
        if current_rsi > 70:
            print("Status: OVERBOUGHT ⚠️")
        elif current_rsi < 30:
            print("Status: OVERSOLD ⚠️")
        else:
            print("Status: NEUTRAL ✓")
    else:
        print("❌ Failed to fetch data")


def test_rsi_alphavantage(symbol: str = "AAPL"):
    """Test RSI from AlphaVantage API."""
    if "ALPHAVANTAGE_API_KEY" not in os.environ:
        print("\n⚠️  Skipping AlphaVantage test - API key not set")
        return
    
    print(f"\n📊 Testing RSI with AlphaVantage for {symbol}")
    print("=" * 50)
    
    cache_manager = CacheManager()
    provider = AlphaVantageProvider(cache_manager)
    
    try:
        # Get RSI directly from API
        rsi_df = provider.get_technical_indicator(symbol, "RSI", interval="daily", time_period=14)
        
        if rsi_df is not None and not rsi_df.empty:
            print(f"\nData shape: {rsi_df.shape}")
            print(f"Date range: {rsi_df.index[0]} to {rsi_df.index[-1]}")
            
            # Show last 10 RSI values
            print("\nLast 10 RSI values from AlphaVantage:")
            print(rsi_df.tail(10))
            
            # Current RSI
            current_rsi = rsi_df['RSI'].iloc[-1]
            print(f"\nCurrent RSI: {current_rsi:.2f}")
            
            if current_rsi > 70:
                print("Status: OVERBOUGHT ⚠️")
            elif current_rsi < 30:
                print("Status: OVERSOLD ⚠️")
            else:
                print("Status: NEUTRAL ✓")
        else:
            print("❌ Failed to fetch RSI data")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")


def compare_rsi_methods(symbol: str = "AAPL"):
    """Compare RSI calculations between manual calculation and API."""
    print(f"\n🔄 Comparing RSI methods for {symbol}")
    print("=" * 50)
    
    cache_manager = CacheManager()
    
    # Get Yahoo data and calculate RSI manually
    yahoo_provider = YahooProvider(cache_manager)
    df = yahoo_provider.get_ohlcv(symbol, period="1mo", interval="1d")
    
    if df is not None:
        if 'Date' in df.columns:
            df.set_index('Date', inplace=True)
        
        df['RSI_Manual'] = calculate_rsi(df['Close'])
        manual_rsi = df['RSI_Manual'].iloc[-1]
        
        print(f"Manual RSI calculation: {manual_rsi:.2f}")
        
        # Try AlphaVantage RSI
        if "ALPHAVANTAGE_API_KEY" in os.environ:
            alpha_provider = AlphaVantageProvider(cache_manager)
            try:
                rsi_df = alpha_provider.get_technical_indicator(symbol, "RSI", interval="daily", time_period=14)
                if rsi_df is not None and not rsi_df.empty:
                    api_rsi = rsi_df['RSI'].iloc[-1]
                    print(f"AlphaVantage API RSI: {api_rsi:.2f}")
                    
                    diff = abs(manual_rsi - api_rsi)
                    print(f"Difference: {diff:.2f}")
                    
                    if diff < 2:
                        print("✅ RSI calculations are consistent")
                    else:
                        print("⚠️  RSI calculations show significant difference")
            except:
                pass


def main():
    """Run all RSI tests."""
    symbols = ["AAPL", "MSFT", "TSLA"]
    
    for symbol in symbols:
        print(f"\n{'#' * 60}")
        print(f"Testing {symbol}")
        print(f"{'#' * 60}")
        
        test_rsi_yahoo(symbol)
        test_rsi_alphavantage(symbol)
        compare_rsi_methods(symbol)


if __name__ == "__main__":
    main() 