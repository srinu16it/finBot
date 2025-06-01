#!/usr/bin/env python3
"""
Compare analysis across different timeframes to understand discrepancies.

Usage:
    ./venv_test/bin/python enhancements/examples/compare_timeframes.py
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from enhanced_pattern_analyzer import run_enhanced_analysis


def compare_timeframes(symbol: str):
    """Compare analysis results across different timeframes."""
    print(f"\n🔍 Comparing Timeframe Analysis for {symbol}")
    print("=" * 60)
    
    timeframes = [
        ("Daily (30 days)", "daily", 30),
        ("Daily (90 days)", "daily", 90),
        ("Weekly (6 months)", "weekly", 180),
    ]
    
    results = []
    
    for name, timeframe, period in timeframes:
        print(f"\n📊 Analyzing {name}...")
        
        try:
            report = run_enhanced_analysis(
                symbol,
                use_alphavantage=False,  # Use Yahoo for consistency
                timeframe=timeframe,
                period_days=period
            )
            
            if report:
                results.append({
                    "Timeframe": name,
                    "Patterns": len(report.get("patterns", [])),
                    "Pattern Types": ", ".join([p["type"] for p in report.get("patterns", [])][:3]),
                    "Market Outlook": report.get("market_outlook", "N/A").upper(),
                    "Price Trend": report.get("price_based_outlook", "N/A").upper(),
                    "RSI": f"{report['technical_indicators'].get('RSI', 0):.1f}",
                    "Current Price": f"${report.get('current_price', 0):.2f}"
                })
            else:
                results.append({
                    "Timeframe": name,
                    "Patterns": 0,
                    "Pattern Types": "Error",
                    "Market Outlook": "ERROR",
                    "Price Trend": "ERROR",
                    "RSI": "N/A",
                    "Current Price": "N/A"
                })
                
        except Exception as e:
            print(f"Error: {str(e)}")
            results.append({
                "Timeframe": name,
                "Patterns": 0,
                "Pattern Types": "Error",
                "Market Outlook": "ERROR",
                "Price Trend": "ERROR",
                "RSI": "N/A",
                "Current Price": "N/A"
            })
    
    # Display comparison table
    if results:
        df = pd.DataFrame(results)
        print("\n📈 Timeframe Comparison:")
        print(df.to_string(index=False))
        
        # Analysis summary
        print("\n💡 Key Insights:")
        
        # Check for outlook differences
        outlooks = [r["Market Outlook"] for r in results if r["Market Outlook"] != "ERROR"]
        if len(set(outlooks)) > 1:
            print("⚠️  Different timeframes show different market outlooks!")
            print("   This is common and indicates:")
            print("   - Short-term vs long-term divergence")
            print("   - Possible trend reversal points")
            print("   - Need for multi-timeframe confirmation")
        else:
            print("✅ All timeframes show consistent outlook")
        
        # Pattern analysis
        daily_patterns = results[0]["Patterns"] if results else 0
        weekly_patterns = results[2]["Patterns"] if len(results) > 2 else 0
        
        if daily_patterns > 0 and weekly_patterns == 0:
            print("\n📊 Daily patterns detected but not on weekly:")
            print("   - May be short-term formations")
            print("   - Good for options with 30-45 day expiration")
            print("   - Monitor for weekly confirmation")
        elif weekly_patterns > 0 and daily_patterns == 0:
            print("\n📊 Weekly patterns detected but not on daily:")
            print("   - Longer-term formations developing")
            print("   - May need more time to play out")
            print("   - Consider longer-dated options")


def explain_tradingview_difference():
    """Explain why our analysis might differ from TradingView."""
    print("\n" + "=" * 60)
    print("📺 Why Our Analysis May Differ from TradingView:")
    print("=" * 60)
    
    print("""
1. **Timeframe Differences**:
   - Our default: Daily charts (90 days)
   - TradingView: Often uses weekly or longer timeframes
   - Different timeframes = different patterns and signals

2. **Indicator Calculations**:
   - RSI, MACD periods may differ
   - Moving average types (SMA vs EMA)
   - Calculation methods can vary

3. **Pattern Recognition**:
   - We focus on specific chart patterns (Double Top/Bottom, H&S, etc.)
   - TradingView may use different pattern algorithms
   - AI/ML approaches vs traditional technical analysis

4. **Analysis Window**:
   - Our patterns: Based on recent price action
   - TradingView: May consider longer historical context
   - Different lookback periods affect signals

5. **Options vs Stock Trading**:
   - Our recommendations: Optimized for options (30-45 days)
   - TradingView: General trading signals
   - Options require different timing considerations

**Best Practice**: 
- Use multiple timeframes for confirmation
- Compare daily AND weekly analysis
- Consider both pattern and indicator signals
- Align timeframe with your trading style
""")


def main():
    """Run timeframe comparison."""
    symbols = ["QUBT", "AAPL", "TSLA"]
    
    for symbol in symbols:
        compare_timeframes(symbol)
    
    explain_tradingview_difference()


if __name__ == "__main__":
    main() 