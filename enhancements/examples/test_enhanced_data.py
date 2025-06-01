#!/usr/bin/env python3
"""
Test Enhanced Data Integration (Options IV & News Sentiment)

This script demonstrates how real-time IV and news sentiment
enhance the pattern-based options recommendations.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from enhancements.examples.enhanced_pattern_analyzer import run_enhanced_analysis
from enhancements.data_providers.alpha_provider import AlphaVantageProvider
from enhancements.data_access.cache import CacheManager

def test_enhanced_recommendations():
    """Test how enhanced data improves recommendations."""
    
    print("Testing Enhanced Data Integration")
    print("=" * 60)
    print("Adding: Real-time IV + News Sentiment")
    print("=" * 60)
    print()
    
    # Test with a specific symbol
    symbol = "AAPL"
    
    print(f"📊 Analyzing {symbol} with enhanced data")
    print("-" * 40)
    
    # Run enhanced analysis
    report = run_enhanced_analysis(symbol, use_alphavantage=True)
    
    if not report:
        print("❌ Analysis failed")
        return
    
    # Show core conditions
    adv = report.get('advanced_conditions', {})
    print("\n🎯 Core Entry Conditions:")
    print(f"  Pattern Bias: {adv['pattern_bias']}")
    print(f"  ADX: {adv['ADX']:.1f} {'✅' if adv['ADX_condition_met'] else '❌'}")
    print(f"  Weekly Trend: {'✅' if adv['weekly_trend_condition_met'] else '❌'}")
    
    # Show enhanced data
    print("\n📰 News Sentiment:")
    news = adv.get('news_sentiment', {})
    if news and news.get('articles_analyzed', 0) > 0:
        sentiment = news['sentiment_score']
        print(f"  Sentiment Score: {sentiment:.2f}")
        print(f"  Articles Analyzed: {news['articles_analyzed']}")
        print(f"  News Check: {'✅' if adv.get('news_condition_met', True) else '❌ BLOCKED'}")
        
        if sentiment < -0.5:
            print("  ⚠️  Very negative news - Entry would be blocked!")
        elif sentiment > 0.3:
            print("  ✅ Positive news - Supports entry")
    else:
        print("  No news data available")
    
    # Show volatility analysis
    print("\n📊 Enhanced Volatility Analysis:")
    indicators = report['technical_indicators']
    if indicators['IV']:
        print(f"  Real-time IV: {indicators['IV']:.1f}%")
        print(f"  HV(60): {indicators['HV_60']:.1f}%")
        iv_hv = indicators['IV'] / indicators['HV_60'] if indicators['HV_60'] > 0 else 0
        print(f"  IV/HV Ratio: {iv_hv:.2f}")
        
        # IV-based strategy recommendation
        if iv_hv > 1.2:
            print("  💡 High IV environment - Prefer selling premium (spreads)")
        else:
            print("  💡 Normal IV - Can buy premium (long options)")
        
        # Check IV skew
        iv_data = indicators.get('IV_data', {})
        if iv_data and 'iv_skew' in iv_data:
            skew = iv_data['iv_skew']
            print(f"  IV Skew: {skew:.1f}%")
            if skew > 5:
                print("  📉 Put skew detected - Market fearful")
            elif skew < -5:
                print("  📈 Call skew detected - Market greedy")
    
    # Final recommendation
    print("\n💡 Enhanced Recommendation:")
    if adv['entry_conditions_met']:
        print("  ✅ ALL CONDITIONS MET - Trade recommended")
        
        # Show how enhanced data refines the strategy
        if news and news.get('sentiment_score', 0) > 0.3:
            print("  📰 Positive news adds confidence")
        
        if indicators['IV'] and indicators['IV'] / indicators['HV_60'] > 1.2:
            print("  📊 High IV suggests credit spreads over debit spreads")
    else:
        print("  ❌ Entry conditions not met")
        reasons = []
        if not adv['ADX_condition_met']:
            reasons.append("ADX < 20")
        if not adv['weekly_trend_condition_met']:
            reasons.append("Weekly trend misaligned")
        if not adv.get('news_condition_met', True):
            reasons.append("Negative news sentiment")
        print(f"  Reasons: {', '.join(reasons)}")
    
    # Compare with/without enhanced data
    print("\n📊 Impact of Enhanced Data:")
    print("  Without: Technical patterns + ADX + Weekly trend")
    print("  With: Above + Real-time IV + News sentiment + IV skew")
    print("  Result: More precise entries, better strategy selection")

if __name__ == "__main__":
    # Check if API key is set
    if "ALPHAVANTAGE_API_KEY" not in os.environ:
        print("⚠️  Set ALPHAVANTAGE_API_KEY to test enhanced features")
        print("   Basic analysis will run without real-time data")
    
    test_enhanced_recommendations() 