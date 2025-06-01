#!/usr/bin/env python3
"""
Test Candlestick Pattern Integration

This script demonstrates how candlestick patterns complement the 
30-45 day options strategy without disrupting it.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from enhancements.examples.enhanced_pattern_analyzer import run_enhanced_analysis
import json

def test_candlestick_timing():
    """Test how candlestick patterns affect entry timing."""
    
    symbols = ['AAPL', 'MSFT', 'TSLA']
    
    print("Testing Candlestick Pattern Integration")
    print("=" * 60)
    print("Purpose: Enhance entry/exit timing for 30-45 day options")
    print("=" * 60)
    print()
    
    for symbol in symbols:
        print(f"\n📊 Analyzing {symbol}")
        print("-" * 40)
        
        # Get full analysis
        result = run_enhanced_analysis(symbol)
        
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
            continue
            
        # Show pattern detection
        patterns = result.get('patterns', [])
        if patterns:
            pattern = patterns[0]
            print(f"Pattern: {pattern['pattern'].replace('_', ' ').title()}")
            print(f"Confidence: {pattern.get('confidence_score', 0)*100:.0f}%")
        else:
            print("Pattern: None detected")
            
        # Show entry conditions
        adv = result.get('advanced_conditions', {})
        print(f"\nEntry Conditions:")
        print(f"  ADX: {adv['ADX']:.1f} {'✅' if adv['ADX_condition_met'] else '❌'}")
        print(f"  Weekly Trend: {'✅' if adv['weekly_trend_condition_met'] else '❌'}")
        print(f"  Pattern Bias: {adv['pattern_bias']}")
        print(f"  All Met: {'✅' if adv['entry_conditions_met'] else '❌'}")
        
        # Show candlestick timing
        timing = adv.get('candlestick_timing', {})
        print(f"\nCandlestick Timing:")
        print(f"  Status: {timing.get('timing', 'none')}")
        if timing.get('pattern'):
            print(f"  Pattern: {timing['pattern'].replace('_', ' ').title()}")
        print(f"  Action: {timing.get('description', 'No action')}")
        
        # Show exit warnings
        exit_warnings = result.get('exit_warnings', [])
        if exit_warnings:
            print(f"\n⚠️  Exit Warnings:")
            for warning in exit_warnings:
                print(f"  - {warning['pattern'].replace('_', ' ').title()} on {warning['date'].strftime('%Y-%m-%d')}")
                print(f"    {warning['description']}")
        
        # Show how it affects the strategy
        if adv['entry_conditions_met']:
            if timing.get('timing') == 'confirmed':
                print(f"\n💡 Strategy: ENTER NOW - {timing['pattern']} confirms entry")
            elif timing.get('timing') == 'wait':
                print(f"\n💡 Strategy: WAIT - Pattern ready but need candlestick confirmation")
            else:
                print(f"\n💡 Strategy: ENTER - No specific candlestick timing needed")
        else:
            print(f"\n💡 Strategy: NO TRADE - Entry conditions not met")
            
        # Show options recommendation if available
        recs = result.get('options_recommendations', [])
        if recs:
            print(f"\nOptions Trade:")
            print(f"  Strategy: {recs}")  # Just display the raw recommendation for now

if __name__ == "__main__":
    test_candlestick_timing() 