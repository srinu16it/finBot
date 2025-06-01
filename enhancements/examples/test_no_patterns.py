#!/usr/bin/env python3
"""
Test that no patterns = no trade recommendations
"""

import os
import sys
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from enhancements.examples.enhanced_pattern_analyzer import run_enhanced_analysis

def test_no_patterns_logic():
    """Test that when no patterns are detected, no trades are recommended."""
    
    print("Testing No Patterns Logic")
    print("=" * 60)
    
    # Test with a symbol that might not have patterns
    symbol = "UNH"  # From the user's screenshot
    
    print(f"\n📊 Analyzing {symbol}")
    report = run_enhanced_analysis(symbol, use_alphavantage=False, period_days=90)
    
    if not report:
        print("❌ Analysis failed")
        return
        
    # Check results
    print(f"\nPatterns Found: {report['patterns_detected']}")
    print(f"Market Outlook: {report['market_outlook']}")
    print(f"Pattern Bias: {report['advanced_conditions']['pattern_bias']}")
    print(f"Entry Conditions Met: {report['advanced_conditions']['entry_conditions_met']}")
    
    # Check recommendations
    recs = report['options_recommendations']
    if recs:
        rec = recs[0]
        print(f"\nRecommendation: {rec['strategy_type']}")
        print(f"Description: {rec['description']}")
        if 'detailed_explanation' in rec:
            print(f"Details: {rec['detailed_explanation']}")
    
    # Verify logic
    print("\n✅ Logic Check:")
    if report['patterns_detected'] == 0:
        print("  - No patterns detected ✓")
        
        if report['market_outlook'] == 'no_patterns':
            print("  - Market outlook shows 'no_patterns' ✓")
        else:
            print(f"  - ❌ Market outlook should be 'no_patterns', got '{report['market_outlook']}'")
            
        if report['advanced_conditions']['pattern_bias'] == 'neutral':
            print("  - Pattern bias is neutral ✓")
        else:
            print(f"  - ❌ Pattern bias should be 'neutral', got '{report['advanced_conditions']['pattern_bias']}'")
            
        if not report['advanced_conditions']['entry_conditions_met']:
            print("  - Entry conditions not met ✓")
        else:
            print("  - ❌ Entry conditions should NOT be met!")
            
        if recs and recs[0]['strategy_type'] == 'NO TRADE':
            print("  - Recommendation is NO TRADE ✓")
            if 'No patterns detected' in recs[0].get('detailed_explanation', ''):
                print("  - Explanation mentions no patterns ✓")
        else:
            print(f"  - ❌ Should recommend NO TRADE, got {recs[0]['strategy_type'] if recs else 'None'}")

if __name__ == "__main__":
    test_no_patterns_logic() 