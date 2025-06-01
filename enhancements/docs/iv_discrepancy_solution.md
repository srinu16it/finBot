# IV Discrepancy Solution

## Issue Identified
Yahoo Finance and AlphaVantage show very different IV values for TSLA:
- **Yahoo**: 62.3% (realistic)
- **AlphaVantage**: 161.6% (unrealistic)

## Root Cause
AlphaVantage appears to be:
1. Using far OTM options with inflated IVs (200%+ for puts)
2. Not properly identifying ATM strikes
3. Returning data in inconsistent formats (decimals vs percentages)

## Current Behavior
- **Yahoo IV < HV*1.2**: Shows "Normal IV - Buy Directional"
- **Alpha IV > HV*1.2**: Shows "High IV - Sell Premium"

With TSLA HV at 89.6%, the threshold is 107.5%:
- Yahoo 62.3% < 107.5% → Buy spreads
- Alpha 161.6% > 107.5% → Sell spreads

## Recommended Solution

### Short-term Fix
Trust Yahoo Finance IV data more than AlphaVantage for options strategies since:
1. Yahoo focuses on near-the-money options
2. Yahoo's IV aligns with market expectations
3. AlphaVantage may include illiquid far OTM options

### Long-term Fix
1. Validate IV ranges (reject if > 150% for most stocks)
2. Use multiple ATM strikes to calculate average
3. Filter options by open interest and volume
4. Cross-reference with other data sources

### User Guidance
When you see conflicting IV recommendations:
- **Yahoo says "Normal IV"**: Trust this for established stocks
- **Alpha says "High IV"**: Be skeptical if IV > 150%
- Check actual option prices on your broker
- High IV is typically 20-50% above HV, not 100%+ above

### Code Update Needed
```python
# In enhanced_pattern_analyzer.py
if iv and iv > 150:  # Unrealistic IV
    logger.warning(f"Suspicious IV from AlphaVantage: {iv}%")
    # Fall back to Yahoo or HV-based estimate
    iv = hv_30 * 1.1  # Use 10% premium as estimate
```

The discrepancy explains why you see different strategy recommendations. For now, trust Yahoo's assessment for realistic IV values. 