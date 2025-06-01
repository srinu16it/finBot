# AlphaVantage Data Provider Guide

## Overview

The enhanced AlphaVantage data provider offers access to comprehensive market data from [AlphaVantage](https://www.alphavantage.co/documentation/) with built-in caching and rate limiting.

## Features

- 🚦 **Automatic Rate Limiting**: 5 calls/minute (free tier) or higher for premium
- 💾 **DuckDB Caching**: Reduces API calls and improves performance
- 📊 **Multiple Data Types**: Daily, intraday, quotes, technical indicators, and options
- 🔑 **Premium Support**: Enhanced features for premium subscriptions

## Setup

### 1. Get Your API Key

Get your free API key from: https://www.alphavantage.co/support/#api-key

### 2. Set Environment Variables

```bash
# Required
export ALPHAVANTAGE_API_KEY="your_api_key_here"

# Optional - for premium features
export ALPHAVANTAGE_PREMIUM="true"  # If you have premium subscription
```

Or create a `.env` file:
```
ALPHAVANTAGE_API_KEY=your_api_key_here
ALPHAVANTAGE_PREMIUM=false
```

## Usage Examples

### Basic Usage

```python
from enhancements.data_providers.alpha_provider import AlphaVantageProvider

# Initialize provider
provider = AlphaVantageProvider()

# Get daily data
df = provider.get_daily("AAPL", outputsize="compact")
print(df.head())

# Get intraday data
df = provider.get_intraday("AAPL", interval="5min")
print(df.head())

# Get real-time quote
quote = provider.get_quote("AAPL")
print(f"Current price: ${quote['price']}")
```

### With Premium Features

```python
# Initialize with premium flag
provider = AlphaVantageProvider(is_premium=True)

# Access premium features like real-time options
options = provider.get_realtime_options("AAPL")
print(f"Calls: {len(options['calls'])} contracts")
print(f"Puts: {len(options['puts'])} contracts")
```

### Technical Indicators

```python
# Get RSI
rsi_df = provider.get_technical_indicator("AAPL", "RSI", interval="daily", time_period=14)

# Get MACD
macd_df = provider.get_technical_indicator("AAPL", "MACD", interval="daily")

# Get Bollinger Bands
bb_df = provider.get_technical_indicator("AAPL", "BBANDS", interval="daily")
```

## Available Functions

Based on the [AlphaVantage documentation](https://www.alphavantage.co/documentation/), the provider supports:

### Core Stock APIs
- **Daily/Weekly/Monthly**: Historical OHLCV data
- **Intraday**: 1min, 5min, 15min, 30min, 60min intervals
- **Quote**: Real-time price quotes

### Technical Indicators
- SMA, EMA, WMA, DEMA, TEMA
- RSI, MACD, STOCH, ADX
- BBANDS (Bollinger Bands)
- CCI, AROON, MFI
- And many more...

### Premium Features
- Real-time options data
- Extended historical data
- Higher rate limits

## Testing with Streamlit

Run the test UI:
```bash
streamlit run enhancements/tests/test_alphavantage_ui.py
```

This provides an interactive interface to:
- Test different data types
- Monitor cache performance
- Verify rate limiting
- Visualize data

## Caching Behavior

The provider uses intelligent caching with different TTLs:

| Data Type | Cache Duration |
|-----------|---------------|
| Daily/Weekly/Monthly | 1 hour |
| Intraday (1min, 5min) | 5 minutes |
| Intraday (15min+) | 10 minutes |
| Technical Indicators | 1 hour |
| Quotes | 1 minute |
| Options | 5 minutes |

## Rate Limiting

The provider automatically handles rate limits:

- **Free Tier**: 5 API calls per minute
- **Premium Tier**: Higher limits (contact AlphaVantage)

When the limit is reached, the provider will automatically wait before making the next request.

## Error Handling

The provider handles common errors gracefully:

```python
# Example with error handling
df = provider.get_daily("INVALID_SYMBOL")
if df is None:
    print("Failed to fetch data - check symbol or API limits")
```

## Best Practices

1. **Use Caching**: Always use caching unless you need real-time data
   ```python
   df = provider.get_daily("AAPL", use_cache=True)  # Default
   ```

2. **Check Rate Limits**: Monitor your API usage
   ```python
   # The provider logs rate limit warnings
   import logging
   logging.basicConfig(level=logging.INFO)
   ```

3. **Handle None Returns**: Always check if data was returned
   ```python
   df = provider.get_daily("AAPL")
   if df is not None:
       # Process data
   ```

4. **Use Appropriate Intervals**: Choose the right data granularity for your needs

## Troubleshooting

### Common Issues

1. **"ALPHAVANTAGE_API_KEY environment variable not set"**
   - Set your API key in the environment or .env file

2. **"API call frequency limit reached"**
   - You've hit the rate limit, wait 1 minute
   - Consider upgrading to premium

3. **No data returned**
   - Check if the symbol is valid
   - Verify your API key is active
   - Check AlphaVantage service status

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Integration with FinBot

The AlphaVantage provider can be used as an alternative or complement to Yahoo Finance:

```python
# Use in custom nodes
def enhanced_api_node(state):
    provider = AlphaVantageProvider()
    
    # Try AlphaVantage first
    df = provider.get_daily(state["symbol"])
    
    if df is None:
        # Fallback to Yahoo
        from enhancements.data_providers.yahoo_provider import YahooProvider
        yahoo = YahooProvider()
        df = yahoo.get_ohlcv(state["symbol"])
    
    state["ohlcv"] = df
    return state
```

## Further Resources

- [AlphaVantage API Documentation](https://www.alphavantage.co/documentation/)
- [Premium Plans](https://www.alphavantage.co/premium/)
- [Support](https://www.alphavantage.co/support/) 