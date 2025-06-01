"""
AlphaVantage data provider with caching and rate limiting.

This module provides access to AlphaVantage API with automatic
rate limiting to stay within free tier limits.
"""

import os
import time
import requests
import pandas as pd
from typing import Optional, Dict, Any
from datetime import datetime
import logging
from threading import Lock

from enhancements.data_access.cache import CacheManager


logger = logging.getLogger(__name__)


class AlphaVantageProvider:
    """
    AlphaVantage data provider with caching and rate limiting.
    
    Features:
    - Automatic rate limiting (5 calls/minute for free tier)
    - Caching of API responses
    - Error handling and retry logic
    - Support for multiple data types
    - Premium subscription support
    """
    
    BASE_URL = "https://www.alphavantage.co/query"
    RATE_LIMIT_FREE = 5  # Calls per minute for free tier
    RATE_LIMIT_PREMIUM = 75  # Calls per minute for premium tier (example)
    
    def __init__(self, cache_manager: Optional[CacheManager] = None, is_premium: bool = False):
        """
        Initialize the AlphaVantage provider.
        
        Args:
            cache_manager: Optional CacheManager instance
            is_premium: Whether using premium subscription (default: False)
        """
        # Try both naming conventions for API key
        self.api_key = os.getenv("ALPHAVANTAGE_API_KEY") or os.getenv("ALPHA_VANTAGE_API_KEY")
        if not self.api_key:
            raise ValueError("ALPHAVANTAGE_API_KEY or ALPHA_VANTAGE_API_KEY environment variable not set")
        
        # Check for premium flag in environment
        self.is_premium = is_premium or os.getenv("ALPHAVANTAGE_PREMIUM", "false").lower() == "true" or os.getenv("ALPHA_VANTAGE_PREMIUM", "false").lower() == "true"
        self.rate_limit = self.RATE_LIMIT_PREMIUM if self.is_premium else self.RATE_LIMIT_FREE
        
        self.cache_manager = cache_manager or CacheManager()
        self.provider_name = "alphavantage"
        
        # Rate limiting
        self._last_call_times = []
        self._rate_limit_lock = Lock()
        
        logger.info(f"AlphaVantage provider initialized. Premium: {self.is_premium}, Rate limit: {self.rate_limit}/min")
    
    def _rate_limit(self):
        """Enforce rate limiting based on subscription tier."""
        with self._rate_limit_lock:
            now = time.time()
            # Remove calls older than 1 minute
            self._last_call_times = [t for t in self._last_call_times if now - t < 60]
            
            # If we've made the limit calls in the last minute, wait
            if len(self._last_call_times) >= self.rate_limit:
                sleep_time = 60 - (now - self._last_call_times[0]) + 0.1
                if sleep_time > 0:
                    logger.info(f"Rate limit reached ({self.rate_limit}/min), sleeping for {sleep_time:.1f} seconds")
                    time.sleep(sleep_time)
            
            # Record this call
            self._last_call_times.append(time.time())
    
    def _make_request(self, params: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """
        Make a request to AlphaVantage API with rate limiting.
        
        Args:
            params: Request parameters
            
        Returns:
            JSON response or None if error
        """
        # Add API key to params
        params['apikey'] = self.api_key
        
        # Add premium parameter if applicable
        if self.is_premium:
            params['premium'] = 'true'
        
        # Enforce rate limit
        self._rate_limit()
        
        try:
            response = requests.get(self.BASE_URL, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Check for API errors
            if "Error Message" in data:
                logger.error(f"API error: {data['Error Message']}")
                return None
            elif "Note" in data:
                logger.warning(f"API note: {data['Note']}")
                # For rate limit messages, still return None
                if "Thank you for using Alpha Vantage" in data.get("Note", ""):
                    logger.error("API call frequency limit reached")
                    return None
                return None
            
            return data
            
        except Exception as e:
            logger.error(f"Error making AlphaVantage request: {str(e)}")
            return None
    
    def get_daily(self, symbol: str, outputsize: str = "compact", 
                  use_cache: bool = True) -> Optional[pd.DataFrame]:
        """
        Get daily time series data.
        
        Args:
            symbol: Stock symbol
            outputsize: 'compact' (100 days) or 'full' (20+ years)
            use_cache: Whether to use cached data
            
        Returns:
            DataFrame with daily OHLCV data or None if error
        """
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "outputsize": outputsize
        }
        
        # Try cache first
        if use_cache:
            cached_data = self.cache_manager.get(self.provider_name, symbol, params)
            if cached_data:
                logger.info(f"Cache hit for {symbol} daily data")
                return pd.DataFrame(cached_data)
        
        # Fetch from API
        logger.info(f"Fetching {symbol} daily data from AlphaVantage")
        data = self._make_request(params)
        
        if not data or "Time Series (Daily)" not in data:
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        # Rename columns to match standard naming
        column_mapping = {
            '1. open': 'Open',
            '2. high': 'High',
            '3. low': 'Low',
            '4. close': 'Close',
            '5. volume': 'Volume'
        }
        df = df.rename(columns=column_mapping)
        
        # Convert to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col])
        
        # Cache the data
        if use_cache:
            # Convert datetime index to string format before caching
            df_cache = df.copy()
            df_cache.index = df_cache.index.strftime('%Y-%m-%d %H:%M:%S')
            cache_data = df_cache.reset_index().rename(columns={'index': 'Date'}).to_dict(orient='records')
            self.cache_manager.set(
                self.provider_name,
                symbol,
                cache_data,
                params=params,
                ttl=3600  # 1 hour TTL
            )
        
        return df
    
    def get_intraday(self, symbol: str, interval: str = "5min",
                     outputsize: str = "compact", use_cache: bool = True) -> Optional[pd.DataFrame]:
        """
        Get intraday time series data.
        
        Args:
            symbol: Stock symbol
            interval: Time interval (1min, 5min, 15min, 30min, 60min)
            outputsize: 'compact' or 'full'
            use_cache: Whether to use cached data
            
        Returns:
            DataFrame with intraday OHLCV data or None if error
        """
        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": symbol,
            "interval": interval,
            "outputsize": outputsize
        }
        
        # Try cache first
        if use_cache:
            cached_data = self.cache_manager.get(self.provider_name, symbol, params)
            if cached_data:
                logger.info(f"Cache hit for {symbol} intraday data")
                df = pd.DataFrame(cached_data)
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                    df = df.set_index('Date')
                return df
        
        # Fetch from API
        logger.info(f"Fetching {symbol} intraday data from AlphaVantage")
        data = self._make_request(params)
        
        if not data:
            return None
        
        # Find the time series key
        time_series_key = f"Time Series ({interval})"
        if time_series_key not in data:
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(data[time_series_key], orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        # Rename columns to match standard naming
        column_mapping = {
            '1. open': 'Open',
            '2. high': 'High', 
            '3. low': 'Low',
            '4. close': 'Close',
            '5. volume': 'Volume'
        }
        df = df.rename(columns=column_mapping)
        
        # Convert to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col])
        
        # Cache the data
        if use_cache:
            cache_data = df.reset_index().rename(columns={'index': 'Date'}).to_dict(orient='records')
            # Shorter TTL for intraday data
            ttl = 300 if interval in ['1min', '5min'] else 600
            self.cache_manager.set(
                self.provider_name,
                symbol,
                cache_data,
                params=params,
                ttl=ttl
            )
        
        return df
    
    def get_technical_indicator(self, symbol: str, indicator: str,
                               interval: str = "daily", time_period: int = 14,
                               series_type: str = "close", use_cache: bool = True) -> Optional[pd.DataFrame]:
        """
        Get technical indicator data.
        
        Args:
            symbol: Stock symbol
            indicator: Indicator name (SMA, EMA, RSI, MACD, etc.)
            interval: Time interval
            time_period: Number of data points for indicator calculation
            series_type: Price type to use (close, open, high, low)
            use_cache: Whether to use cached data
            
        Returns:
            DataFrame with indicator values or None if error
        """
        params = {
            "function": indicator.upper(),
            "symbol": symbol,
            "interval": interval,
            "time_period": str(time_period),
            "series_type": series_type
        }
        
        # Try cache first
        if use_cache:
            cached_data = self.cache_manager.get(self.provider_name, symbol, params)
            if cached_data:
                logger.info(f"Cache hit for {symbol} {indicator} data")
                df = pd.DataFrame(cached_data)
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                    df = df.set_index('Date')
                return df
        
        # Fetch from API
        logger.info(f"Fetching {symbol} {indicator} data from AlphaVantage")
        data = self._make_request(params)
        
        if not data:
            return None
        
        # Find the technical analysis key
        ta_key = f"Technical Analysis: {indicator.upper()}"
        if ta_key not in data:
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(data[ta_key], orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        # Convert to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col])
        
        # Cache the data
        if use_cache:
            cache_data = df.reset_index().rename(columns={'index': 'Date'}).to_dict(orient='records')
            self.cache_manager.set(
                self.provider_name,
                symbol,
                cache_data,
                params=params,
                ttl=3600  # 1 hour TTL
            )
        
        return df
    
    def get_quote(self, symbol: str, use_cache: bool = True) -> Optional[Dict[str, Any]]:
        """
        Get current quote for a symbol using real-time data.
        
        Args:
            symbol: Stock symbol
            use_cache: Whether to use cached data
            
        Returns:
            Dictionary with quote data or None
        """
        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": symbol,
            "entitlement": "realtime"
        }
        
        # Check cache first
        if use_cache:
            cached_data = self.cache_manager.get(
                self.provider_name,
                symbol,
                params
            )
            if cached_data:
                logger.info(f"Cache hit for {symbol} quote")
                return cached_data
        
        # Fetch from API
        try:
            # Use the real-time endpoint
            url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&entitlement=realtime&apikey={self.api_key}"
            
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Check for errors
            if "Error Message" in data:
                logger.error(f"API error: {data['Error Message']}")
                return None
            
            if "Global Quote" not in data:
                logger.error(f"No quote data in response for {symbol}")
                return None
            
            # Extract quote data
            quote_data = data["Global Quote"]
            
            # Transform to consistent format
            result = {
                "symbol": quote_data.get("01. symbol", symbol),
                "price": float(quote_data.get("05. price", 0)),
                "open": float(quote_data.get("02. open", 0)),
                "high": float(quote_data.get("03. high", 0)),
                "low": float(quote_data.get("04. low", 0)),
                "volume": int(quote_data.get("06. volume", 0)),
                "latest_trading_day": quote_data.get("07. latest trading day"),
                "previous_close": float(quote_data.get("08. previous close", 0)),
                "change": float(quote_data.get("09. change", 0)),
                "change_percent": quote_data.get("10. change percent", "0%").replace("%", ""),
                "timestamp": datetime.now().isoformat()
            }
            
            # Cache with short TTL (1 minute for real-time quotes)
            if use_cache:
                self.cache_manager.set(
                    self.provider_name,
                    symbol,
                    result,
                    params=params,
                    ttl=60
                )
            
            logger.info(f"Successfully fetched real-time quote for {symbol}: ${result['price']}")
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error fetching quote: {e}")
            return None
        except (KeyError, ValueError) as e:
            logger.error(f"Error parsing quote data: {e}")
            return None
    
    def get_realtime_options(self, symbol: str, date: Optional[str] = None, 
                            use_cache: bool = True) -> Optional[Dict[str, pd.DataFrame]]:
        """
        Get realtime options data (Premium feature).
        
        Based on AlphaVantage documentation: https://www.alphavantage.co/documentation/
        
        Args:
            symbol: Stock symbol
            date: Options expiration date (YYYY-MM-DD format)
            use_cache: Whether to use cached data
            
        Returns:
            Dictionary with 'calls' and 'puts' DataFrames or None if error
        """
        if not self.is_premium:
            logger.warning("Realtime options require premium subscription")
            return None
            
        params = {
            "function": "REALTIME_OPTIONS",
            "symbol": symbol
        }
        
        if date:
            params["date"] = date
            
        # Try cache first
        if use_cache:
            cached_data = self.cache_manager.get(self.provider_name, symbol, params)
            if cached_data:
                logger.info(f"Cache hit for {symbol} options data")
                return {
                    'calls': pd.DataFrame(cached_data.get('calls', [])),
                    'puts': pd.DataFrame(cached_data.get('puts', []))
                }
        
        # Fetch from API
        logger.info(f"Fetching {symbol} realtime options from AlphaVantage")
        data = self._make_request(params)
        
        if not data:
            return None
            
        # Process options data
        calls_data = data.get('calls', [])
        puts_data = data.get('puts', [])
        
        result = {
            'calls': pd.DataFrame(calls_data),
            'puts': pd.DataFrame(puts_data)
        }
        
        # Cache the data
        if use_cache:
            cache_data = {
                'calls': calls_data,
                'puts': puts_data
            }
            self.cache_manager.set(
                self.provider_name,
                symbol,
                cache_data,
                params=params,
                ttl=300  # 5 minutes for options
            )
        
        return result
    
    def get_options_chain(self, symbol: str, require_greeks: bool = True) -> Optional[Dict]:
        """
        Get real-time options chain data with Greeks and IV.
        
        Args:
            symbol: Stock symbol
            require_greeks: Include Greeks and IV data
            
        Returns:
            Options chain data or None
        """
        params = {
            "function": "REALTIME_OPTIONS",
            "symbol": symbol,
            "require_greeks": str(require_greeks).lower(),
            "apikey": self.api_key  # Use regular API key - it works!
        }
        
        try:
            logger.info(f"Fetching options chain for {symbol}")
            data = self._make_request(params)
            
            # Log the raw response for debugging
            if data:
                logger.debug(f"API Response keys: {list(data.keys())}")
                if 'data' in data:  # Changed from 'options' to 'data'
                    logger.info(f"Found {len(data['data'])} options for {symbol}")
                    return self._process_options_data(data)
                else:
                    # Log what we actually got
                    logger.warning(f"No 'data' key in response. Keys found: {list(data.keys())}")
                    # Check for error messages
                    if 'Error Message' in data:
                        logger.error(f"API Error: {data['Error Message']}")
                    elif 'Note' in data:
                        logger.warning(f"API Note: {data['Note']}")
                    elif 'Information' in data:
                        logger.info(f"API Info: {data['Information']}")
            else:
                logger.warning(f"No data returned from API for {symbol}")
                
            return None
                
        except Exception as e:
            logger.error(f"Error fetching options chain: {e}")
            return None
    
    def get_4hour_data(self, symbol: str, outputsize: str = "full", 
                       use_cache: bool = True) -> Optional[pd.DataFrame]:
        """
        Get 4-hour OHLCV data for a symbol.
        
        AlphaVantage doesn't directly support 4h interval, so we'll use 60min 
        data and resample to 4h, similar to Yahoo approach.
        
        Args:
            symbol: Stock symbol
            outputsize: 'compact' or 'full' (full recommended for 2 months of data)
            use_cache: Whether to use cached data if available
            
        Returns:
            DataFrame with 4-hour OHLCV data or None if error
        """
        # First try to get from cache
        params = {"outputsize": outputsize, "interval": "4h"}
        
        if use_cache:
            cached_data = self.cache_manager.get(self.provider_name, symbol, params)
            if cached_data:
                logger.info(f"Cache hit for {symbol} 4-hour data")
                df = pd.DataFrame(cached_data)
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                    df = df.set_index('Date')
                return df
        
        # Get 60-minute data and resample
        logger.info(f"Fetching 60-minute data for {symbol} to create 4-hour bars")
        
        try:
            # Get 60-minute intraday data
            hist_60min = self.get_intraday(symbol, interval="60min", 
                                           outputsize=outputsize, use_cache=use_cache)
            
            if hist_60min is None or hist_60min.empty:
                logger.warning(f"No 60-minute data returned for {symbol}")
                return None
            
            # Resample to 4-hour bars
            # AlphaVantage returns EST/EDT times, so we align with market hours
            # Market hours: 9:30 AM - 4:00 PM ET
            # Create 4-hour bars without offset first
            hist_4h = hist_60min.resample('4h').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
            
            # Filter to only include bars during market hours (more flexible)
            # Keep bars between 4 AM and 8 PM ET to account for pre/post market
            hist_4h['hour'] = hist_4h.index.hour
            hist_4h = hist_4h[(hist_4h['hour'] >= 4) & (hist_4h['hour'] <= 20)]
            
            # Further filter to get approximately 2 bars per day
            # Keep bars starting around 8-10 AM and 12-2 PM
            hist_4h = hist_4h[
                ((hist_4h['hour'] >= 8) & (hist_4h['hour'] <= 10)) |
                ((hist_4h['hour'] >= 12) & (hist_4h['hour'] <= 14))
            ]
            hist_4h = hist_4h.drop('hour', axis=1)
            
            # If still empty, use all 4-hour bars without filtering
            if hist_4h.empty:
                logger.warning("Hour filtering resulted in empty DataFrame, using all 4-hour bars")
                hist_4h = hist_60min.resample('4h').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                }).dropna()
            
            # Reset index to make date a column
            hist_4h.reset_index(inplace=True)
            hist_4h.rename(columns={'index': 'Date'}, inplace=True)
            
            # Cache the data
            if use_cache and not hist_4h.empty:
                cache_data = hist_4h.to_dict(orient='records')
                self.cache_manager.set(
                    self.provider_name,
                    symbol,
                    cache_data,
                    params=params,
                    ttl=1800  # 30 minutes TTL for 4H data
                )
            
            return hist_4h
            
        except Exception as e:
            logger.error(f"Error creating 4-hour data for {symbol}: {str(e)}")
            return None
    
    def get_news_sentiment(self, symbol: str, time_from: Optional[str] = None) -> Optional[Dict]:
        """
        Get news sentiment analysis for a symbol.
        
        Args:
            symbol: Stock symbol
            time_from: Start time for news (format: YYYYMMDDTHHMM)
            
        Returns:
            News sentiment data or None
        """
        # Use specific realtime API key if available for news sentiment
        realtime_api_key = os.getenv("ALPHA_VANTAGE_REALTIME_API_KEY") or self.api_key
        
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": symbol,
            "apikey": realtime_api_key
        }
        
        if time_from:
            params["time_from"] = time_from
            
        try:
            logger.info(f"Fetching news sentiment for {symbol}")
            data = self._make_request(params)
            
            if data and 'feed' in data:
                # Calculate aggregate sentiment, pass the symbol
                return self._process_news_sentiment(data, symbol)
            else:
                logger.warning(f"No news data found for {symbol}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching news sentiment: {e}")
            return None
    
    def _process_options_data(self, data: Dict) -> Dict:
        """Process raw options data into structured format."""
        # The API returns data in 'data' field, not 'options'
        options = data.get('data', [])
        
        if not options:
            return None
        
        # Get underlying price from first option's mark or last price
        # Note: AlphaVantage doesn't provide underlying price directly
        # We'll estimate from ATM options
        current_price = None
        
        # Separate calls and puts
        calls = [opt for opt in options if opt.get('type') == 'call']
        puts = [opt for opt in options if opt.get('type') == 'put']
        
        # Estimate current price from ATM options
        if calls:
            # Find the call with highest open interest near the money
            sorted_calls = sorted(calls, key=lambda x: int(x.get('open_interest', 0)), reverse=True)
            for call in sorted_calls[:10]:  # Check top 10 by OI
                if float(call.get('mark', 0)) > 0:
                    strike = float(call.get('strike', 0))
                    mark = float(call.get('mark', 0))
                    # Rough estimate: current price = strike + call premium for near ATM
                    if not current_price and mark < strike * 0.1:  # Premium less than 10% of strike
                        current_price = strike + mark
                        break
        
        if not current_price and options:
            # Fallback: use the strike with most volume/OI
            current_price = float(sorted(options, key=lambda x: int(x.get('open_interest', 0)), reverse=True)[0].get('strike', 100))
        
        # Find ATM IV
        atm_call_iv = None
        atm_put_iv = None
        
        # Find closest strikes to current price
        if current_price:
            for call in calls:
                strike = float(call.get('strike', 0))
                if abs(strike - current_price) / current_price < 0.05:  # Within 5% of current price
                    iv_value = call.get('implied_volatility')
                    if iv_value:
                        # AlphaVantage returns IV in inconsistent format:
                        # - Values < 1 are decimals (0.67 = 67%)
                        # - Values > 5 are likely errors or already percentages
                        # Most realistic IVs are between 10% and 200%
                        iv_float = float(iv_value)
                        if iv_float < 1.0:
                            # This is a decimal representing percentage
                            atm_call_iv = iv_float * 100
                        elif iv_float > 5.0:
                            # This is likely an error or already percentage
                            # Don't multiply, just use as is (but it's probably wrong)
                            atm_call_iv = iv_float
                        else:
                            # Between 1 and 5, could be 1.5 = 150%
                            atm_call_iv = iv_float * 100
                        break
                        
            for put in puts:
                strike = float(put.get('strike', 0))
                if abs(strike - current_price) / current_price < 0.05:  # Within 5% of current price
                    iv_value = put.get('implied_volatility')
                    if iv_value:
                        # AlphaVantage returns IV in inconsistent format:
                        # - Values < 1 are decimals (0.67 = 67%)
                        # - Values > 5 are likely errors or already percentages
                        # Most realistic IVs are between 10% and 200%
                        iv_float = float(iv_value)
                        if iv_float < 1.0:
                            # This is a decimal representing percentage
                            atm_put_iv = iv_float * 100
                        elif iv_float > 5.0:
                            # This is likely an error or already percentage
                            # Don't multiply, just use as is (but it's probably wrong)
                            atm_put_iv = iv_float
                        else:
                            # Between 1 and 5, could be 1.5 = 150%
                            atm_put_iv = iv_float * 100
                        break
        
        # Calculate average IV and skew
        avg_iv = None
        if atm_call_iv and atm_put_iv:
            avg_iv = (atm_call_iv + atm_put_iv) / 2
            iv_skew = atm_put_iv - atm_call_iv
        elif atm_call_iv:
            avg_iv = atm_call_iv
            iv_skew = 0
        elif atm_put_iv:
            avg_iv = atm_put_iv
            iv_skew = 0
        else:
            iv_skew = 0
        
        return {
            'underlying_price': current_price,
            'atm_iv': avg_iv,
            'call_iv': atm_call_iv,
            'put_iv': atm_put_iv,
            'iv_skew': iv_skew,
            'options_count': len(options)
        }
    
    def _process_news_sentiment(self, data: Dict, symbol: str) -> Dict:
        """Process news sentiment into aggregate score."""
        feed = data.get('feed', [])
        
        if not feed:
            return {'sentiment_score': 0, 'relevance_score': 0, 'articles': 0}
        
        total_sentiment = 0
        total_relevance = 0
        count = 0
        
        recent_articles = []
        
        for article in feed[:10]:  # Last 10 articles
            ticker_sentiment = article.get('ticker_sentiment', [])
            for ts in ticker_sentiment:
                if ts.get('ticker') == symbol:
                    sentiment = float(ts.get('ticker_sentiment_score', 0))
                    relevance = float(ts.get('relevance_score', 0))
                    
                    total_sentiment += sentiment * relevance
                    total_relevance += relevance
                    count += 1
                    
                    recent_articles.append({
                        'title': article.get('title'),
                        'sentiment': sentiment,
                        'relevance': relevance,
                        'time': article.get('time_published')
                    })
        
        avg_sentiment = total_sentiment / total_relevance if total_relevance > 0 else 0
        
        return {
            'sentiment_score': avg_sentiment,
            'relevance_score': total_relevance / count if count > 0 else 0,
            'articles_analyzed': count,
            'recent_articles': recent_articles[:5]
        } 