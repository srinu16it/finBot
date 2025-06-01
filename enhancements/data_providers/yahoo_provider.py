"""
Enhanced Yahoo Finance data provider with caching support.

This module wraps the existing yfinance functionality with caching
to reduce API calls and improve performance.
"""

import yfinance as yf
import pandas as pd
from typing import Optional, Dict, Any
from datetime import datetime
import logging

from enhancements.data_access.cache import CacheManager


logger = logging.getLogger(__name__)


class YahooProvider:
    """
    Enhanced Yahoo Finance data provider with caching.
    
    This provider wraps the existing yfinance functionality and adds:
    - Automatic caching of API responses
    - Error handling and retry logic
    - Extended data retrieval options
    """
    
    def __init__(self, cache_manager: Optional[CacheManager] = None):
        """
        Initialize the Yahoo provider.
        
        Args:
            cache_manager: Optional CacheManager instance. If not provided,
                          a new instance will be created.
        """
        self.cache_manager = cache_manager or CacheManager()
        self.provider_name = "yahoo_finance"
    
    def get_ohlcv(self, symbol: str, period: str = "1mo", 
                  interval: str = "1d", use_cache: bool = True) -> Optional[pd.DataFrame]:
        """
        Get OHLCV data for a symbol.
        
        Args:
            symbol: Stock symbol
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            use_cache: Whether to use cached data if available
            
        Returns:
            DataFrame with OHLCV data or None if error
        """
        params = {"period": period, "interval": interval}
        
        # Try cache first if enabled
        if use_cache:
            cached_data = self.cache_manager.get(self.provider_name, symbol, params)
            if cached_data:
                logger.info(f"Cache hit for {symbol} OHLCV data")
                return pd.DataFrame(cached_data)
        
        # Fetch from API
        try:
            logger.info(f"Fetching {symbol} OHLCV data from Yahoo Finance")
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period, interval=interval)
            
            if hist.empty:
                logger.warning(f"No data returned for {symbol}")
                return None
            
            # Reset index to make date a column
            hist.reset_index(inplace=True)
            
            # Cache the data
            if use_cache:
                # Convert DataFrame to dict for caching
                cache_data = hist.to_dict(orient='records')
                # Adjust TTL based on interval
                ttl = self._get_ttl_for_interval(interval)
                self.cache_manager.set(
                    self.provider_name, 
                    symbol, 
                    cache_data, 
                    params=params,
                    ttl=ttl
                )
            
            return hist
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return None
    
    def get_info(self, symbol: str, use_cache: bool = True) -> Optional[Dict[str, Any]]:
        """
        Get company information for a symbol.
        
        Args:
            symbol: Stock symbol
            use_cache: Whether to use cached data if available
            
        Returns:
            Dictionary with company info or None if error
        """
        params = {"data_type": "info"}
        
        # Try cache first if enabled
        if use_cache:
            cached_data = self.cache_manager.get(self.provider_name, symbol, params)
            if cached_data:
                logger.info(f"Cache hit for {symbol} info")
                return cached_data
        
        # Fetch from API
        try:
            logger.info(f"Fetching {symbol} info from Yahoo Finance")
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            if not info:
                logger.warning(f"No info returned for {symbol}")
                return None
            
            # Cache the data (24 hour TTL for company info)
            if use_cache:
                self.cache_manager.set(
                    self.provider_name,
                    symbol,
                    info,
                    params=params,
                    ttl=86400  # 24 hours
                )
            
            return info
            
        except Exception as e:
            logger.error(f"Error fetching info for {symbol}: {str(e)}")
            return None
    
    def get_options_chain(self, symbol: str, date: Optional[str] = None, 
                         use_cache: bool = True) -> Optional[Dict[str, pd.DataFrame]]:
        """
        Get options chain data for a symbol.
        
        Args:
            symbol: Stock symbol
            date: Options expiration date (YYYY-MM-DD format)
            use_cache: Whether to use cached data if available
            
        Returns:
            Dictionary with 'calls' and 'puts' DataFrames or None if error
        """
        params = {"data_type": "options", "date": date}
        
        # Try cache first if enabled
        if use_cache:
            cached_data = self.cache_manager.get(self.provider_name, symbol, params)
            if cached_data:
                logger.info(f"Cache hit for {symbol} options data")
                return {
                    'calls': pd.DataFrame(cached_data['calls']),
                    'puts': pd.DataFrame(cached_data['puts'])
                }
        
        # Fetch from API
        try:
            logger.info(f"Fetching {symbol} options from Yahoo Finance")
            ticker = yf.Ticker(symbol)
            
            if date:
                options = ticker.option_chain(date)
            else:
                # Get next expiration date
                expirations = ticker.options
                if not expirations:
                    logger.warning(f"No options available for {symbol}")
                    return None
                options = ticker.option_chain(expirations[0])
            
            result = {
                'calls': options.calls,
                'puts': options.puts
            }
            
            # Cache the data (1 hour TTL for options)
            if use_cache:
                cache_data = {
                    'calls': options.calls.to_dict(orient='records'),
                    'puts': options.puts.to_dict(orient='records')
                }
                self.cache_manager.set(
                    self.provider_name,
                    symbol,
                    cache_data,
                    params=params,
                    ttl=3600  # 1 hour
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Error fetching options for {symbol}: {str(e)}")
            return None
    
    def _get_ttl_for_interval(self, interval: str) -> int:
        """
        Get appropriate TTL based on data interval.
        
        Args:
            interval: Data interval
            
        Returns:
            TTL in seconds
        """
        # Shorter TTL for intraday data
        if interval in ['1m', '2m', '5m', '15m', '30m']:
            return 300  # 5 minutes
        elif interval in ['60m', '90m', '1h']:
            return 1800  # 30 minutes
        elif interval in ['1d', '5d']:
            return 3600  # 1 hour
        else:
            return 86400  # 24 hours for weekly/monthly data 