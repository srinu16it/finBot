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
    
    def get_news_sentiment(self, symbol: str, use_cache: bool = True) -> Optional[Dict]:
        """
        Get news sentiment analysis for a symbol using Yahoo Finance news.
        
        Args:
            symbol: Stock symbol
            use_cache: Whether to use cached data
            
        Returns:
            News sentiment data or None
        """
        params = {"data_type": "news_sentiment"}
        
        # Try cache first
        if use_cache:
            cached_data = self.cache_manager.get(self.provider_name, symbol, params)
            if cached_data:
                logger.info(f"Cache hit for {symbol} news sentiment")
                return cached_data
                
        try:
            logger.info(f"Fetching news for {symbol} from Yahoo Finance")
            ticker = yf.Ticker(symbol)
            
            # Get news articles
            news = ticker.news
            
            if not news:
                logger.warning(f"No news data found for {symbol}")
                return None
                
            # Process news for sentiment
            return self._process_news_sentiment(news, symbol)
            
        except Exception as e:
            logger.error(f"Error fetching news for {symbol}: {e}")
            return None
    
    def _process_news_sentiment(self, news: list, symbol: str) -> Dict:
        """
        Process news articles to estimate sentiment.
        
        Note: Yahoo Finance doesn't provide sentiment scores directly,
        so we'll use a simple keyword-based approach.
        """
        if not news:
            return {
                'sentiment_score': 0,
                'relevance_score': 0,
                'articles_analyzed': 0,
                'recent_articles': []
            }
            
        # Positive and negative keywords for basic sentiment analysis
        positive_keywords = [
            'upgrade', 'buy', 'outperform', 'positive', 'growth', 'surge', 'rally',
            'gain', 'profit', 'beat', 'strong', 'bullish', 'up', 'rise', 'soar',
            'record', 'high', 'breakthrough', 'innovative', 'leading', 'success',
            'opportunity', 'optimistic', 'boost', 'improve', 'exceed', 'golden'
        ]
        
        negative_keywords = [
            'downgrade', 'sell', 'underperform', 'negative', 'decline', 'fall',
            'drop', 'loss', 'miss', 'weak', 'bearish', 'down', 'plunge', 'crash',
            'low', 'concern', 'risk', 'warning', 'problem', 'issue', 'lawsuit',
            'difficult', 'challenge', 'struggle', 'fail', 'worst', 'crisis'
        ]
        
        total_sentiment = 0
        count = 0
        recent_articles = []
        
        # Analyze up to 10 most recent articles
        for article in news[:10]:
            # Extract content from the nested structure
            content = article.get('content', article)
            
            title = content.get('title', '').lower()
            summary = content.get('summary', '').lower()
            
            # Combine title and summary for analysis
            text = f"{title} {summary}"
            
            # Skip if no text
            if not text.strip():
                continue
                
            # Count positive and negative keywords
            positive_count = sum(1 for keyword in positive_keywords if keyword in text)
            negative_count = sum(1 for keyword in negative_keywords if keyword in text)
            
            # Calculate simple sentiment score (-1 to 1)
            if positive_count + negative_count > 0:
                sentiment = (positive_count - negative_count) / (positive_count + negative_count)
            else:
                sentiment = 0
                
            # Estimate relevance - higher if symbol is mentioned
            relevance = 1.0 if symbol.lower() in text else 0.7
            
            # Boost relevance for financial news sources
            provider_info = content.get('provider', {})
            provider_name = provider_info.get('displayName', '').lower()
            if provider_name in ['yahoo finance', 'reuters', 'bloomberg', 'marketwatch', 'cnbc']:
                relevance = min(relevance * 1.2, 1.0)
                
            total_sentiment += sentiment * relevance
            count += 1
            
            # Format timestamp
            pub_date = content.get('pubDate', '')
            if pub_date:
                try:
                    from datetime import datetime
                    # Parse ISO format date
                    dt = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
                    publish_time = dt.strftime('%Y-%m-%d %H:%M')
                except:
                    publish_time = pub_date
            else:
                publish_time = 'Unknown'
                
            recent_articles.append({
                'title': content.get('title', 'No title'),
                'sentiment': sentiment,
                'relevance': relevance,
                'time': publish_time,
                'publisher': provider_info.get('displayName', 'Unknown'),
                'link': content.get('canonicalUrl', {}).get('url', '')
            })
            
        # Calculate average sentiment
        avg_sentiment = total_sentiment / count if count > 0 else 0
        
        result = {
            'sentiment_score': avg_sentiment,
            'relevance_score': 0.75,  # Yahoo news is generally relevant
            'articles_analyzed': count,
            'recent_articles': recent_articles[:5],
            'source': 'Yahoo Finance',
            'method': 'keyword-based'
        }
        
        # Cache the result (30 minutes TTL for news)
        self.cache_manager.set(
            self.provider_name,
            symbol,
            result,
            params={"data_type": "news_sentiment"},
            ttl=1800  # 30 minutes
        )
        
        return result
    
    def get_4hour_data(self, symbol: str, period: str = "2mo", 
                       use_cache: bool = True) -> Optional[pd.DataFrame]:
        """
        Get 4-hour OHLCV data for a symbol.
        
        Note: Yahoo Finance doesn't directly support 4h interval,
        so we'll use 1h data and resample to 4h.
        
        Args:
            symbol: Stock symbol
            period: Time period (1mo, 2mo, 3mo)
            use_cache: Whether to use cached data if available
            
        Returns:
            DataFrame with 4-hour OHLCV data or None if error
        """
        # First try to get from cache
        params = {"period": period, "interval": "4h"}
        
        if use_cache:
            cached_data = self.cache_manager.get(self.provider_name, symbol, params)
            if cached_data:
                logger.info(f"Cache hit for {symbol} 4-hour data")
                return pd.DataFrame(cached_data)
        
        # Get 1-hour data and resample
        logger.info(f"Fetching 1-hour data for {symbol} to create 4-hour bars")
        
        try:
            ticker = yf.Ticker(symbol)
            # Get 60 days of 1-hour data
            hist_1h = ticker.history(period=period, interval="1h")
            
            if hist_1h.empty:
                logger.warning(f"No 1-hour data returned for {symbol}")
                return None
            
            # Resample to 4-hour bars
            # Group by 4-hour periods starting from market open (9:30 AM ET)
            hist_4h = hist_1h.resample('4h', offset='9h30min').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
            
            # Filter to only include bars during market hours
            # Market hours: 9:30 AM - 4:00 PM ET (6.5 hours)
            # So we get approximately 2 4-hour bars per day
            hist_4h['hour'] = hist_4h.index.hour
            hist_4h = hist_4h[(hist_4h['hour'] >= 9) & (hist_4h['hour'] <= 16)]
            hist_4h = hist_4h.drop('hour', axis=1)
            
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
            logger.error(f"Error fetching 4-hour data for {symbol}: {str(e)}")
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