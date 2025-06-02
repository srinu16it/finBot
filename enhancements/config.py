"""
Configuration settings for the enhanced pattern analyzer.
"""

import os


class Config:
    """Configuration settings for data providers and analysis."""
    
    # Data provider preferences
    # Set to 'alphavantage' to always use AlphaVantage
    # Set to 'yahoo' to always use Yahoo Finance  
    # Set to 'auto' to let the system decide based on environment
    DATA_PROVIDER_PREFERENCE = os.getenv('DATA_PROVIDER_PREFERENCE', 'auto')
    
    # Rate limiting settings for EC2
    # Yahoo Finance rate limit on EC2 (seconds between requests)
    YAHOO_EC2_RATE_LIMIT = float(os.getenv('YAHOO_EC2_RATE_LIMIT', '5.0'))
    
    # Yahoo Finance rate limit on local machine (seconds between requests)
    YAHOO_LOCAL_RATE_LIMIT = float(os.getenv('YAHOO_LOCAL_RATE_LIMIT', '2.0'))
    
    # Whether to automatically fallback to AlphaVantage when Yahoo fails
    AUTO_FALLBACK_TO_ALPHAVANTAGE = os.getenv('AUTO_FALLBACK_TO_ALPHAVANTAGE', 'true').lower() == 'true'
    
    # Cache settings
    CACHE_DIR = os.getenv('CACHE_DIR', 'cache')
    CACHE_DB_NAME = os.getenv('CACHE_DB_NAME', 'finbot_cache.db')
    
    # Analysis settings
    DEFAULT_ANALYSIS_PERIOD_DAYS = int(os.getenv('DEFAULT_ANALYSIS_PERIOD_DAYS', '90'))
    MIN_ADX_FOR_ENTRY = float(os.getenv('MIN_ADX_FOR_ENTRY', '20.0'))
    
    @classmethod
    def get_preferred_provider(cls, is_ec2: bool = False) -> str:
        """
        Get the preferred data provider based on configuration and environment.
        
        Args:
            is_ec2: Whether running on EC2
            
        Returns:
            'alphavantage' or 'yahoo'
        """
        if cls.DATA_PROVIDER_PREFERENCE == 'alphavantage':
            return 'alphavantage'
        elif cls.DATA_PROVIDER_PREFERENCE == 'yahoo':
            return 'yahoo'
        else:  # auto
            # On EC2, prefer AlphaVantage if available
            if is_ec2 and "ALPHAVANTAGE_API_KEY" in os.environ:
                return 'alphavantage'
            # Otherwise use Yahoo
            return 'yahoo' 