#!/usr/bin/env python3
"""
Test Yahoo Finance News Sentiment
"""

import os
import sys
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from enhancements.data_providers.yahoo_provider import YahooProvider
from enhancements.data_access.cache import CacheManager

def test_yahoo_news():
    """Test Yahoo Finance news sentiment."""
    
    print("Testing Yahoo Finance News Sentiment")
    print("=" * 60)
    
    provider = YahooProvider(CacheManager())
    symbols = ["NVDA", "TSLA", "AAPL"]
    
    for symbol in symbols:
        print(f"\n🧪 Testing {symbol} news...")
        
        try:
            # Get news sentiment
            news_data = provider.get_news_sentiment(symbol)
            
            if news_data:
                print("✅ News data retrieved:")
                print(f"  Sentiment Score: {news_data['sentiment_score']:.3f}")
                print(f"  Articles Analyzed: {news_data['articles_analyzed']}")
                print(f"  Source: {news_data['source']}")
                print(f"  Method: {news_data['method']}")
                
                # Show recent articles
                if news_data.get('recent_articles'):
                    print("\n  Recent Articles:")
                    for i, article in enumerate(news_data['recent_articles'][:3]):
                        print(f"\n  Article {i+1}:")
                        print(f"    Title: {article['title'][:80]}...")
                        print(f"    Sentiment: {article['sentiment']:.2f}")
                        print(f"    Publisher: {article['publisher']}")
                        print(f"    Time: {article['time']}")
            else:
                print("❌ No news data returned")
                
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Compare with AlphaVantage
    print("\n\n📊 Comparing with AlphaVantage...")
    print("-" * 40)
    
    if "ALPHAVANTAGE_API_KEY" in os.environ or "ALPHA_VANTAGE_API_KEY" in os.environ:
        from enhancements.data_providers.alpha_provider import AlphaVantageProvider
        alpha_provider = AlphaVantageProvider(CacheManager())
        
        for symbol in ["NVDA"]:
            print(f"\nComparing {symbol}:")
            
            # Yahoo
            yahoo_news = provider.get_news_sentiment(symbol)
            if yahoo_news:
                print(f"  Yahoo Sentiment: {yahoo_news['sentiment_score']:.3f}")
            
            # AlphaVantage
            alpha_news = alpha_provider.get_news_sentiment(symbol)
            if alpha_news:
                print(f"  Alpha Sentiment: {alpha_news['sentiment_score']:.3f}")
    else:
        print("AlphaVantage API key not set, skipping comparison")

if __name__ == "__main__":
    test_yahoo_news() 