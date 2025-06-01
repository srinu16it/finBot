#!/usr/bin/env python3
"""
Test News Sentiment API
"""

import os
import sys
import requests
import json
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
env_path = Path(__file__).parent.parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path)

def test_news_api():
    """Test news sentiment API to see the actual response."""
    
    print("Testing NEWS_SENTIMENT API")
    print("=" * 60)
    
    api_key = "MYKYNF7L9MR0AMX9"
    symbols = ["NVDA", "TSLA", "AAPL"]
    
    for symbol in symbols:
        print(f"\n🧪 Testing {symbol} news...")
        
        # Direct API call
        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&apikey={api_key}"
        
        try:
            response = requests.get(url, timeout=10)
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                
                # Show structure
                print(f"Top-level keys: {list(data.keys())}")
                
                # Check feed
                if 'feed' in data:
                    feed = data['feed']
                    print(f"Feed items: {len(feed)}")
                    
                    if feed:
                        # Show first article
                        first_article = feed[0]
                        print("\nFirst article:")
                        print(f"  Title: {first_article.get('title', 'N/A')[:80]}...")
                        print(f"  Time: {first_article.get('time_published', 'N/A')}")
                        
                        # Check ticker sentiment
                        ticker_sentiment = first_article.get('ticker_sentiment', [])
                        print(f"  Ticker sentiments: {len(ticker_sentiment)}")
                        
                        # Find our ticker
                        for ts in ticker_sentiment:
                            if ts.get('ticker') == symbol:
                                print(f"\n  Found {symbol} sentiment:")
                                print(f"    Score: {ts.get('ticker_sentiment_score')}")
                                print(f"    Label: {ts.get('ticker_sentiment_label')}")
                                print(f"    Relevance: {ts.get('relevance_score')}")
                                break
                
                # Test through our provider
                print(f"\n📡 Testing through AlphaVantageProvider...")
                from enhancements.data_providers.alpha_provider import AlphaVantageProvider
                from enhancements.data_access.cache import CacheManager
                
                provider = AlphaVantageProvider(CacheManager())
                news_data = provider.get_news_sentiment(symbol)
                
                if news_data:
                    print("✅ Provider returned data:")
                    print(f"  Sentiment Score: {news_data.get('sentiment_score')}")
                    print(f"  Articles Analyzed: {news_data.get('articles_analyzed')}")
                else:
                    print("❌ Provider returned None")
                    
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_news_api() 