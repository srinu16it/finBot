#!/usr/bin/env python3
"""
Pattern Analysis with AlphaVantage Data Integration.

This script demonstrates how to:
1. Fetch data from AlphaVantage
2. Run pattern detection
3. Generate options recommendations
4. Use confidence tracking

Usage:
    ./venv_test/bin/python enhancements/examples/pattern_analysis_with_alphavantage.py
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import FinBot components
from myfirstfinbot.nodes.api_node import api_node
from myfirstfinbot.nodes.chart_pattern_node import chart_pattern_node
from myfirstfinbot.nodes.market_analysis_node import market_analysis_node
from myfirstfinbot.nodes.options_strategy_node import options_strategy_node
from myfirstfinbot.state.graph_state import GraphState

# Import enhanced components
from enhancements.data_providers.alpha_provider import AlphaVantageProvider
from enhancements.data_access.cache import CacheManager
from enhancements.patterns.confidence import PatternConfidenceEngine

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnhancedPatternAnalyzer:
    """Enhanced pattern analyzer using AlphaVantage data and confidence tracking."""
    
    def __init__(self):
        """Initialize the analyzer with providers and engines."""
        # Initialize cache and data provider
        self.cache_manager = CacheManager(cache_dir="enhancements/cache")
        self.alpha_provider = AlphaVantageProvider(self.cache_manager)
        
        # Initialize confidence engine
        self.confidence_engine = PatternConfidenceEngine()
        
        # Available patterns to analyze
        self.patterns = [
            "double_top", "double_bottom",
            "triple_top", "triple_bottom",
            "head_and_shoulders", "inverse_head_and_shoulders",
            "ascending_wedge", "descending_wedge",
            "bullish_pennant", "bearish_pennant",
            "bullish_flag", "bearish_flag",
            "ascending_triangle", "descending_triangle"
        ]
    
    def fetch_enhanced_data(self, symbol: str, period_days: int = 90) -> pd.DataFrame:
        """
        Fetch comprehensive data from AlphaVantage.
        
        Args:
            symbol: Stock symbol
            period_days: Number of days of historical data
            
        Returns:
            DataFrame with OHLCV data and technical indicators
        """
        logger.info(f"Fetching data for {symbol}...")
        
        # Get daily OHLCV data
        ohlcv_df = self.alpha_provider.get_daily(symbol, outputsize="full")
        if ohlcv_df is None:
            raise ValueError(f"Failed to fetch data for {symbol}")
        
        # Filter to requested period
        ohlcv_df = ohlcv_df.head(period_days)
        
        # Fetch technical indicators
        logger.info("Fetching technical indicators...")
        
        # RSI
        rsi_df = self.alpha_provider.get_technical_indicator(
            symbol, "RSI", interval="daily", time_period=14
        )
        if rsi_df is not None:
            ohlcv_df['RSI'] = rsi_df['RSI'].reindex(ohlcv_df.index)
        
        # MACD
        macd_df = self.alpha_provider.get_technical_indicator(
            symbol, "MACD", interval="daily"
        )
        if macd_df is not None:
            ohlcv_df['MACD'] = macd_df['MACD'].reindex(ohlcv_df.index)
            ohlcv_df['MACD_Signal'] = macd_df['MACD_Signal'].reindex(ohlcv_df.index)
            ohlcv_df['MACD_Hist'] = macd_df['MACD_Hist'].reindex(ohlcv_df.index)
        
        # EMA
        ema9_df = self.alpha_provider.get_technical_indicator(
            symbol, "EMA", interval="daily", time_period=9
        )
        if ema9_df is not None:
            ohlcv_df['EMA_9'] = ema9_df['EMA'].reindex(ohlcv_df.index)
        
        ema21_df = self.alpha_provider.get_technical_indicator(
            symbol, "EMA", interval="daily", time_period=21
        )
        if ema21_df is not None:
            ohlcv_df['EMA_21'] = ema21_df['EMA'].reindex(ohlcv_df.index)
        
        # Calculate VWAP (simple approximation)
        ohlcv_df['VWAP'] = (ohlcv_df['Volume'] * (ohlcv_df['High'] + ohlcv_df['Low'] + ohlcv_df['Close']) / 3).cumsum() / ohlcv_df['Volume'].cumsum()
        
        return ohlcv_df
    
    def prepare_state(self, symbol: str, ohlcv_df: pd.DataFrame) -> GraphState:
        """
        Prepare GraphState for pattern analysis.
        
        Args:
            symbol: Stock symbol
            ohlcv_df: DataFrame with OHLCV and indicators
            
        Returns:
            Initialized GraphState
        """
        # Convert DataFrame to format expected by nodes
        ohlcv_data = ohlcv_df[['Open', 'High', 'Low', 'Close', 'Volume']].to_dict(orient='records')
        
        # Prepare indicators
        indicators = {
            'EMA_9': ohlcv_df['EMA_9'].dropna().to_dict() if 'EMA_9' in ohlcv_df else {},
            'EMA_21': ohlcv_df['EMA_21'].dropna().to_dict() if 'EMA_21' in ohlcv_df else {},
            'RSI': ohlcv_df['RSI'].dropna().to_dict() if 'RSI' in ohlcv_df else {},
            'MACD': {
                'MACD': ohlcv_df['MACD'].dropna().to_dict() if 'MACD' in ohlcv_df else {},
                'Signal': ohlcv_df['MACD_Signal'].dropna().to_dict() if 'MACD_Signal' in ohlcv_df else {},
                'Histogram': ohlcv_df['MACD_Hist'].dropna().to_dict() if 'MACD_Hist' in ohlcv_df else {}
            },
            'VWAP': ohlcv_df['VWAP'].dropna().to_dict() if 'VWAP' in ohlcv_df else {}
        }
        
        # Get current quote
        quote = self.alpha_provider.get_quote(symbol)
        current_price = quote['price'] if quote else ohlcv_df['Close'].iloc[0]
        
        # Initialize state
        state = GraphState(
            symbol=symbol,
            ohlcv=ohlcv_data,
            indicators=indicators,
            current_price=current_price,
            messages=[]
        )
        
        return state
    
    def analyze_patterns(self, state: GraphState) -> GraphState:
        """
        Run pattern analysis through FinBot nodes.
        
        Args:
            state: Current GraphState
            
        Returns:
            Updated GraphState with pattern analysis
        """
        logger.info("Running pattern detection...")
        
        # Run through pattern detection node
        state = chart_pattern_node(state)
        
        # Extract detected patterns
        if hasattr(state, 'detected_patterns') and state.detected_patterns:
            logger.info(f"Detected patterns: {state.detected_patterns}")
            
            # Update confidence scores for detected patterns
            for pattern in state.detected_patterns:
                pattern_name = pattern.get('pattern')
                pattern_type = pattern.get('type', 'neutral')
                
                # Map pattern type to direction
                direction = 'bullish' if 'bullish' in pattern_type else 'bearish'
                
                # Get confidence score
                confidence = self.confidence_engine.get_pattern_confidence(
                    pattern_name, direction
                )
                pattern['confidence_score'] = confidence.get('confidence_score', 0.5)
                pattern['historical_win_rate'] = confidence.get('win_rate', 0.5)
        
        return state
    
    def generate_recommendations(self, state: GraphState) -> GraphState:
        """
        Generate market analysis and options recommendations.
        
        Args:
            state: Current GraphState with patterns
            
        Returns:
            Updated GraphState with recommendations
        """
        logger.info("Generating market analysis...")
        
        # Run market analysis
        state = market_analysis_node(state)
        
        logger.info("Generating options strategies...")
        
        # Run options strategy generation
        state = options_strategy_node(state)
        
        return state
    
    def create_summary_report(self, state: GraphState) -> dict:
        """
        Create a comprehensive summary report.
        
        Args:
            state: Final GraphState
            
        Returns:
            Summary dictionary
        """
        report = {
            'symbol': state.symbol,
            'current_price': state.current_price,
            'analysis_timestamp': datetime.now().isoformat(),
            'patterns_detected': [],
            'market_analysis': {},
            'options_recommendations': [],
            'confidence_metrics': {}
        }
        
        # Add pattern information
        if hasattr(state, 'detected_patterns') and state.detected_patterns:
            for pattern in state.detected_patterns:
                pattern_info = {
                    'name': pattern.get('pattern'),
                    'type': pattern.get('type'),
                    'confidence_score': pattern.get('confidence_score', 0.5),
                    'historical_win_rate': pattern.get('historical_win_rate', 0.5),
                    'significance': pattern.get('significance', 'medium')
                }
                report['patterns_detected'].append(pattern_info)
        
        # Add market analysis
        if hasattr(state, 'market_analysis') and state.market_analysis:
            report['market_analysis'] = {
                'trend': state.market_analysis.get('trend', 'neutral'),
                'sentiment': state.market_analysis.get('sentiment', 'neutral'),
                'key_levels': state.market_analysis.get('key_levels', {}),
                'recommendation': state.market_analysis.get('recommendation', '')
            }
        
        # Add options recommendations
        if hasattr(state, 'options_strategies') and state.options_strategies:
            for strategy in state.options_strategies:
                strategy_info = {
                    'strategy_type': strategy.get('strategy_type'),
                    'description': strategy.get('description'),
                    'risk_level': strategy.get('risk_level', 'medium'),
                    'potential_profit': strategy.get('potential_profit'),
                    'max_loss': strategy.get('max_loss'),
                    'breakeven': strategy.get('breakeven')
                }
                report['options_recommendations'].append(strategy_info)
        
        # Add overall confidence metrics
        if report['patterns_detected']:
            avg_confidence = np.mean([p['confidence_score'] for p in report['patterns_detected']])
            avg_win_rate = np.mean([p['historical_win_rate'] for p in report['patterns_detected']])
            
            report['confidence_metrics'] = {
                'average_pattern_confidence': avg_confidence,
                'average_historical_win_rate': avg_win_rate,
                'recommendation_strength': 'strong' if avg_confidence > 0.7 else 'moderate' if avg_confidence > 0.5 else 'weak'
            }
        
        return report
    
    def analyze_symbol(self, symbol: str, period_days: int = 90) -> dict:
        """
        Complete analysis pipeline for a symbol.
        
        Args:
            symbol: Stock symbol
            period_days: Analysis period
            
        Returns:
            Analysis report
        """
        try:
            # Fetch data
            ohlcv_df = self.fetch_enhanced_data(symbol, period_days)
            
            # Prepare state
            state = self.prepare_state(symbol, ohlcv_df)
            
            # Run pattern analysis
            state = self.analyze_patterns(state)
            
            # Generate recommendations
            state = self.generate_recommendations(state)
            
            # Create report
            report = self.create_summary_report(state)
            
            return report
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {str(e)}")
            return {
                'error': str(e),
                'symbol': symbol,
                'analysis_timestamp': datetime.now().isoformat()
            }


def main():
    """Main execution function."""
    # Check for API key
    if "ALPHAVANTAGE_API_KEY" not in os.environ:
        print("❌ Please set the ALPHAVANTAGE_API_KEY environment variable!")
        print("Get your free API key at: https://www.alphavantage.co/support/#api-key")
        return
    
    # Initialize analyzer
    analyzer = EnhancedPatternAnalyzer()
    
    # Analyze a symbol
    symbol = "AAPL"
    print(f"\n🔍 Analyzing {symbol} with Enhanced Pattern Detection")
    print("=" * 60)
    
    # Run analysis
    report = analyzer.analyze_symbol(symbol, period_days=90)
    
    # Display results
    if 'error' in report:
        print(f"\n❌ Analysis failed: {report['error']}")
        return
    
    print(f"\n📊 Analysis Report for {report['symbol']}")
    print(f"Current Price: ${report['current_price']:.2f}")
    print(f"Analysis Time: {report['analysis_timestamp']}")
    
    # Pattern Detection Results
    print("\n🎯 Detected Patterns:")
    if report['patterns_detected']:
        for pattern in report['patterns_detected']:
            print(f"\n  • {pattern['name']} ({pattern['type']})")
            print(f"    Confidence: {pattern['confidence_score']:.2%}")
            print(f"    Historical Win Rate: {pattern['historical_win_rate']:.2%}")
            print(f"    Significance: {pattern['significance']}")
    else:
        print("  No significant patterns detected")
    
    # Market Analysis
    print("\n📈 Market Analysis:")
    if report['market_analysis']:
        print(f"  Trend: {report['market_analysis'].get('trend', 'N/A')}")
        print(f"  Sentiment: {report['market_analysis'].get('sentiment', 'N/A')}")
        print(f"  Recommendation: {report['market_analysis'].get('recommendation', 'N/A')}")
    
    # Options Recommendations
    print("\n💡 Options Strategy Recommendations:")
    if report['options_recommendations']:
        for i, strategy in enumerate(report['options_recommendations'], 1):
            print(f"\n  Strategy {i}: {strategy['strategy_type']}")
            print(f"  {strategy['description']}")
            print(f"  Risk Level: {strategy['risk_level']}")
            if strategy.get('potential_profit'):
                print(f"  Potential Profit: {strategy['potential_profit']}")
            if strategy.get('max_loss'):
                print(f"  Maximum Loss: {strategy['max_loss']}")
    else:
        print("  No specific options strategies recommended")
    
    # Confidence Metrics
    if report['confidence_metrics']:
        print("\n📊 Overall Confidence Metrics:")
        print(f"  Average Pattern Confidence: {report['confidence_metrics']['average_pattern_confidence']:.2%}")
        print(f"  Average Historical Win Rate: {report['confidence_metrics']['average_historical_win_rate']:.2%}")
        print(f"  Recommendation Strength: {report['confidence_metrics']['recommendation_strength'].upper()}")
    
    # Cache statistics
    print("\n💾 Cache Statistics:")
    stats = analyzer.cache_manager.get_stats()
    print(f"  Total Entries: {stats.get('total_entries', 0)}")
    print(f"  Hit Rate: {stats.get('hit_rate', 0):.2%}")
    
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main() 