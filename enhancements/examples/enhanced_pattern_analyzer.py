#!/usr/bin/env python3
"""
Enhanced Pattern Analyzer for FinBot

This module implements the complete enhanced pattern analysis workflow
with 7 technical patterns, options recommendations, and professional-grade
trading requirements.

Features:
- 7 technical patterns (Double Top/Bottom, Head & Shoulders, Triangles, etc.)
- Yahoo Finance data integration with 6-month history
- Technical indicators (EMA, RSI, MACD, ADX, ATR, HV)
- Weekly trend analysis
- Options strategy recommendations based on HV/IV analysis
- Professional entry conditions (ADX ≥ 20, weekly trend alignment)
- ATR-based stop losses
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import Optional, Dict, List, Any
from scipy.signal import argrelextrema

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Load .env file from project root
    env_path = Path(__file__).parent.parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

# Handle both API key naming conventions
if "ALPHA_VANTAGE_API_KEY" in os.environ and "ALPHAVANTAGE_API_KEY" not in os.environ:
    os.environ["ALPHAVANTAGE_API_KEY"] = os.environ["ALPHA_VANTAGE_API_KEY"]

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import data providers and utilities
from enhancements.data_providers.yahoo_provider import YahooProvider
from enhancements.data_providers.alpha_provider import AlphaVantageProvider
from enhancements.data_access.cache import CacheManager
from enhancements.patterns.confidence import PatternConfidenceEngine, update_pattern_outcome
from enhancements.patterns.candlestick_patterns import CandlestickPatternDetector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnhancedPatternDetector:
    """Enhanced pattern detection with multiple patterns and confidence scoring."""
    
    def __init__(self, cache_manager=None):
        """Initialize the detector."""
        self.cache_manager = cache_manager or CacheManager()
        self.confidence_engine = PatternConfidenceEngine()
        
    def detect_double_top_bottom(self, df: pd.DataFrame, min_peak_distance: int = 5) -> list:
        """Detect double top/bottom patterns."""
        patterns = []
        prices = df["Close"].values
        
        # Find local maxima and minima
        peaks = argrelextrema(prices, np.greater_equal, order=min_peak_distance)[0]
        troughs = argrelextrema(prices, np.less_equal, order=min_peak_distance)[0]
        
        # Double Top Detection
        if len(peaks) >= 2:
            for i in range(len(peaks)-1):
                peak1_idx, peak2_idx = peaks[i], peaks[i+1]
                peak1, peak2 = prices[peak1_idx], prices[peak2_idx]
                
                # Check if peaks are similar height
                if abs(peak1 - peak2) / peak1 < 0.03:  # Within 3%
                    # Find neckline (lowest point between peaks)
                    neckline = prices[peak1_idx:peak2_idx].min()
                    
                    # Check if price broke below neckline
                    if peak2_idx < len(prices) - 1:
                        current_price = prices[-1]
                        if current_price < neckline:
                            confidence_data = self.confidence_engine.get_pattern_confidence(
                                "double_top", "bearish"
                            )
                            patterns.append({
                                "pattern": "double_top",
                                "type": "bearish",
                                "start_idx": peak1_idx,
                                "end_idx": peak2_idx,
                                "peak1": peak1,
                                "peak2": peak2,
                                "neckline": neckline,
                                "current_price": current_price,
                                "confidence_score": confidence_data.get("confidence_score", 0.75),
                                "historical_win_rate": confidence_data.get("win_rate", 0.65),
                                "peaks": [(peak1_idx, peak1), (peak2_idx, peak2)],
                                "troughs": [(max(0, peak1_idx - 20), prices[max(0, peak1_idx - 20)]),
                                          (max(0, peak2_idx - 20), prices[max(0, peak2_idx - 20)])]
                            })
        
        # Double Bottom Detection
        if len(troughs) >= 2:
            # Keep track of which troughs have been used to avoid overlapping patterns
            used_troughs = set()
            
            for i in range(len(troughs)-1):
                # Skip if either trough has been used
                if i in used_troughs or i+1 in used_troughs:
                    continue
                    
                trough1_idx, trough2_idx = troughs[i], troughs[i+1]
                trough1, trough2 = prices[trough1_idx], prices[trough2_idx]
                
                # Check if troughs are similar depth
                if abs(trough1 - trough2) / trough1 < 0.03:  # Within 3%
                    # Find neckline (highest point between troughs)
                    neckline = prices[trough1_idx:trough2_idx].max()
                    
                    # Check if price broke above neckline
                    if trough2_idx < len(prices) - 1:
                        current_price = prices[-1]
                        if current_price > neckline:
                            confidence_data = self.confidence_engine.get_pattern_confidence(
                                "double_bottom", "bullish"
                            )
                            patterns.append({
                                "pattern": "double_bottom",
                                "type": "bullish",
                                "start_idx": trough1_idx,
                                "end_idx": trough2_idx,
                                "trough1": trough1,
                                "trough2": trough2,
                                "neckline": neckline,
                                "current_price": current_price,
                                "confidence_score": confidence_data.get("confidence_score", 0.75),
                                "historical_win_rate": confidence_data.get("win_rate", 0.65),
                                "peaks": [(max(0, trough1_idx - 20), prices[max(0, trough1_idx - 20)]),
                                          (max(0, trough2_idx - 20), prices[max(0, trough2_idx - 20)])],
                                "troughs": [(trough1_idx, trough1), (trough2_idx, trough2)]
                            })
                            # Mark these troughs as used
                            used_troughs.add(i)
                            used_troughs.add(i+1)
        
        return patterns
    
    def detect_head_and_shoulders(self, df: pd.DataFrame) -> list:
        """Detect head and shoulders patterns."""
        patterns = []
        prices = df["Close"].values
        
        # Find peaks and troughs
        peaks = argrelextrema(prices, np.greater_equal, order=5)[0]
        troughs = argrelextrema(prices, np.less_equal, order=5)[0]
        
        # Head and Shoulders (bearish)
        if len(peaks) >= 3:
            for i in range(len(peaks)-2):
                left_shoulder_idx = peaks[i]
                head_idx = peaks[i+1]
                right_shoulder_idx = peaks[i+2]
                
                left_shoulder = prices[left_shoulder_idx]
                head = prices[head_idx]
                right_shoulder = prices[right_shoulder_idx]
                
                # Check pattern criteria
                if (head > left_shoulder and head > right_shoulder and
                    abs(left_shoulder - right_shoulder) / left_shoulder < 0.05):
                    
                    # Find neckline
                    neckline = np.mean([
                        prices[left_shoulder_idx:head_idx].min(),
                        prices[head_idx:right_shoulder_idx].min()
                    ])
                    
                    current_price = prices[-1]
                    if current_price < neckline:
                        confidence_data = self.confidence_engine.get_pattern_confidence(
                            "head_and_shoulders", "bearish"
                        )
                        patterns.append({
                            "pattern": "head_and_shoulders",
                            "type": "bearish",
                            "left_shoulder": left_shoulder,
                            "head": head,
                            "right_shoulder": right_shoulder,
                            "neckline": neckline,
                            "current_price": current_price,
                            "confidence_score": confidence_data.get("confidence_score", 0.75),
                            "historical_win_rate": confidence_data.get("win_rate", 0.65),
                            "peaks": [(left_shoulder_idx, left_shoulder),
                                      (head_idx, head),
                                      (right_shoulder_idx, right_shoulder)],
                            "troughs": [(max(0, left_shoulder_idx - 20), prices[max(0, left_shoulder_idx - 20)]),
                                      (max(0, head_idx - 20), prices[max(0, head_idx - 20)]),
                                      (max(0, right_shoulder_idx - 20), prices[max(0, right_shoulder_idx - 20)])]
                        })
        
        # Inverse Head and Shoulders (bullish)
        if len(troughs) >= 3:
            for i in range(len(troughs)-2):
                left_shoulder_idx = troughs[i]
                head_idx = troughs[i+1]
                right_shoulder_idx = troughs[i+2]
                
                left_shoulder = prices[left_shoulder_idx]
                head = prices[head_idx]
                right_shoulder = prices[right_shoulder_idx]
                
                # Check pattern criteria
                if (head < left_shoulder and head < right_shoulder and
                    abs(left_shoulder - right_shoulder) / left_shoulder < 0.05):
                    
                    # Find neckline
                    neckline = np.mean([
                        prices[left_shoulder_idx:head_idx].max(),
                        prices[head_idx:right_shoulder_idx].max()
                    ])
                    
                    current_price = prices[-1]
                    if current_price > neckline:
                        confidence_data = self.confidence_engine.get_pattern_confidence(
                            "inverse_head_and_shoulders", "bullish"
                        )
                        patterns.append({
                            "pattern": "inverse_head_and_shoulders",
                            "type": "bullish",
                            "left_shoulder": left_shoulder,
                            "head": head,
                            "right_shoulder": right_shoulder,
                            "neckline": neckline,
                            "current_price": current_price,
                            "confidence_score": confidence_data.get("confidence_score", 0.75),
                            "historical_win_rate": confidence_data.get("win_rate", 0.65),
                            "peaks": [(max(0, left_shoulder_idx - 20), prices[max(0, left_shoulder_idx - 20)]),
                                      (max(0, head_idx - 20), prices[max(0, head_idx - 20)]),
                                      (max(0, right_shoulder_idx - 20), prices[max(0, right_shoulder_idx - 20)])],
                            "troughs": [(troughs[i], prices[troughs[i]]),
                                      (troughs[i+1], prices[troughs[i+1]]),
                                      (troughs[i+2], prices[troughs[i+2]])]
                        })
        
        return patterns
    
    def detect_triangles(self, df: pd.DataFrame, min_points: int = 4) -> list:
        """Detect triangle patterns (ascending, descending, symmetric)."""
        patterns = []
        prices = df["Close"].values
        
        # Get recent price action (last 30-60 periods)
        window = min(60, len(prices))
        recent_prices = prices[-window:]
        
        # Find peaks and troughs
        peaks = argrelextrema(recent_prices, np.greater_equal, order=3)[0]
        troughs = argrelextrema(recent_prices, np.less_equal, order=3)[0]
        
        if len(peaks) >= 2 and len(troughs) >= 2:
            # Fit trend lines
            peak_slope = np.polyfit(peaks, recent_prices[peaks], 1)[0]
            trough_slope = np.polyfit(troughs, recent_prices[troughs], 1)[0]
            
            # Ascending Triangle: flat top, rising bottom
            if abs(peak_slope) < 0.001 and trough_slope > 0.001:
                confidence_data = self.confidence_engine.get_pattern_confidence(
                    "ascending_triangle", "bullish"
                )
                patterns.append({
                    "pattern": "ascending_triangle",
                    "type": "bullish",
                    "resistance": np.mean(recent_prices[peaks]),
                    "support_slope": trough_slope,
                    "confidence_score": confidence_data.get("confidence_score", 0.75),
                    "historical_win_rate": confidence_data.get("win_rate", 0.65),
                    "peaks": [(peaks[i], recent_prices[peaks[i]]) for i in range(len(peaks))],
                    "troughs": [(troughs[i], recent_prices[troughs[i]]) for i in range(len(troughs))]
                })
            
            # Descending Triangle: falling top, flat bottom
            elif peak_slope < -0.001 and abs(trough_slope) < 0.001:
                confidence_data = self.confidence_engine.get_pattern_confidence(
                    "descending_triangle", "bearish"
                )
                patterns.append({
                    "pattern": "descending_triangle",
                    "type": "bearish",
                    "support": np.mean(recent_prices[troughs]),
                    "resistance_slope": peak_slope,
                    "confidence_score": confidence_data.get("confidence_score", 0.75),
                    "historical_win_rate": confidence_data.get("win_rate", 0.65),
                    "peaks": [(peaks[i], recent_prices[peaks[i]]) for i in range(len(peaks))],
                    "troughs": [(troughs[i], recent_prices[troughs[i]]) for i in range(len(troughs))]
                })
        
        return patterns
    
    def detect_all_patterns(self, df: pd.DataFrame) -> list:
        """Detect all supported patterns."""
        all_patterns = []
        
        # Detect various patterns
        all_patterns.extend(self.detect_double_top_bottom(df))
        all_patterns.extend(self.detect_head_and_shoulders(df))
        all_patterns.extend(self.detect_triangles(df))
        
        # Sort by confidence score
        all_patterns.sort(key=lambda x: x.get("confidence_score", 0), reverse=True)
        
        return all_patterns


class OptionsRecommendationEngine:
    """Generate options strategy recommendations based on patterns."""
    
    def __init__(self):
        """Initialize the engine."""
        self.strategies = {
            "bullish": [
                {
                    "name": "Long Call",
                    "description": "Buy call options to profit from upward movement",
                    "detailed_explanation": """
                    A Long Call strategy involves buying call options when you expect the stock price to rise.
                    
                    How it works:
                    - You pay a premium to buy the right (not obligation) to purchase shares at the strike price
                    - Maximum loss is limited to the premium paid
                    - Unlimited profit potential as stock rises
                    
                    Best for: Strong bullish outlook with defined risk
                    """,
                    "risk": "medium",
                    "complexity": "simple",
                    "timeline": {
                        "recommended_dte": "30-45 days",
                        "explanation": "Provides time for the move to develop while minimizing time decay"
                    },
                    "entry_exit": {
                        "entry": "Buy when IV is relatively low and pattern confirms",
                        "exit": "Sell when target reached or 2 weeks before expiration"
                    }
                },
                {
                    "name": "Bull Call Spread",
                    "description": "Buy ATM call, sell OTM call to reduce cost",
                    "detailed_explanation": """
                    A Bull Call Spread (Debit Spread) involves buying a call at one strike and selling a call at a higher strike.
                    
                    How it works:
                    - Buy call at strike A (ATM)
                    - Sell call at strike B (OTM)
                    - Net debit = Max loss
                    - Max profit = (Strike B - Strike A) - Net debit
                    
                    Best for: Moderate bullish outlook with defined risk/reward
                    """,
                    "risk": "low",
                    "complexity": "moderate",
                    "timeline": {
                        "recommended_dte": "30-60 days",
                        "explanation": "Allows time for directional move with reduced cost vs long call"
                    },
                    "entry_exit": {
                        "entry": "Enter when expecting moderate upward move",
                        "exit": "Close at 50-75% of max profit or near expiration"
                    }
                },
                {
                    "name": "Cash Secured Put",
                    "description": "Sell put options to generate income or buy at discount",
                    "detailed_explanation": """
                    A Cash Secured Put involves selling put options while holding cash to buy shares if assigned.
                    
                    How it works:
                    - Sell put option and collect premium
                    - Hold cash equal to 100 shares × strike price
                    - If assigned, buy shares at strike price (minus premium received)
                    - If not assigned, keep the premium
                    
                    Best for: Neutral to bullish outlook, wanting to own stock at lower price
                    """,
                    "risk": "medium",
                    "complexity": "moderate",
                    "timeline": {
                        "recommended_dte": "30-45 days",
                        "explanation": "Optimal for premium collection with manageable assignment risk"
                    },
                    "entry_exit": {
                        "entry": "Sell puts at support levels or desired entry price",
                        "exit": "Buy back at 50% profit or let expire if OTM"
                    }
                }
            ],
            "bearish": [
                {
                    "name": "Long Put",
                    "description": "Buy put options to profit from downward movement",
                    "detailed_explanation": """
                    A Long Put strategy involves buying put options when you expect the stock price to fall.
                    
                    How it works:
                    - Pay premium for the right to sell shares at strike price
                    - Maximum loss limited to premium paid
                    - Profit increases as stock falls below strike minus premium
                    
                    Best for: Strong bearish outlook or portfolio protection
                    """,
                    "risk": "medium",
                    "complexity": "simple",
                    "timeline": {
                        "recommended_dte": "30-45 days",
                        "explanation": "Balance between time for move and time decay"
                    },
                    "entry_exit": {
                        "entry": "Buy when resistance holds or breakdown confirms",
                        "exit": "Sell at target or 2 weeks before expiration"
                    }
                },
                {
                    "name": "Bear Put Spread",
                    "description": "Buy ATM put, sell OTM put to reduce cost",
                    "detailed_explanation": """
                    A Bear Put Spread involves buying a put at one strike and selling a put at a lower strike.
                    
                    How it works:
                    - Buy put at strike A (ATM)
                    - Sell put at strike B (OTM, lower)
                    - Net debit = Max loss
                    - Max profit = (Strike A - Strike B) - Net debit
                    
                    Best for: Moderate bearish outlook with limited risk
                    """,
                    "risk": "low",
                    "complexity": "moderate",
                    "timeline": {
                        "recommended_dte": "30-60 days",
                        "explanation": "Time for bearish move with reduced cost"
                    },
                    "entry_exit": {
                        "entry": "Enter on failed breakout or resistance rejection",
                        "exit": "Close at 50-75% max profit"
                    }
                },
                {
                    "name": "Covered Call",
                    "description": "Sell call options against stock position",
                    "detailed_explanation": """
                    A Covered Call involves owning 100 shares and selling call options against them.
                    
                    How it works:
                    - Own 100 shares of stock
                    - Sell 1 call option (usually OTM)
                    - Collect premium as income
                    - If called away, sell shares at strike price + keep premium
                    
                    Best for: Neutral to slightly bearish outlook on owned shares
                    """,
                    "risk": "low",
                    "complexity": "simple",
                    "timeline": {
                        "recommended_dte": "30-45 days",
                        "explanation": "Monthly income generation strategy"
                    },
                    "entry_exit": {
                        "entry": "Sell calls at resistance or above cost basis",
                        "exit": "Let expire or roll to next month"
                    }
                }
            ],
            "neutral": [
                {
                    "name": "Iron Condor",
                    "description": "Sell OTM call and put spreads for range-bound markets",
                    "detailed_explanation": """
                    An Iron Condor combines a bear call spread and bull put spread to profit from low volatility.
                    
                    How it works:
                    - Sell OTM call spread above current price
                    - Sell OTM put spread below current price
                    - Collect premium from both sides
                    - Profit if stock stays between short strikes
                    
                    Best for: Low volatility, range-bound markets
                    """,
                    "risk": "medium",
                    "complexity": "advanced",
                    "timeline": {
                        "recommended_dte": "30-45 days",
                        "explanation": "Optimal for theta decay while managing gamma risk"
                    },
                    "entry_exit": {
                        "entry": "Enter when IV is high and expecting consolidation",
                        "exit": "Close at 25-50% profit or manage at 21 DTE"
                    }
                },
                {
                    "name": "Straddle",
                    "description": "Buy ATM call and put for high volatility plays",
                    "detailed_explanation": """
                    A Long Straddle involves buying both a call and put at the same strike price.
                    
                    How it works:
                    - Buy ATM call option
                    - Buy ATM put option (same strike)
                    - Profit from large moves in either direction
                    - Loss if stock doesn't move enough to cover both premiums
                    
                    Best for: Expecting big move but uncertain of direction (earnings, events)
                    """,
                    "risk": "high",
                    "complexity": "moderate",
                    "timeline": {
                        "recommended_dte": "30-60 days",
                        "explanation": "Before major catalyst, allowing time for volatility expansion"
                    },
                    "entry_exit": {
                        "entry": "Buy before known catalyst when IV is relatively low",
                        "exit": "Sell after the move or IV expansion"
                    }
                }
            ]
        }
    
    def recommend_strategies(self, patterns: list, current_price: float) -> list:
        """Recommend options strategies based on detected patterns."""
        recommendations = []
        
        if not patterns:
            # No patterns, suggest neutral strategies
            market_direction = "neutral"
        else:
            # Determine overall market direction from patterns
            bullish_count = sum(1 for p in patterns if p["type"] == "bullish")
            bearish_count = sum(1 for p in patterns if p["type"] == "bearish")
            
            if bullish_count > bearish_count:
                market_direction = "bullish"
            elif bearish_count > bullish_count:
                market_direction = "bearish"
            else:
                market_direction = "neutral"
        
        # Get strategies for market direction
        available_strategies = self.strategies.get(market_direction, [])
        
        # Calculate strikes based on current price
        atm_strike = round(current_price / 5) * 5  # Round to nearest $5
        otm_call_strike = atm_strike + 5
        otm_put_strike = atm_strike - 5
        itm_call_strike = atm_strike - 5
        itm_put_strike = atm_strike + 5
        
        # Calculate specific dates
        from datetime import datetime, timedelta
        today = datetime.now()
        exp_30_days = today + timedelta(days=30)
        exp_45_days = today + timedelta(days=45)
        exp_60_days = today + timedelta(days=60)
        
        for strategy in available_strategies:
            recommendation = {
                "strategy_type": strategy["name"],
                "description": strategy["description"],
                "detailed_explanation": strategy["detailed_explanation"],
                "risk_level": strategy["risk"],
                "complexity": strategy["complexity"],
                "market_outlook": market_direction,
                "timeline": strategy["timeline"],
                "entry_exit": strategy["entry_exit"],
                "suggested_strikes": {},
                "suggested_expirations": {
                    "primary": exp_45_days.strftime("%Y-%m-%d"),
                    "alternative": exp_30_days.strftime("%Y-%m-%d"),
                    "conservative": exp_60_days.strftime("%Y-%m-%d")
                }
            }
            
            # Add specific strike recommendations
            if "Call" in strategy["name"]:
                if "Long" in strategy["name"]:
                    recommendation["suggested_strikes"]["call"] = atm_strike
                    recommendation["suggested_strikes"]["alternative_call"] = itm_call_strike
                elif "Spread" in strategy["name"]:
                    recommendation["suggested_strikes"]["buy_call"] = atm_strike
                    recommendation["suggested_strikes"]["sell_call"] = otm_call_strike
                elif "Covered" in strategy["name"]:
                    recommendation["suggested_strikes"]["sell_call"] = otm_call_strike
            
            if "Put" in strategy["name"]:
                if "Long" in strategy["name"]:
                    recommendation["suggested_strikes"]["put"] = atm_strike
                    recommendation["suggested_strikes"]["alternative_put"] = itm_put_strike
                elif "Spread" in strategy["name"]:
                    recommendation["suggested_strikes"]["buy_put"] = atm_strike
                    recommendation["suggested_strikes"]["sell_put"] = otm_put_strike
                elif "Cash Secured" in strategy["name"]:
                    recommendation["suggested_strikes"]["sell_put"] = otm_put_strike
            
            if "Iron Condor" in strategy["name"]:
                recommendation["suggested_strikes"]["call_spread"] = {
                    "sell": otm_call_strike,
                    "buy": otm_call_strike + 5
                }
                recommendation["suggested_strikes"]["put_spread"] = {
                    "sell": otm_put_strike,
                    "buy": otm_put_strike - 5
                }
            
            if "Straddle" in strategy["name"]:
                recommendation["suggested_strikes"]["strike"] = atm_strike
            
            recommendations.append(recommendation)
        
        return recommendations


def run_enhanced_analysis(symbol: str, use_alphavantage: bool = True, timeframe: str = "daily", period_days: int = 90):
    """
    Run enhanced pattern analysis on a symbol with advanced options requirements.
    
    Args:
        symbol: Stock symbol
        use_alphavantage: Whether to use AlphaVantage data
        timeframe: Analysis timeframe ('daily', 'weekly', 'intraday')
        period_days: Number of days to analyze
    """
    logger.info(f"Starting enhanced analysis for {symbol} - Timeframe: {timeframe}, Period: {period_days} days")
    
    # Initialize components
    cache_manager = CacheManager()
    pattern_detector = EnhancedPatternDetector(cache_manager)
    options_engine = OptionsRecommendationEngine()
    candlestick_detector = CandlestickPatternDetector()
    
    # Fetch data - always get 6 months for proper weekly analysis
    if use_alphavantage and "ALPHAVANTAGE_API_KEY" in os.environ:
        logger.info("Using AlphaVantage data provider")
        try:
            provider = AlphaVantageProvider(cache_manager)
            df = provider.get_daily(symbol, outputsize="full")
            if df is not None:
                # Sort to ensure newest data is last
                df = df.sort_index()
                # Use tail to get the MOST RECENT data, not the oldest!
                df = df.tail(max(period_days, 180))  # Get most recent 180+ days
            else:
                logger.warning("AlphaVantage data fetch failed, falling back to Yahoo Finance")
                use_alphavantage = False
        except Exception as e:
            logger.error(f"AlphaVantage error: {e}, falling back to Yahoo Finance")
            use_alphavantage = False
            df = None
    
    # Use Yahoo Finance as primary or fallback
    if not use_alphavantage or df is None:
        logger.info("Using Yahoo Finance data provider")
        provider = YahooProvider(cache_manager)
        
        # Convert period_days to Yahoo format
        if period_days <= 30:
            yahoo_period = "1mo"
        elif period_days <= 60:
            yahoo_period = "2mo"
        elif period_days <= 90:
            yahoo_period = "3mo"
        elif period_days <= 180:
            yahoo_period = "6mo"
        elif period_days <= 365:
            yahoo_period = "1y"
        else:
            yahoo_period = "2y"
            
        # Always fetch at least 6 months for proper weekly analysis
        min_period = "6mo"
        if yahoo_period in ["1mo", "2mo", "3mo"]:
            yahoo_period = min_period
            
        logger.info(f"Fetching {yahoo_period} of data for {period_days} day analysis")
        df = provider.get_ohlcv(symbol, period=yahoo_period, interval="1d")
        
        if df is not None and 'Date' in df.columns:
            df.set_index('Date', inplace=True)
    
    if df is None or df.empty:
        logger.error(f"Failed to fetch data for {symbol}")
        return None
    
    # Ensure we have datetime index for proper resampling
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception as e:
            logger.error(f"Failed to convert index to datetime: {e}")
            return None
    
    # Sort by date to ensure proper order
    df = df.sort_index()
    
    # Log data quality check
    logger.info(f"Data range: {df.index[0]} to {df.index[-1]}")
    logger.info(f"Total rows: {len(df)}")
    logger.info(f"Latest close price: ${df['Close'].iloc[-1]:.2f}")
    
    # Create weekly resampled data for trend analysis
    weekly_df = df.resample('W').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()  # Remove any NaN rows
    
    # Log weekly data check
    if len(weekly_df) > 0:
        logger.info(f"Weekly data: {len(weekly_df)} weeks")
        logger.info(f"Latest weekly close: ${weekly_df['Close'].iloc[-1]:.2f}")
    
    # For pattern detection, use only the requested period
    pattern_df = df.tail(period_days) if len(df) > period_days else df
    
    # Add technical indicators
    logger.info("Calculating technical indicators")
    pattern_df = pattern_df.copy()  # Avoid SettingWithCopyWarning
    pattern_df["EMA_9"] = pattern_df["Close"].ewm(span=9, adjust=False).mean()
    pattern_df["EMA_21"] = pattern_df["Close"].ewm(span=21, adjust=False).mean()
    
    # RSI Calculation - Fixed version
    def calculate_rsi(data, period=14):
        """Calculate RSI with proper handling of edge cases."""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        # Avoid division by zero
        rs = gain / loss.replace(0, 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        # Fill initial NaN values with 50 (neutral)
        rsi = rsi.fillna(50)
        
        return rsi
    
    # Calculate RSI
    pattern_df["RSI"] = calculate_rsi(pattern_df["Close"], period=14)
    
    # MACD
    pattern_df["MACD"] = pattern_df["Close"].ewm(span=12, adjust=False).mean() - pattern_df["Close"].ewm(span=26, adjust=False).mean()
    pattern_df["MACD_Signal"] = pattern_df["MACD"].ewm(span=9, adjust=False).mean()
    
    # ATR calculation
    high_low = pattern_df['High'] - pattern_df['Low']
    high_close = abs(pattern_df['High'] - pattern_df['Close'].shift(1))
    low_close = abs(pattern_df['Low'] - pattern_df['Close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    pattern_df['ATR'] = tr.rolling(window=14).mean()
    
    # ADX calculation - import from advanced analyzer
    from enhancements.examples.advanced_options_analyzer import AdvancedOptionsAnalyzer
    analyzer = AdvancedOptionsAnalyzer(cache_manager)
    pattern_df['ADX'] = analyzer.calculate_adx(pattern_df)
    
    # Historical Volatility
    daily_returns = pattern_df['Close'].pct_change()
    pattern_df['HV_60'] = daily_returns.rolling(window=60).std() * np.sqrt(252) * 100
    pattern_df['HV_30'] = daily_returns.rolling(window=30).std() * np.sqrt(252) * 100
    
    # Weekly indicators
    weekly_df['SMA_20'] = weekly_df['Close'].rolling(window=20).mean()
    
    # Debug weekly data issues
    if len(weekly_df) > 0:
        logger.info(f"Weekly data debugging:")
        logger.info(f"  Total weekly rows: {len(weekly_df)}")
        logger.info(f"  Weekly date range: {weekly_df.index[0]} to {weekly_df.index[-1]}")
        logger.info(f"  Last 3 weekly closes:")
        for i in range(max(-3, -len(weekly_df)), 0):
            logger.info(f"    {weekly_df.index[i]}: ${weekly_df['Close'].iloc[i]:.2f}")
        
        # Check for data quality issues
        if weekly_df['Close'].min() < 50:  # TSLA hasn't been below $50 recently
            logger.warning(f"  ⚠️ Suspicious weekly close found: ${weekly_df['Close'].min():.2f}")
    
    # Detect patterns
    logger.info("Detecting patterns")
    patterns = pattern_detector.detect_all_patterns(pattern_df)
    
    # Get current price - use real-time quote if available
    current_price = pattern_df["Close"].iloc[-1]
    
    # Try to get real-time quote for more accurate current price
    if use_alphavantage and "ALPHAVANTAGE_API_KEY" in os.environ:
        try:
            quote = provider.get_quote(symbol)
            if quote and 'price' in quote:
                current_price = quote['price']
                logger.info(f"Using real-time price: ${current_price}")
        except Exception as e:
            logger.warning(f"Failed to get real-time quote, using last close: {e}")
    
    # Get IV from options
    iv = None
    iv_data = None
    news_sentiment = None
    four_hour_timing = None  # New: 4-hour timing analysis
    
    try:
        # Try to get real-time options data with IV
        if use_alphavantage and "ALPHAVANTAGE_API_KEY" in os.environ:
            options_data = provider.get_options_chain(symbol)
            if options_data and 'atm_iv' in options_data:
                iv = options_data['atm_iv']
                iv_data = options_data
                logger.info(f"Got real-time IV: {iv:.1f}%")
                
                # Validate IV - reject unrealistic values
                if iv > 150:
                    logger.warning(f"Suspicious IV from AlphaVantage: {iv:.1f}% - using Yahoo fallback")
                    iv = None  # Force fallback to Yahoo
                    iv_data = None
            
            # Get news sentiment
            news = provider.get_news_sentiment(symbol)
            if news:
                news_sentiment = news
                logger.info(f"News sentiment score: {news['sentiment_score']:.2f}")
        
        # If no news from AlphaVantage, try Yahoo Finance
        if not news_sentiment:
            logger.info("Trying Yahoo Finance for news sentiment")
            yahoo_provider = YahooProvider(cache_manager)
            yahoo_news = yahoo_provider.get_news_sentiment(symbol)
            if yahoo_news:
                news_sentiment = yahoo_news
                logger.info(f"Yahoo news sentiment score: {yahoo_news['sentiment_score']:.2f}")
        
        # Fallback to previous IV calculation if needed
        if not iv:
            iv = analyzer.get_iv_from_options(symbol)
            
        # NEW: Get 4-hour chart data for timing
        logger.info("Fetching 4-hour chart data for entry timing")
        
        # Use the same provider as main analysis for consistency
        if use_alphavantage and "ALPHAVANTAGE_API_KEY" in os.environ and hasattr(provider, 'get_4hour_data'):
            logger.info("Using AlphaVantage for 4-hour data")
            df_4h = provider.get_4hour_data(symbol, outputsize="full")
        else:
            # Fallback to Yahoo for 4-hour data
            logger.info("Using Yahoo Finance for 4-hour data")
            if not 'yahoo_provider' in locals():
                yahoo_provider = YahooProvider(cache_manager)
            df_4h = yahoo_provider.get_4hour_data(symbol, period="2mo")
        
        if df_4h is not None and not df_4h.empty:
            # Convert Date column to index if needed
            if 'Date' in df_4h.columns:
                df_4h.set_index('Date', inplace=True)
            
            # Analyze 4-hour timing
            from enhancements.patterns.entry_timing_4h import FourHourEntryTiming
            timing_analyzer = FourHourEntryTiming()
            
            # Create daily signal summary for 4H analysis
            # Initialize pattern_bias before using it
            if patterns:
                bullish_patterns = sum(1 for p in patterns if p.get("type") == "bullish")
                bearish_patterns = sum(1 for p in patterns if p.get("type") == "bearish")
                
                if bullish_patterns > bearish_patterns:
                    pattern_bias = "bullish"
                elif bearish_patterns > bullish_patterns:
                    pattern_bias = "bearish"
                else:
                    pattern_bias = "neutral"
            else:
                pattern_bias = "neutral"
            
            daily_signal = {
                'market_outlook': pattern_bias,
                'current_price': current_price,
                'patterns': patterns[:1] if patterns else []  # Primary pattern
            }
            
            four_hour_timing = timing_analyzer.analyze_4h_entry(df_4h, daily_signal)
            logger.info(f"4H timing: {four_hour_timing.get('timing', 'unavailable')}")
        else:
            logger.warning("Could not fetch 4-hour data for timing analysis")
            
    except Exception as e:
        logger.warning(f"Failed to get enhanced data: {e}")
    
    # Advanced entry conditions check
    latest_daily = pattern_df.iloc[-1]
    latest_weekly = weekly_df.iloc[-1]
    
    # Check ADX
    adx_value = latest_daily['ADX'] if not pd.isna(latest_daily['ADX']) else 0
    adx_condition_met = adx_value >= 20
    
    # Determine market outlook based on patterns
    if patterns:
        bullish_patterns = sum(1 for p in patterns if p.get("type") == "bullish")
        bearish_patterns = sum(1 for p in patterns if p.get("type") == "bearish")
        
        if bullish_patterns > bearish_patterns:
            pattern_bias = "bullish"
        elif bearish_patterns > bullish_patterns:
            pattern_bias = "bearish"
        else:
            pattern_bias = "neutral"
    else:
        # No patterns detected - should be neutral, not directional
        pattern_bias = "neutral"
        logger.info("No patterns detected - setting bias to neutral")
    
    # Check weekly trend condition
    weekly_trend_condition_met = False
    if not pd.isna(latest_weekly.get('SMA_20', float('nan'))):
        if pattern_bias == 'bullish':
            weekly_trend_condition_met = latest_weekly['Close'] > latest_weekly['SMA_20']
        elif pattern_bias == 'bearish':
            weekly_trend_condition_met = latest_weekly['Close'] < latest_weekly['SMA_20']
    
    # All entry conditions
    entry_conditions_met = adx_condition_met and weekly_trend_condition_met and pattern_bias != "neutral"
    
    # Check news sentiment if available
    news_condition_met = True  # Default to true if no news data
    if news_sentiment and news_sentiment.get('sentiment_score') is not None:
        sentiment_score = news_sentiment['sentiment_score']
        if sentiment_score < -0.5:
            news_condition_met = False
            logger.warning(f"Negative news sentiment: {sentiment_score:.2f}")
        elif sentiment_score > 0.3:
            logger.info(f"Positive news sentiment: {sentiment_score:.2f}")
    
    # Update entry conditions with news
    entry_conditions_met = entry_conditions_met and news_condition_met
    
    # Check candlestick timing if other conditions are met
    candlestick_timing = None
    if entry_conditions_met and patterns:
        # Get the primary pattern for timing
        primary_pattern = patterns[0] if patterns else None
        if primary_pattern:
            candlestick_timing = candlestick_detector.get_entry_timing(
                pattern_df.tail(20), 
                primary_pattern['pattern']
            )
    
    # Check for exit warnings if in a position
    exit_warnings = []
    if pattern_bias != "neutral":
        position_type = "long" if pattern_bias == "bullish" else "short"
        exit_patterns = candlestick_detector.detect_exit_warnings(
            pattern_df.tail(10), 
            position_type
        )
        if exit_patterns:
            exit_warnings = exit_patterns
    
    # Generate options recommendations with advanced filters
    if entry_conditions_met:
        # Get enhanced recommendations based on HV/IV
        recommendations = []
        hv_60 = latest_daily.get('HV_60', 50)
        hv_30 = latest_daily.get('HV_30', 30)
        current_iv = iv if iv else hv_30
        
        if pattern_bias == 'bullish':
            if current_iv > hv_60 * 1.2:  # High IV
                recommendations.append({
                    'strategy_type': 'Bull Put Spread',
                    'description': 'Sell put spread to collect premium in high IV environment',
                    'detailed_explanation': 'IV is significantly higher than HV - sell premium',
                    'risk_level': 'medium',
                    'complexity': 'moderate',
                    'market_outlook': 'bullish',
                    'entry_conditions': 'All conditions met',
                    'stop_loss': current_price - 1.5 * latest_daily['ATR']
                })
            else:
                recommendations.append({
                    'strategy_type': 'Bull Call Spread',
                    'description': 'Buy call spread for directional play',
                    'detailed_explanation': 'Normal IV environment - buy directional spread',
                    'risk_level': 'medium',
                    'complexity': 'moderate',
                    'market_outlook': 'bullish',
                    'entry_conditions': 'All conditions met',
                    'stop_loss': current_price - 1.5 * latest_daily['ATR']
                })
        elif pattern_bias == 'bearish':
            if current_iv > hv_60 * 1.2:  # High IV
                recommendations.append({
                    'strategy_type': 'Bear Call Spread',
                    'description': 'Sell call spread to collect premium in high IV environment',
                    'detailed_explanation': 'IV is significantly higher than HV - sell premium',
                    'risk_level': 'medium',
                    'complexity': 'moderate',
                    'market_outlook': 'bearish',
                    'entry_conditions': 'All conditions met',
                    'stop_loss': current_price + 1.5 * latest_daily['ATR']
                })
            else:
                recommendations.append({
                    'strategy_type': 'Bear Put Spread',
                    'description': 'Buy put spread for directional play',
                    'detailed_explanation': 'Normal IV environment - buy directional spread',
                    'risk_level': 'medium',
                    'complexity': 'moderate',
                    'market_outlook': 'bearish',
                    'entry_conditions': 'All conditions met',
                    'stop_loss': current_price + 1.5 * latest_daily['ATR']
                })
    else:
        # No trade recommendations
        missing_conditions = []
        if not adx_condition_met:
            missing_conditions.append("ADX < 20")
        if not weekly_trend_condition_met:
            missing_conditions.append("Weekly trend not aligned")
        if pattern_bias == "neutral":
            if not patterns:
                missing_conditions.append("No patterns detected")
            else:
                missing_conditions.append("No clear pattern direction")
        if not news_condition_met:
            missing_conditions.append("Negative news sentiment")
            
        recommendations = [{
            'strategy_type': 'NO TRADE',
            'description': 'Entry conditions not met',
            'detailed_explanation': f'Missing: {", ".join(missing_conditions)}',
            'risk_level': 'none',
            'complexity': 'none',
            'market_outlook': pattern_bias,
            'entry_conditions': 'Not met'
        }]
    
    # Create analysis report
    report = {
        "symbol": symbol,
        "analysis_date": datetime.now().isoformat(),
        "current_price": current_price,
        "patterns_detected": len(patterns),
        "patterns": patterns,
        "technical_indicators": {
            "EMA_9": float(pattern_df["EMA_9"].iloc[-1]) if not pd.isna(pattern_df["EMA_9"].iloc[-1]) else 0.0,
            "EMA_21": float(pattern_df["EMA_21"].iloc[-1]) if not pd.isna(pattern_df["EMA_21"].iloc[-1]) else 0.0,
            "RSI": float(pattern_df["RSI"].iloc[-1]) if not pd.isna(pattern_df["RSI"].iloc[-1]) else 50.0,
            "MACD": float(pattern_df["MACD"].iloc[-1]) if not pd.isna(pattern_df["MACD"].iloc[-1]) else 0.0,
            "MACD_Signal": float(pattern_df["MACD_Signal"].iloc[-1]) if not pd.isna(pattern_df["MACD_Signal"].iloc[-1]) else 0.0,
            "ADX": float(adx_value),
            "ATR": float(latest_daily['ATR']) if not pd.isna(latest_daily['ATR']) else 0.0,
            "HV_60": float(latest_daily['HV_60']) if not pd.isna(latest_daily['HV_60']) else 0.0,
            "HV_30": float(latest_daily['HV_30']) if not pd.isna(latest_daily['HV_30']) else 0.0,
            "IV": float(iv) if iv else None,
            "IV_data": iv_data if iv_data else None
        },
        "advanced_conditions": {
            "ADX": adx_value,
            "ADX_condition_met": adx_condition_met,
            "weekly_close": float(latest_weekly['Close']),
            "weekly_SMA_20": float(latest_weekly['SMA_20']) if not pd.isna(latest_weekly.get('SMA_20', float('nan'))) else None,
            "weekly_trend_condition_met": weekly_trend_condition_met,
            "entry_conditions_met": entry_conditions_met,
            "pattern_bias": pattern_bias,
            "candlestick_timing": candlestick_timing if candlestick_timing else {
                "timing": "none",
                "description": "No candlestick patterns detected"
            },
            "news_sentiment": news_sentiment if news_sentiment else {
                "sentiment_score": 0,
                "relevance_score": 0,
                "articles_analyzed": 0
            },
            "news_condition_met": news_condition_met
        },
        "options_recommendations": recommendations,
        "market_outlook": pattern_bias if patterns else "no_patterns",
        "pattern_based_outlook": pattern_bias,
        "price_based_outlook": "bullish" if pattern_df["Close"].iloc[-1] > pattern_df["Close"].iloc[-20] else "bearish",
        "analysis_parameters": {
            "timeframe": timeframe,
            "period_days": period_days,
            "data_points": len(pattern_df),
            "date_range": {
                "start": pattern_df.index[0].strftime("%Y-%m-%d"),
                "end": pattern_df.index[-1].strftime("%Y-%m-%d")
            },
            "pattern_detection_window": "Daily" if timeframe == "daily" else "Weekly" if timeframe == "weekly" else "Hourly",
            "options_timeline": "30-45 days (optimal for daily patterns)"
        },
        "candlestick_timing": candlestick_timing,
        "exit_warnings": exit_warnings,
        "four_hour_timing": four_hour_timing  # NEW: 4-hour timing analysis
    }
    
    return report


def main():
    """Main execution function."""
    # Check for environment setup
    if "ALPHAVANTAGE_API_KEY" not in os.environ:
        logger.warning("ALPHAVANTAGE_API_KEY not set, will use Yahoo Finance")
    
    # Example analysis
    symbol = "AAPL"
    
    print(f"\n🔍 Enhanced Pattern Analysis for {symbol}")
    print("=" * 60)
    
    # Run analysis
    report = run_enhanced_analysis(symbol, use_alphavantage=True)
    
    if report:
        print(f"\n📊 Analysis Summary")
        print(f"Symbol: {report['symbol']}")
        print(f"Current Price: ${report['current_price']:.2f}")
        print(f"Patterns Detected: {report['patterns_detected']}")
        print(f"Market Outlook: {report['market_outlook'].upper()}")
        
        # Show patterns
        if report['patterns']:
            print(f"\n🎯 Detected Patterns:")
            for pattern in report['patterns'][:3]:  # Show top 3
                print(f"\n  • {pattern['pattern'].replace('_', ' ').title()}")
                print(f"    Type: {pattern['type']}")
                print(f"    Confidence: {pattern['confidence_score']:.1%}")
                print(f"    Historical Win Rate: {pattern['historical_win_rate']:.1%}")
        
        # Show technical indicators
        print(f"\n📈 Technical Indicators:")
        indicators = report['technical_indicators']
        print(f"  EMA 9/21: {indicators['EMA_9']:.2f} / {indicators['EMA_21']:.2f}")
        print(f"  RSI: {indicators['RSI']:.2f}")
        print(f"  MACD: {indicators['MACD']:.2f} (Signal: {indicators['MACD_Signal']:.2f})")
        print(f"  ADX: {indicators['ADX']:.1f}")
        print(f"  ATR: ${indicators['ATR']:.2f}")
        
        # Show volatility
        print(f"\n📊 Volatility:")
        print(f"  HV(60): {indicators['HV_60']:.1f}%")
        print(f"  HV(30): {indicators['HV_30']:.1f}%")
        if indicators['IV']:
            print(f"  IV: {indicators['IV']:.1f}%")
        
        # Show advanced conditions
        if 'advanced_conditions' in report:
            print(f"\n🎯 Advanced Entry Conditions:")
            adv = report['advanced_conditions']
            print(f"  Pattern Bias: {adv['pattern_bias'].upper()}")
            print(f"  ADX ≥ 20: {'✅' if adv['ADX_condition_met'] else '❌'} ({adv['ADX']:.1f})")
            print(f"  Weekly Trend: {'✅' if adv['weekly_trend_condition_met'] else '❌'}")
            if adv['weekly_SMA_20']:
                print(f"  Weekly Close: ${adv['weekly_close']:.2f} vs SMA: ${adv['weekly_SMA_20']:.2f}")
            print(f"  Entry Conditions Met: {'✅ YES' if adv['entry_conditions_met'] else '❌ NO'}")
        
        # Show options recommendations
        if report['options_recommendations']:
            print(f"\n💡 Options Strategy Recommendations:")
            for rec in report['options_recommendations'][:1]:  # Show primary recommendation
                print(f"\n  • {rec['strategy_type']}")
                print(f"    {rec['description']}")
                print(f"    Risk: {rec['risk_level']} | Complexity: {rec['complexity']}")
                if 'stop_loss' in rec and rec['stop_loss']:
                    print(f"    Stop Loss: ${rec['stop_loss']:.2f}")
                if 'entry_conditions' in rec:
                    print(f"    Entry: {rec['entry_conditions']}")
    else:
        print("❌ Analysis failed")


if __name__ == "__main__":
    main() 