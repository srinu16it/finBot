"""
4-Hour Chart Entry Timing Module

This module provides enhanced entry timing using 4-hour charts to supplement
daily pattern analysis. Each 4-hour bar represents approximately 2 trading days
per week of price behavior, making it ideal for fine-tuning entries.

Key Features:
- 4-hour chart analysis for 1-2 months of data
- Support/resistance identification
- Momentum confirmation
- Candlestick patterns on 4H timeframe
- Intraday trend analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class FourHourEntryTiming:
    """Enhanced entry timing using 4-hour charts."""
    
    def __init__(self):
        """Initialize the 4-hour timing analyzer."""
        self.timeframe = "4h"
        
    def analyze_4h_entry(self, df_4h: pd.DataFrame, daily_signal: Dict) -> Dict:
        """
        Analyze 4-hour chart for optimal entry timing.
        
        Args:
            df_4h: DataFrame with 4-hour OHLCV data
            daily_signal: Signal from daily analysis including direction and levels
            
        Returns:
            Dictionary with timing recommendations
        """
        if df_4h is None or df_4h.empty:
            return {
                "timing": "unavailable",
                "recommendation": "Use daily chart signals",
                "confidence": 0.5
            }
            
        # Add technical indicators to 4H data
        df_4h = self._add_4h_indicators(df_4h)
        
        # Get the latest 4H candle
        latest = df_4h.iloc[-1]
        
        # Determine entry timing based on daily bias
        bias = daily_signal.get('market_outlook', 'neutral')
        
        timing_result = {
            "timeframe": "4-hour",
            "current_4h_close": float(latest['Close']),
            "4h_trend": self._determine_4h_trend(df_4h),
            "support_levels": self._find_support_levels(df_4h),
            "resistance_levels": self._find_resistance_levels(df_4h),
            "entry_zones": [],
            "timing": "wait",
            "confidence": 0.5
        }
        
        if bias == 'bullish':
            timing_result.update(self._analyze_bullish_entry(df_4h, daily_signal))
        elif bias == 'bearish':
            timing_result.update(self._analyze_bearish_entry(df_4h, daily_signal))
        else:
            timing_result["recommendation"] = "No clear bias - wait for direction"
            
        return timing_result
    
    def _add_4h_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to 4-hour data."""
        df = df.copy()
        
        # EMAs for 4H timeframe
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        
        # RSI
        df['RSI'] = self._calculate_rsi(df['Close'], period=14)
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # Stochastic
        df['Stoch_K'], df['Stoch_D'] = self._calculate_stochastic(df)
        
        # ATR for stops
        df['ATR'] = self._calculate_atr(df, period=14)
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    def _calculate_stochastic(self, df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic oscillator."""
        low_min = df['Low'].rolling(window=period).min()
        high_max = df['High'].rolling(window=period).max()
        
        k_percent = 100 * ((df['Close'] - low_min) / (high_max - low_min))
        d_percent = k_percent.rolling(window=3).mean()
        
        return k_percent.fillna(50), d_percent.fillna(50)
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high_low = df['High'] - df['Low']
        high_close = abs(df['High'] - df['Close'].shift(1))
        low_close = abs(df['Low'] - df['Close'].shift(1))
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        
        return true_range.rolling(window=period).mean()
    
    def _determine_4h_trend(self, df: pd.DataFrame) -> str:
        """Determine 4-hour trend direction."""
        latest = df.iloc[-1]
        
        # Check EMA alignment
        if latest['EMA_12'] > latest['EMA_26']:
            ema_trend = "bullish"
        else:
            ema_trend = "bearish"
            
        # Check price vs EMAs
        if latest['Close'] > latest['EMA_12'] and latest['Close'] > latest['EMA_26']:
            price_trend = "bullish"
        elif latest['Close'] < latest['EMA_12'] and latest['Close'] < latest['EMA_26']:
            price_trend = "bearish"
        else:
            price_trend = "neutral"
            
        # Combine signals
        if ema_trend == price_trend:
            return ema_trend
        else:
            return "neutral"
    
    def _find_support_levels(self, df: pd.DataFrame, lookback: int = 50) -> List[float]:
        """Find key support levels from 4H chart."""
        recent_df = df.tail(lookback)
        lows = recent_df['Low'].values
        
        # Find local minima
        support_levels = []
        for i in range(2, len(lows) - 2):
            if (lows[i] < lows[i-1] and lows[i] < lows[i-2] and 
                lows[i] < lows[i+1] and lows[i] < lows[i+2]):
                support_levels.append(float(lows[i]))
        
        # Remove duplicates and sort
        support_levels = sorted(list(set(support_levels)))
        
        # Keep only significant levels (at least 0.5% apart)
        filtered_supports = []
        for level in support_levels:
            if not filtered_supports or level > filtered_supports[-1] * 1.005:
                filtered_supports.append(level)
                
        return filtered_supports[-3:] if len(filtered_supports) > 3 else filtered_supports
    
    def _find_resistance_levels(self, df: pd.DataFrame, lookback: int = 50) -> List[float]:
        """Find key resistance levels from 4H chart."""
        recent_df = df.tail(lookback)
        highs = recent_df['High'].values
        
        # Find local maxima
        resistance_levels = []
        for i in range(2, len(highs) - 2):
            if (highs[i] > highs[i-1] and highs[i] > highs[i-2] and 
                highs[i] > highs[i+1] and highs[i] > highs[i+2]):
                resistance_levels.append(float(highs[i]))
        
        # Remove duplicates and sort
        resistance_levels = sorted(list(set(resistance_levels)), reverse=True)
        
        # Keep only significant levels (at least 0.5% apart)
        filtered_resistances = []
        for level in resistance_levels:
            if not filtered_resistances or level < filtered_resistances[-1] * 0.995:
                filtered_resistances.append(level)
                
        return filtered_resistances[:3]
    
    def _analyze_bullish_entry(self, df: pd.DataFrame, daily_signal: Dict) -> Dict:
        """Analyze 4H chart for bullish entry timing."""
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        entry_score = 0
        entry_reasons = []
        entry_zones = []
        
        # 1. Check if we're at/near support
        supports = self._find_support_levels(df)
        if supports:
            nearest_support = min(supports, key=lambda x: abs(x - latest['Close']))
            support_distance = (latest['Close'] - nearest_support) / latest['Close']
            
            if support_distance < 0.02:  # Within 2% of support
                entry_score += 3
                entry_reasons.append("At 4H support level")
                entry_zones.append({
                    "type": "support_bounce",
                    "level": nearest_support,
                    "action": "Buy near support"
                })
        
        # 2. Check 4H momentum
        if latest['RSI'] < 70 and latest['RSI'] > prev['RSI']:
            entry_score += 2
            entry_reasons.append("4H RSI momentum positive")
            
        if latest['RSI'] < 40:
            entry_score += 1
            entry_reasons.append("4H RSI oversold bounce opportunity")
        
        # 3. Check MACD on 4H
        if latest['MACD'] > latest['MACD_Signal'] and prev['MACD'] <= prev['MACD_Signal']:
            entry_score += 3
            entry_reasons.append("4H MACD bullish cross")
        elif latest['MACD'] > latest['MACD_Signal']:
            entry_score += 1
            entry_reasons.append("4H MACD bullish")
        
        # 4. Check Stochastic
        if latest['Stoch_K'] > latest['Stoch_D'] and latest['Stoch_K'] < 80:
            entry_score += 1
            entry_reasons.append("4H Stochastic bullish")
        
        # 5. Check 4H trend alignment
        if self._determine_4h_trend(df) == "bullish":
            entry_score += 2
            entry_reasons.append("4H trend aligned bullish")
        
        # 6. Pullback in uptrend
        if (latest['Close'] < latest['EMA_12'] and 
            latest['Low'] > latest['EMA_26'] and
            latest['EMA_12'] > latest['EMA_26']):
            entry_score += 2
            entry_reasons.append("Pullback to 4H EMA in uptrend")
            entry_zones.append({
                "type": "ema_pullback",
                "level": float(latest['EMA_12']),
                "action": "Buy at EMA pullback"
            })
        
        # Determine timing recommendation
        if entry_score >= 6:
            timing = "immediate"
            confidence = 0.85
            recommendation = "Strong 4H entry signal - enter on next 4H candle"
        elif entry_score >= 4:
            timing = "soon"
            confidence = 0.70
            recommendation = "Good 4H setup forming - prepare to enter"
        elif entry_score >= 2:
            timing = "wait"
            confidence = 0.55
            recommendation = "Wait for better 4H setup or pullback"
        else:
            timing = "wait"
            confidence = 0.40
            recommendation = "4H conditions not ideal - be patient"
        
        return {
            "timing": timing,
            "recommendation": recommendation,
            "confidence": confidence,
            "entry_score": entry_score,
            "entry_reasons": entry_reasons,
            "entry_zones": entry_zones,
            "suggested_stop": float(latest['Close'] - 1.5 * latest['ATR'])
        }
    
    def _analyze_bearish_entry(self, df: pd.DataFrame, daily_signal: Dict) -> Dict:
        """Analyze 4H chart for bearish entry timing."""
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        entry_score = 0
        entry_reasons = []
        entry_zones = []
        
        # 1. Check if we're at/near resistance
        resistances = self._find_resistance_levels(df)
        if resistances:
            nearest_resistance = min(resistances, key=lambda x: abs(x - latest['Close']))
            resistance_distance = (nearest_resistance - latest['Close']) / latest['Close']
            
            if resistance_distance < 0.02:  # Within 2% of resistance
                entry_score += 3
                entry_reasons.append("At 4H resistance level")
                entry_zones.append({
                    "type": "resistance_rejection",
                    "level": nearest_resistance,
                    "action": "Short near resistance"
                })
        
        # 2. Check 4H momentum
        if latest['RSI'] > 30 and latest['RSI'] < prev['RSI']:
            entry_score += 2
            entry_reasons.append("4H RSI momentum negative")
            
        if latest['RSI'] > 60:
            entry_score += 1
            entry_reasons.append("4H RSI overbought rejection opportunity")
        
        # 3. Check MACD on 4H
        if latest['MACD'] < latest['MACD_Signal'] and prev['MACD'] >= prev['MACD_Signal']:
            entry_score += 3
            entry_reasons.append("4H MACD bearish cross")
        elif latest['MACD'] < latest['MACD_Signal']:
            entry_score += 1
            entry_reasons.append("4H MACD bearish")
        
        # 4. Check Stochastic
        if latest['Stoch_K'] < latest['Stoch_D'] and latest['Stoch_K'] > 20:
            entry_score += 1
            entry_reasons.append("4H Stochastic bearish")
        
        # 5. Check 4H trend alignment
        if self._determine_4h_trend(df) == "bearish":
            entry_score += 2
            entry_reasons.append("4H trend aligned bearish")
        
        # 6. Rally in downtrend
        if (latest['Close'] > latest['EMA_12'] and 
            latest['High'] < latest['EMA_26'] and
            latest['EMA_12'] < latest['EMA_26']):
            entry_score += 2
            entry_reasons.append("Rally to 4H EMA in downtrend")
            entry_zones.append({
                "type": "ema_rally",
                "level": float(latest['EMA_12']),
                "action": "Short at EMA rally"
            })
        
        # Determine timing recommendation
        if entry_score >= 6:
            timing = "immediate"
            confidence = 0.85
            recommendation = "Strong 4H entry signal - enter on next 4H candle"
        elif entry_score >= 4:
            timing = "soon"
            confidence = 0.70
            recommendation = "Good 4H setup forming - prepare to enter"
        elif entry_score >= 2:
            timing = "wait"
            confidence = 0.55
            recommendation = "Wait for better 4H setup or rally"
        else:
            timing = "wait"
            confidence = 0.40
            recommendation = "4H conditions not ideal - be patient"
        
        return {
            "timing": timing,
            "recommendation": recommendation,
            "confidence": confidence,
            "entry_score": entry_score,
            "entry_reasons": entry_reasons,
            "entry_zones": entry_zones,
            "suggested_stop": float(latest['Close'] + 1.5 * latest['ATR'])
        }
    
    def get_4h_summary(self, df_4h: pd.DataFrame) -> Dict:
        """Get a summary of current 4H conditions."""
        if df_4h is None or df_4h.empty:
            return {"status": "No 4H data available"}
            
        latest = df_4h.iloc[-1]
        
        return {
            "last_update": df_4h.index[-1].strftime("%Y-%m-%d %H:%M"),
            "price": float(latest['Close']),
            "4h_trend": self._determine_4h_trend(df_4h),
            "rsi": float(latest.get('RSI', 50)),
            "macd_signal": "bullish" if latest.get('MACD', 0) > latest.get('MACD_Signal', 0) else "bearish",
            "volume_trend": "increasing" if latest['Volume'] > df_4h['Volume'].rolling(10).mean().iloc[-1] else "decreasing"
        } 