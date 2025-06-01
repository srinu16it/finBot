"""
Candlestick Pattern Detection for Options Entry/Exit Timing

This module provides candlestick pattern detection to complement the 
30-45 day options strategy by improving entry/exit timing.

Key principles:
- Candlesticks confirm, not contradict chart patterns
- Used for timing within existing strategy
- Focus on high-probability patterns only
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class CandlestickPatternDetector:
    """
    Detect candlestick patterns for entry/exit timing.
    
    Patterns are categorized by their use:
    - Entry confirmation: Hammer, Bullish Engulfing, Morning Star
    - Exit warning: Shooting Star, Bearish Engulfing, Evening Star
    - Continuation: Three White Soldiers, Three Black Crows
    """
    
    def __init__(self):
        """Initialize the detector with pattern thresholds."""
        # Body size thresholds
        self.doji_threshold = 0.1  # Body less than 10% of range
        self.long_body_threshold = 0.6  # Body more than 60% of range
        
    def detect_entry_patterns(self, df: pd.DataFrame, trend: str = "bullish") -> List[Dict]:
        """
        Detect candlestick patterns that confirm entry.
        
        Args:
            df: OHLCV DataFrame
            trend: Current trend from chart patterns ("bullish" or "bearish")
            
        Returns:
            List of entry confirmation patterns
        """
        patterns = []
        
        if trend == "bullish":
            # Look for bullish entry patterns
            patterns.extend(self._detect_hammer(df))
            patterns.extend(self._detect_bullish_engulfing(df))
            patterns.extend(self._detect_morning_star(df))
        else:  # bearish
            # Look for bearish entry patterns
            patterns.extend(self._detect_shooting_star(df))
            patterns.extend(self._detect_bearish_engulfing(df))
            patterns.extend(self._detect_evening_star(df))
            
        return patterns
    
    def detect_exit_warnings(self, df: pd.DataFrame, position: str = "long") -> List[Dict]:
        """
        Detect patterns that warn of potential reversal.
        
        Args:
            df: OHLCV DataFrame
            position: Current position type ("long" or "short")
            
        Returns:
            List of exit warning patterns
        """
        patterns = []
        
        if position == "long":
            # Look for bearish reversal patterns
            patterns.extend(self._detect_shooting_star(df))
            patterns.extend(self._detect_bearish_engulfing(df))
            patterns.extend(self._detect_evening_star(df))
        else:  # short position
            # Look for bullish reversal patterns
            patterns.extend(self._detect_hammer(df))
            patterns.extend(self._detect_bullish_engulfing(df))
            patterns.extend(self._detect_morning_star(df))
            
        return patterns
    
    def _calculate_body_metrics(self, open_price: float, close_price: float, 
                               high_price: float, low_price: float) -> Dict:
        """Calculate candlestick body metrics."""
        body = abs(close_price - open_price)
        range_hl = high_price - low_price
        
        if range_hl == 0:
            return None
            
        upper_shadow = high_price - max(open_price, close_price)
        lower_shadow = min(open_price, close_price) - low_price
        
        return {
            'body': body,
            'range': range_hl,
            'body_ratio': body / range_hl,
            'upper_shadow': upper_shadow,
            'lower_shadow': lower_shadow,
            'is_bullish': close_price > open_price
        }
    
    def _detect_hammer(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect Hammer pattern (bullish reversal at support).
        
        Criteria:
        - Small body at top of range
        - Long lower shadow (2x body minimum)
        - Little to no upper shadow
        - Appears after downtrend
        """
        patterns = []
        
        if len(df) < 5:
            return patterns
            
        for i in range(4, len(df)):
            curr = df.iloc[i]
            metrics = self._calculate_body_metrics(
                curr['Open'], curr['Close'], curr['High'], curr['Low']
            )
            
            if not metrics:
                continue
                
            # Check if in downtrend (simple check)
            prev_trend = df['Close'].iloc[i-4:i].mean() > df['Close'].iloc[i]
            
            # Hammer criteria
            if (metrics['body_ratio'] < 0.3 and  # Small body
                metrics['lower_shadow'] > 2 * metrics['body'] and  # Long lower shadow
                metrics['upper_shadow'] < metrics['body'] * 0.5 and  # Small upper shadow
                prev_trend):  # After downtrend
                
                patterns.append({
                    'pattern': 'hammer',
                    'type': 'bullish_reversal',
                    'index': i,
                    'date': df.index[i],
                    'confidence': 0.7,
                    'action': 'consider_entry',
                    'description': 'Hammer at support - potential reversal'
                })
                
        return patterns
    
    def _detect_bullish_engulfing(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect Bullish Engulfing pattern.
        
        Criteria:
        - Previous candle is bearish
        - Current candle is bullish
        - Current body engulfs previous body
        """
        patterns = []
        
        if len(df) < 2:
            return patterns
            
        for i in range(1, len(df)):
            curr = df.iloc[i]
            prev = df.iloc[i-1]
            
            # Check if current is bullish and previous is bearish
            if (curr['Close'] > curr['Open'] and 
                prev['Close'] < prev['Open']):
                
                # Check if current body engulfs previous
                if (curr['Open'] < prev['Close'] and 
                    curr['Close'] > prev['Open']):
                    
                    patterns.append({
                        'pattern': 'bullish_engulfing',
                        'type': 'bullish_reversal',
                        'index': i,
                        'date': df.index[i],
                        'confidence': 0.75,
                        'action': 'strong_entry_signal',
                        'description': 'Bullish engulfing - strong reversal signal'
                    })
                    
        return patterns
    
    def _detect_shooting_star(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect Shooting Star pattern (bearish reversal at resistance).
        
        Criteria:
        - Small body at bottom of range
        - Long upper shadow (2x body minimum)
        - Little to no lower shadow
        - Appears after uptrend
        """
        patterns = []
        
        if len(df) < 5:
            return patterns
            
        for i in range(4, len(df)):
            curr = df.iloc[i]
            metrics = self._calculate_body_metrics(
                curr['Open'], curr['Close'], curr['High'], curr['Low']
            )
            
            if not metrics:
                continue
                
            # Check if in uptrend
            prev_trend = df['Close'].iloc[i-4:i].mean() < df['Close'].iloc[i]
            
            # Shooting star criteria
            if (metrics['body_ratio'] < 0.3 and  # Small body
                metrics['upper_shadow'] > 2 * metrics['body'] and  # Long upper shadow
                metrics['lower_shadow'] < metrics['body'] * 0.5 and  # Small lower shadow
                prev_trend):  # After uptrend
                
                patterns.append({
                    'pattern': 'shooting_star',
                    'type': 'bearish_reversal',
                    'index': i,
                    'date': df.index[i],
                    'confidence': 0.7,
                    'action': 'consider_exit',
                    'description': 'Shooting star at resistance - potential reversal'
                })
                
        return patterns
    
    def _detect_bearish_engulfing(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect Bearish Engulfing pattern.
        
        Criteria:
        - Previous candle is bullish
        - Current candle is bearish
        - Current body engulfs previous body
        """
        patterns = []
        
        if len(df) < 2:
            return patterns
            
        for i in range(1, len(df)):
            curr = df.iloc[i]
            prev = df.iloc[i-1]
            
            # Check if current is bearish and previous is bullish
            if (curr['Close'] < curr['Open'] and 
                prev['Close'] > prev['Open']):
                
                # Check if current body engulfs previous
                if (curr['Open'] > prev['Close'] and 
                    curr['Close'] < prev['Open']):
                    
                    patterns.append({
                        'pattern': 'bearish_engulfing',
                        'type': 'bearish_reversal',
                        'index': i,
                        'date': df.index[i],
                        'confidence': 0.75,
                        'action': 'strong_exit_signal',
                        'description': 'Bearish engulfing - strong reversal signal'
                    })
                    
        return patterns
    
    def _detect_morning_star(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect Morning Star pattern (3-candle bullish reversal).
        
        Criteria:
        - First: Large bearish candle
        - Second: Small body (star)
        - Third: Large bullish candle
        """
        patterns = []
        
        if len(df) < 3:
            return patterns
            
        for i in range(2, len(df)):
            first = df.iloc[i-2]
            second = df.iloc[i-1]
            third = df.iloc[i]
            
            # Calculate metrics
            first_body = abs(first['Close'] - first['Open'])
            second_body = abs(second['Close'] - second['Open'])
            third_body = abs(third['Close'] - third['Open'])
            
            # Morning star criteria
            if (first['Close'] < first['Open'] and  # First is bearish
                third['Close'] > third['Open'] and  # Third is bullish
                second_body < first_body * 0.3 and  # Second has small body
                second_body < third_body * 0.3 and  # Second smaller than third
                third['Close'] > (first['Open'] + first['Close']) / 2):  # Third closes above first's midpoint
                
                patterns.append({
                    'pattern': 'morning_star',
                    'type': 'bullish_reversal',
                    'index': i,
                    'date': df.index[i],
                    'confidence': 0.8,
                    'action': 'strong_entry_signal',
                    'description': 'Morning star - high probability reversal'
                })
                
        return patterns
    
    def _detect_evening_star(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect Evening Star pattern (3-candle bearish reversal).
        
        Criteria:
        - First: Large bullish candle
        - Second: Small body (star)
        - Third: Large bearish candle
        """
        patterns = []
        
        if len(df) < 3:
            return patterns
            
        for i in range(2, len(df)):
            first = df.iloc[i-2]
            second = df.iloc[i-1]
            third = df.iloc[i]
            
            # Calculate metrics
            first_body = abs(first['Close'] - first['Open'])
            second_body = abs(second['Close'] - second['Open'])
            third_body = abs(third['Close'] - third['Open'])
            
            # Evening star criteria
            if (first['Close'] > first['Open'] and  # First is bullish
                third['Close'] < third['Open'] and  # Third is bearish
                second_body < first_body * 0.3 and  # Second has small body
                second_body < third_body * 0.3 and  # Second smaller than third
                third['Close'] < (first['Open'] + first['Close']) / 2):  # Third closes below first's midpoint
                
                patterns.append({
                    'pattern': 'evening_star',
                    'type': 'bearish_reversal',
                    'index': i,
                    'date': df.index[i],
                    'confidence': 0.8,
                    'action': 'strong_exit_signal',
                    'description': 'Evening star - high probability reversal'
                })
                
        return patterns
    
    def get_entry_timing(self, df: pd.DataFrame, chart_pattern: str) -> Optional[Dict]:
        """
        Get specific entry timing based on chart pattern and candlesticks.
        
        Args:
            df: OHLCV DataFrame
            chart_pattern: The detected chart pattern type
            
        Returns:
            Entry timing recommendation
        """
        # Map chart patterns to trend
        bullish_patterns = ['double_bottom', 'inverse_head_and_shoulders', 'ascending_triangle']
        bearish_patterns = ['double_top', 'head_and_shoulders', 'descending_triangle']
        
        if chart_pattern in bullish_patterns:
            trend = "bullish"
        elif chart_pattern in bearish_patterns:
            trend = "bearish"
        else:
            trend = "neutral"
            
        # Get recent candlestick patterns
        recent_patterns = self.detect_entry_patterns(df.tail(10), trend)
        
        if recent_patterns:
            latest = recent_patterns[-1]
            return {
                'timing': 'confirmed',
                'pattern': latest['pattern'],
                'action': latest['action'],
                'confidence': latest['confidence'],
                'description': f"Entry confirmed by {latest['pattern']} pattern"
            }
        else:
            return {
                'timing': 'wait',
                'pattern': None,
                'action': 'wait_for_confirmation',
                'confidence': 0.5,
                'description': f"Wait for {trend} candlestick confirmation"
            } 