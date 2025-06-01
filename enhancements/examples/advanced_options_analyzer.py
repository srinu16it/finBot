#!/usr/bin/env python3
"""
Advanced Options Analysis with Professional Trading Workflow.

This implements a sophisticated options trading system with:
- 6 months daily data with weekly resampling
- Pattern detection with trend confirmation
- HV/IV analysis for strategy selection
- ATR-based risk management
- Proper entry/exit conditions

Usage:
    ./venv_test/bin/python enhancements/examples/advanced_options_analyzer.py
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from scipy.signal import argrelextrema
import logging
import yfinance as yf

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from enhancements.data_providers.yahoo_provider import YahooProvider
from enhancements.data_access.cache import CacheManager
from enhancements.patterns.confidence import PatternConfidenceEngine
from enhanced_pattern_analyzer import EnhancedPatternDetector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AdvancedOptionsAnalyzer:
    """Professional options analysis with strict entry conditions."""
    
    def __init__(self, cache_manager=None):
        """Initialize the analyzer."""
        self.cache_manager = cache_manager or CacheManager()
        self.confidence_engine = PatternConfidenceEngine()
        
    def fetch_and_prepare_data(self, symbol: str):
        """
        Fetch 6 months of daily data and prepare weekly resampling.
        
        Returns:
            daily_df: Daily OHLCV data
            weekly_df: Weekly resampled data
        """
        logger.info(f"Fetching 6 months of daily data for {symbol}")
        
        provider = YahooProvider(self.cache_manager)
        daily_df = provider.get_ohlcv(symbol, period="6mo", interval="1d")
        
        if daily_df is None or daily_df.empty:
            raise ValueError(f"Failed to fetch data for {symbol}")
        
        # Set Date as index if needed
        if 'Date' in daily_df.columns:
            daily_df.set_index('Date', inplace=True)
        
        # Resample to weekly
        weekly_df = daily_df.resample('W').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        })
        
        logger.info(f"Daily data: {len(daily_df)} bars, Weekly data: {len(weekly_df)} bars")
        
        return daily_df, weekly_df
    
    def calculate_advanced_indicators(self, daily_df: pd.DataFrame, weekly_df: pd.DataFrame):
        """
        Calculate all required indicators including ADX and ATR.
        """
        # Daily indicators
        daily_df['EMA_9'] = daily_df['Close'].ewm(span=9, adjust=False).mean()
        daily_df['EMA_21'] = daily_df['Close'].ewm(span=21, adjust=False).mean()
        
        # ATR calculation (14 period)
        high_low = daily_df['High'] - daily_df['Low']
        high_close = abs(daily_df['High'] - daily_df['Close'].shift(1))
        low_close = abs(daily_df['Low'] - daily_df['Close'].shift(1))
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        daily_df['ATR'] = tr.rolling(window=14).mean()
        
        # ADX calculation
        daily_df['ADX'] = self.calculate_adx(daily_df)
        
        # RSI
        daily_df['RSI'] = self.calculate_rsi(daily_df['Close'])
        
        # Weekly indicators
        weekly_df['SMA_20'] = weekly_df['Close'].rolling(window=20).mean()
        weekly_df['RSI'] = self.calculate_rsi(weekly_df['Close'])
        
        # Historical Volatility (60-day and 30-day)
        daily_returns = daily_df['Close'].pct_change()
        daily_df['HV_60'] = daily_returns.rolling(window=60).std() * np.sqrt(252) * 100
        daily_df['HV_30'] = daily_returns.rolling(window=30).std() * np.sqrt(252) * 100
        
        return daily_df, weekly_df
    
    def calculate_adx(self, df: pd.DataFrame, period: int = 14):
        """Calculate Average Directional Index (ADX) using Wilder's method."""
        # Ensure we have enough data
        if len(df) < period * 2:
            return pd.Series(0, index=df.index)
        
        # Calculate directional movement
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        # Calculate +DM and -DM
        up_move = high.diff()
        down_move = -low.diff()
        
        plus_dm = pd.Series(0.0, index=df.index)
        minus_dm = pd.Series(0.0, index=df.index)
        
        plus_dm[(up_move > down_move) & (up_move > 0)] = up_move[(up_move > down_move) & (up_move > 0)]
        minus_dm[(down_move > up_move) & (down_move > 0)] = down_move[(down_move > up_move) & (down_move > 0)]
        
        # Calculate True Range
        hl = high - low
        hc = abs(high - close.shift(1))
        lc = abs(low - close.shift(1))
        
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        
        # Apply Wilder's smoothing (EMA with alpha = 1/period)
        # First value is SMA, then EMA
        atr = pd.Series(0.0, index=df.index)
        smoothed_plus_dm = pd.Series(0.0, index=df.index)
        smoothed_minus_dm = pd.Series(0.0, index=df.index)
        
        # Initialize with SMA
        atr.iloc[period-1] = tr.iloc[:period].mean()
        smoothed_plus_dm.iloc[period-1] = plus_dm.iloc[:period].mean()
        smoothed_minus_dm.iloc[period-1] = minus_dm.iloc[:period].mean()
        
        # Apply Wilder's smoothing
        for i in range(period, len(df)):
            atr.iloc[i] = (atr.iloc[i-1] * (period - 1) + tr.iloc[i]) / period
            smoothed_plus_dm.iloc[i] = (smoothed_plus_dm.iloc[i-1] * (period - 1) + plus_dm.iloc[i]) / period
            smoothed_minus_dm.iloc[i] = (smoothed_minus_dm.iloc[i-1] * (period - 1) + minus_dm.iloc[i]) / period
        
        # Calculate +DI and -DI
        plus_di = 100 * smoothed_plus_dm / atr
        minus_di = 100 * smoothed_minus_dm / atr
        
        # Replace any inf/nan with 0
        plus_di = plus_di.replace([np.inf, -np.inf], 0).fillna(0)
        minus_di = minus_di.replace([np.inf, -np.inf], 0).fillna(0)
        
        # Calculate DX
        di_diff = abs(plus_di - minus_di)
        di_sum = plus_di + minus_di
        dx = pd.Series(0.0, index=df.index)
        dx[di_sum != 0] = 100 * di_diff[di_sum != 0] / di_sum[di_sum != 0]
        
        # Calculate ADX (smoothed DX)
        adx = pd.Series(0.0, index=df.index)
        
        # First ADX value is average of first period DX values
        first_adx_idx = period * 2 - 1
        if first_adx_idx < len(df):
            adx.iloc[first_adx_idx] = dx.iloc[period:first_adx_idx+1].mean()
            
            # Apply Wilder's smoothing to ADX
            for i in range(first_adx_idx + 1, len(df)):
                adx.iloc[i] = (adx.iloc[i-1] * (period - 1) + dx.iloc[i]) / period
        
        return adx
    
    def calculate_rsi(self, data, period=14):
        """Calculate RSI with proper handling."""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss.replace(0, 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.fillna(50)
    
    def detect_patterns(self, df: pd.DataFrame):
        """Detect patterns on daily data using the enhanced pattern detector."""
        # Use the enhanced pattern detector for proper pattern detection
        pattern_detector = EnhancedPatternDetector(self.cache_manager)
        patterns = pattern_detector.detect_all_patterns(df)
        
        # If no patterns detected, check for simple trend
        if not patterns and len(df) > 20:
            sma_20 = df['Close'].rolling(20).mean().iloc[-1]
            current_price = df['Close'].iloc[-1]
            
            if current_price > sma_20 * 1.02:  # 2% above SMA
                patterns.append({
                    'pattern': 'uptrend',
                    'type': 'bullish',
                    'confidence_score': 0.6,
                    'historical_win_rate': 0.5
                })
            elif current_price < sma_20 * 0.98:  # 2% below SMA
                patterns.append({
                    'pattern': 'downtrend', 
                    'type': 'bearish',
                    'confidence_score': 0.6,
                    'historical_win_rate': 0.5
                })
        
        return patterns
    
    def get_iv_from_options(self, symbol: str):
        """
        Fetch implied volatility from options chain.
        Returns average IV of ATM options 30-45 days out.
        """
        try:
            ticker = yf.Ticker(symbol)
            
            # Get available expiration dates
            expirations = ticker.options
            if not expirations:
                return None
            
            # Find expiration 30-45 days out
            target_date = datetime.now() + timedelta(days=37)
            best_expiry = min(expirations, 
                            key=lambda x: abs((datetime.strptime(x, '%Y-%m-%d') - target_date).days))
            
            # Get options chain
            opt_chain = ticker.option_chain(best_expiry)
            calls = opt_chain.calls
            puts = opt_chain.puts
            
            # Get current price
            current_price = ticker.history(period="1d")['Close'].iloc[-1]
            
            # Find ATM options
            calls['distance'] = abs(calls['strike'] - current_price)
            puts['distance'] = abs(puts['strike'] - current_price)
            
            atm_call = calls.nsmallest(1, 'distance')
            atm_put = puts.nsmallest(1, 'distance')
            
            # Average IV of ATM call and put
            if not atm_call.empty and not atm_put.empty:
                avg_iv = (atm_call['impliedVolatility'].iloc[0] + 
                         atm_put['impliedVolatility'].iloc[0]) / 2 * 100
                return avg_iv
            
        except Exception as e:
            logger.warning(f"Failed to fetch IV for {symbol}: {e}")
        
        return None
    
    def determine_options_strategy(self, pattern_bias: str, daily_df: pd.DataFrame, 
                                 weekly_df: pd.DataFrame, iv: float = None):
        """
        Determine options strategy based on conditions.
        
        Returns:
            strategy: Recommended strategy
            entry_conditions_met: Boolean
            details: Dict with analysis details
        """
        details = {}
        
        # Get latest values
        latest_daily = daily_df.iloc[-1]
        latest_weekly = weekly_df.iloc[-1]
        
        # Check ADX condition
        adx_value = latest_daily['ADX']
        # Handle NaN or very small values
        if pd.isna(adx_value) or adx_value < 1:
            adx_value = 0
            
        adx_condition = adx_value >= 20
        details['ADX'] = adx_value
        details['ADX_condition_met'] = adx_condition
        
        # Check weekly trend condition
        weekly_trend_condition = False
        if pattern_bias == 'bullish':
            weekly_trend_condition = latest_weekly['Close'] > latest_weekly['SMA_20']
        elif pattern_bias == 'bearish':
            weekly_trend_condition = latest_weekly['Close'] < latest_weekly['SMA_20']
        
        details['weekly_close'] = latest_weekly['Close']
        details['weekly_SMA_20'] = latest_weekly['SMA_20']
        details['weekly_trend_condition_met'] = weekly_trend_condition
        
        # All entry conditions
        entry_conditions_met = adx_condition and weekly_trend_condition
        
        # Determine strategy based on HV/IV
        hv_60 = latest_daily['HV_60']
        hv_30 = latest_daily['HV_30']
        
        details['HV_60'] = hv_60
        details['HV_30'] = hv_30
        details['IV'] = iv
        
        if iv is None:
            # Default to HV if IV not available
            iv = hv_30
        
        strategy = None
        if entry_conditions_met:
            if pattern_bias == 'bullish':
                if iv > hv_60 * 1.2:  # IV significantly higher than HV
                    strategy = {
                        'type': 'Bull Put Spread',
                        'rationale': 'High IV - Sell premium',
                        'description': 'Sell put spread to collect premium'
                    }
                else:
                    strategy = {
                        'type': 'Bull Call Spread',
                        'rationale': 'Normal/Low IV - Buy debit spread',
                        'description': 'Buy call spread for directional play'
                    }
            elif pattern_bias == 'bearish':
                if iv > hv_60 * 1.2:  # IV significantly higher than HV
                    strategy = {
                        'type': 'Bear Call Spread',
                        'rationale': 'High IV - Sell premium',
                        'description': 'Sell call spread to collect premium'
                    }
                else:
                    strategy = {
                        'type': 'Bear Put Spread',
                        'rationale': 'Normal/Low IV - Buy debit spread',
                        'description': 'Buy put spread for directional play'
                    }
        
        # Calculate stop loss
        atr = latest_daily['ATR']
        stop_loss = 1.5 * atr
        details['ATR'] = atr
        details['stop_loss_points'] = stop_loss
        details['stop_loss_price'] = latest_daily['Close'] - stop_loss if pattern_bias == 'bullish' else latest_daily['Close'] + stop_loss
        
        return strategy, entry_conditions_met, details
    
    def analyze_symbol(self, symbol: str):
        """
        Complete analysis workflow for a symbol.
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Analyzing {symbol} - Professional Options Workflow")
        logger.info(f"{'='*60}")
        
        try:
            # Step 1: Fetch and prepare data
            daily_df, weekly_df = self.fetch_and_prepare_data(symbol)
            
            # Step 2: Calculate indicators
            daily_df, weekly_df = self.calculate_advanced_indicators(daily_df, weekly_df)
            
            # Step 3: Detect patterns
            patterns = self.detect_patterns(daily_df)
            
            # Determine pattern bias
            if patterns:
                bullish_patterns = sum(1 for p in patterns if p['type'] == 'bullish')
                bearish_patterns = sum(1 for p in patterns if p['type'] == 'bearish')
                
                if bullish_patterns > bearish_patterns:
                    pattern_bias = 'bullish'
                elif bearish_patterns > bullish_patterns:
                    pattern_bias = 'bearish'
                else:
                    pattern_bias = 'neutral'
            else:
                pattern_bias = 'neutral'
            
            # Step 4: Get IV from options
            iv = self.get_iv_from_options(symbol)
            
            # Step 5: Determine strategy
            strategy, conditions_met, details = self.determine_options_strategy(
                pattern_bias, daily_df, weekly_df, iv
            )
            
            # Create report
            report = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'pattern_bias': pattern_bias,
                'patterns': patterns,
                'entry_conditions_met': conditions_met,
                'strategy': strategy,
                'analysis_details': details,
                'current_price': daily_df['Close'].iloc[-1],
                'recommendations': []
            }
            
            # Add recommendations
            if conditions_met and strategy:
                report['recommendations'].append({
                    'action': 'ENTER',
                    'strategy': strategy['type'],
                    'rationale': strategy['rationale'],
                    'stop_loss': details['stop_loss_price'],
                    'position_size': 'Use 1-2% portfolio risk',
                    'expiration': '30-45 days out'
                })
            else:
                report['recommendations'].append({
                    'action': 'NO TRADE',
                    'reason': 'Entry conditions not met',
                    'missing_conditions': []
                })
                
                if not details.get('ADX_condition_met'):
                    adx_val = details['ADX']
                    adx_display = f"{adx_val:.1f}" if adx_val > 0 else "Insufficient data"
                    report['recommendations'][0]['missing_conditions'].append(
                        f"ADX too low: {adx_display} < 20"
                    )
                if not details.get('weekly_trend_condition_met'):
                    report['recommendations'][0]['missing_conditions'].append(
                        f"Weekly trend not aligned with pattern bias"
                    )
            
            return report
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {str(e)}")
            return None


def main():
    """Run advanced analysis on sample symbols."""
    analyzer = AdvancedOptionsAnalyzer()
    
    symbols = ["AAPL", "TSLA", "SPY"]
    
    for symbol in symbols:
        report = analyzer.analyze_symbol(symbol)
        
        if report:
            print(f"\n📊 Analysis Report for {symbol}")
            print(f"Pattern Bias: {report['pattern_bias'].upper()}")
            print(f"Current Price: ${report['current_price']:.2f}")
            print(f"Entry Conditions Met: {'✅ YES' if report['entry_conditions_met'] else '❌ NO'}")
            
            print("\n📈 Analysis Details:")
            details = report['analysis_details']
            adx_val = details['ADX']
            adx_display = f"{adx_val:.1f}" if adx_val > 0 else "Insufficient data"
            print(f"ADX: {adx_display} {'✅' if details['ADX_condition_met'] else '❌ (Need ≥ 20)'}")
            print(f"Weekly Close: ${details['weekly_close']:.2f}")
            print(f"Weekly SMA(20): ${details['weekly_SMA_20']:.2f}")
            print(f"Weekly Trend OK: {'✅' if details['weekly_trend_condition_met'] else '❌'}")
            
            print(f"\n📊 Volatility Analysis:")
            print(f"HV(60): {details['HV_60']:.1f}%")
            print(f"HV(30): {details['HV_30']:.1f}%")
            if details['IV']:
                print(f"IV: {details['IV']:.1f}%")
            
            print(f"\n🎯 Risk Management:")
            print(f"ATR(14): ${details['ATR']:.2f}")
            print(f"Stop Loss: ${details['stop_loss_price']:.2f} ({details['stop_loss_points']:.2f} points)")
            
            print(f"\n💡 Recommendations:")
            for rec in report['recommendations']:
                print(f"Action: {rec['action']}")
                if rec['action'] == 'ENTER':
                    print(f"Strategy: {rec['strategy']}")
                    print(f"Rationale: {rec['rationale']}")
                    print(f"Stop Loss: ${rec['stop_loss']:.2f}")
                    print(f"Position Size: {rec['position_size']}")
                    print(f"Expiration: {rec['expiration']}")
                else:
                    print(f"Reason: {rec['reason']}")
                    if rec['missing_conditions']:
                        print("Missing Conditions:")
                        for condition in rec['missing_conditions']:
                            print(f"  - {condition}")


if __name__ == "__main__":
    main() 