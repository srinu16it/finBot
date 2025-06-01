import numpy as np
import pandas as pd

# Candlestick pattern detection module
# Based on Japanese candlestick pattern recognition

def detect_patterns(df):
    """Detects candlestick patterns in a DataFrame with OHLC data
    Returns a dictionary with pattern types and signals"""
    
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Create the bullish/bearish labels for candles
    df['bullish'] = df['Close'] > df['Open']
    df['bearish'] = df['Close'] < df['Open']
    df['body_size'] = abs(df['Close'] - df['Open'])
    df['shadow_upper'] = df['High'] - df[['Open', 'Close']].max(axis=1)
    df['shadow_lower'] = df[['Open', 'Close']].min(axis=1) - df['Low']
    df['range'] = df['High'] - df['Low']
    
    # Calculate typical range for relativity
    df['typical_range'] = df['range'].rolling(window=14).mean()
    
    # Initialize pattern storage
    patterns = {
        "Neutral": "No dominant pattern",
        "Single Candle": "No significant single candle patterns",
        "Double Candle": "No significant double candle patterns",
        "Triple Candle": "No significant triple candle patterns",
        "Confirmation": "No confirmation patterns"
    }
    
    # Only analyze if we have enough data
    if len(df) < 5:
        return patterns
    
    # Get recent data for pattern detection
    recent = df.tail(5).copy()
    
    # SINGLE CANDLE PATTERNS
    
    # Doji detection (very small body)
    recent['doji'] = recent['body_size'] < (0.1 * recent['range'])
    
    # Hammer detection (small body, long lower shadow, small upper shadow)
    recent['hammer'] = (
        (recent['bullish']) &
        (recent['body_size'] < 0.3 * recent['range']) &
        (recent['shadow_lower'] > 2 * recent['body_size']) &
        (recent['shadow_upper'] < 0.1 * recent['range'])
    )
    
    # Inverted Hammer detection
    recent['inverted_hammer'] = (
        (recent['bullish']) &
        (recent['body_size'] < 0.3 * recent['range']) &
        (recent['shadow_upper'] > 2 * recent['body_size']) &
        (recent['shadow_lower'] < 0.1 * recent['range'])
    )
    
    # Hanging Man detection
    recent['hanging_man'] = (
        (recent['bearish']) &
        (recent['body_size'] < 0.3 * recent['range']) &
        (recent['shadow_lower'] > 2 * recent['body_size']) &
        (recent['shadow_upper'] < 0.1 * recent['range'])
    )
    
    # Shooting Star detection
    recent['shooting_star'] = (
        (recent['bearish']) &
        (recent['body_size'] < 0.3 * recent['range']) &
        (recent['shadow_upper'] > 2 * recent['body_size']) &
        (recent['shadow_lower'] < 0.1 * recent['range'])
    )
    
    # DOUBLE CANDLE PATTERNS
    
    # Bullish Engulfing
    bullish_engulfing = (
        (recent['bearish'].shift(1)) &
        (recent['bullish']) &
        (recent['Open'] < recent['Close'].shift(1)) &
        (recent['Close'] > recent['Open'].shift(1))
    )
    
    # Bearish Engulfing
    bearish_engulfing = (
        (recent['bullish'].shift(1)) &
        (recent['bearish']) &
        (recent['Open'] > recent['Close'].shift(1)) &
        (recent['Close'] < recent['Open'].shift(1))
    )
    
    # Bullish Harami
    bullish_harami = (
        (recent['bearish'].shift(1)) &
        (recent['bullish']) &
        (recent['Open'] > recent['Close'].shift(1)) &
        (recent['Close'] < recent['Open'].shift(1)) &
        (recent['body_size'] < recent['body_size'].shift(1))
    )
    
    # Bearish Harami
    bearish_harami = (
        (recent['bullish'].shift(1)) &
        (recent['bearish']) &
        (recent['Open'] < recent['Close'].shift(1)) &
        (recent['Close'] > recent['Open'].shift(1)) &
        (recent['body_size'] < recent['body_size'].shift(1))
    )
    
    # TRIPLE CANDLE PATTERNS
    
    # Morning Star
    morning_star = (
        (recent['bearish'].shift(2)) &
        (recent['body_size'].shift(1) < 0.3 * recent['body_size'].shift(2)) &
        (recent['bullish']) &
        (recent['Close'] > (recent['Open'].shift(2) + recent['Close'].shift(2)) / 2)
    )
    
    # Evening Star
    evening_star = (
        (recent['bullish'].shift(2)) &
        (recent['body_size'].shift(1) < 0.3 * recent['body_size'].shift(2)) &
        (recent['bearish']) &
        (recent['Close'] < (recent['Open'].shift(2) + recent['Close'].shift(2)) / 2)
    )
    
    # Three White Soldiers
    three_white_soldiers = (
        (recent['bullish']) &
        (recent['bullish'].shift(1)) &
        (recent['bullish'].shift(2)) &
        (recent['Close'] > recent['Close'].shift(1)) &
        (recent['Close'].shift(1) > recent['Close'].shift(2)) &
        (recent['Open'] > recent['Open'].shift(1)) &
        (recent['Open'].shift(1) > recent['Open'].shift(2))
    )
    
    # Three Black Crows
    three_black_crows = (
        (recent['bearish']) &
        (recent['bearish'].shift(1)) &
        (recent['bearish'].shift(2)) &
        (recent['Close'] < recent['Close'].shift(1)) &
        (recent['Close'].shift(1) < recent['Close'].shift(2)) &
        (recent['Open'] < recent['Open'].shift(1)) &
        (recent['Open'].shift(1) < recent['Open'].shift(2))
    )
    
    # Extract the pattern signals from the most recent row
    latest = recent.iloc[-1]
    
    # Check for patterns in priority order (most significant first)
    
    # Triple candle patterns (strongest)
    if latest.get('morning_star', False) or morning_star.iloc[-1]:
        patterns["Triple Candle"] = "Morning Star → Strongly Bullish"
    elif latest.get('evening_star', False) or evening_star.iloc[-1]:
        patterns["Triple Candle"] = "Evening Star → Strongly Bearish"
    elif three_white_soldiers.iloc[-1]:
        patterns["Triple Candle"] = "Three White Soldiers → Strongly Bullish"
    elif three_black_crows.iloc[-1]:
        patterns["Triple Candle"] = "Three Black Crows → Strongly Bearish"
    
    # Double candle patterns
    if bullish_engulfing.iloc[-1]:
        patterns["Double Candle"] = "Bullish Engulfing → Bullish"
    elif bearish_engulfing.iloc[-1]:
        patterns["Double Candle"] = "Bearish Engulfing → Bearish"
    elif bullish_harami.iloc[-1]:
        patterns["Double Candle"] = "Bullish Harami → Moderately Bullish"
    elif bearish_harami.iloc[-1]:
        patterns["Double Candle"] = "Bearish Harami → Moderately Bearish"
    
    # Single candle patterns
    if latest['hammer']:
        patterns["Single Candle"] = "Hammer → Bullish"
    elif latest['inverted_hammer']:
        patterns["Single Candle"] = "Inverted Hammer → Potentially Bullish"
    elif latest['hanging_man']:
        patterns["Single Candle"] = "Hanging Man → Bearish"
    elif latest['shooting_star']:
        patterns["Single Candle"] = "Shooting Star → Bearish"
    elif latest['doji']:
        patterns["Neutral"] = "Doji → Indecision (potential reversal)"
    
    return patterns

def get_pattern_description(pattern_signal):
    """Returns a description and trading implications for a pattern"""
    
    descriptions = {
        # Neutral patterns
        "Doji → Indecision (potential reversal)": 
            "A Doji forms when open and close are virtually equal, showing market indecision. "
            "Watch for confirmation before making a trade. Consider straddle options strategies.",
        
        # Single candle patterns
        "Hammer → Bullish": 
            "A hammer has a small body at the top and a long lower shadow, signaling potential trend reversal. "
            "Consider long calls or bull call spreads.",
            
        "Inverted Hammer → Potentially Bullish":
            "An inverted hammer has a small body at the bottom and a long upper shadow. "
            "Wait for confirmation before going long. Consider bull call spreads with protection.",
            
        "Hanging Man → Bearish":
            "A hanging man has a small body at the top and a long lower shadow, appearing in uptrends. "
            "Consider buying puts or bear put spreads.",
            
        "Shooting Star → Bearish":
            "A shooting star has a small body at the bottom and a long upper shadow, signaling rejection of higher prices. "
            "Consider buying puts or establishing bear call spreads.",
        
        # Double candle patterns
        "Bullish Engulfing → Bullish":
            "A bullish engulfing pattern occurs when a larger bullish candle completely engulfs the previous bearish candle. "
            "Strong reversal signal. Consider buying calls or bull call spreads.",
            
        "Bearish Engulfing → Bearish":
            "A bearish engulfing pattern occurs when a larger bearish candle completely engulfs the previous bullish candle. "
            "Strong reversal signal. Consider buying puts or bear put spreads.",
            
        "Bullish Harami → Moderately Bullish":
            "A bullish harami shows a small bullish candle contained within the prior larger bearish candle. "
            "Shows momentum loss in the downtrend. Consider bull call spreads.",
            
        "Bearish Harami → Moderately Bearish":
            "A bearish harami shows a small bearish candle contained within the prior larger bullish candle. "
            "Shows momentum loss in the uptrend. Consider bear put spreads.",
        
        # Triple candle patterns
        "Morning Star → Strongly Bullish":
            "A morning star is a three-candle bottom reversal pattern: big bearish, small indecision, big bullish. "
            "Strong bottom reversal signal. Consider long calls or bull call spreads.",
            
        "Evening Star → Strongly Bearish":
            "An evening star is a three-candle top reversal pattern: big bullish, small indecision, big bearish. "
            "Strong top reversal signal. Consider long puts or bear put spreads.",
            
        "Three White Soldiers → Strongly Bullish":
            "Three consecutive bullish candles, each opening within the previous candle's body and closing higher. "
            "Strong uptrend confirmation. Consider long calls or long call spreads.",
            
        "Three Black Crows → Strongly Bearish":
            "Three consecutive bearish candles, each opening within the previous candle's body and closing lower. "
            "Strong downtrend confirmation. Consider long puts or bear put spreads.",
    }
    
    # Return the description if found, otherwise a generic message
    return descriptions.get(pattern_signal, "No specific description available for this pattern.")

def get_options_recommendation(pattern_signal, current_price, volatility):
    """Generates specific options trading recommendations based on the pattern"""
    
    bullish_patterns = ["Hammer", "Bullish Engulfing", "Bullish Harami", 
                        "Morning Star", "Three White Soldiers"]
    
    bearish_patterns = ["Hanging Man", "Shooting Star", "Bearish Engulfing", 
                       "Bearish Harami", "Evening Star", "Three Black Crows"]
    
    neutral_patterns = ["Doji"]
    
    # Extract pattern name from the signal
    pattern_name = pattern_signal.split("→")[0].strip()
    
    # Default recommendation
    recommendation = {
        "strategy": "Wait for confirmation",
        "options_type": "None",
        "strike_price": current_price,
        "expiration": "N/A",
        "risk_level": "Low"
    }
    
    # Determine if pattern is in our lists
    is_bullish = any(bull_pat in pattern_name for bull_pat in bullish_patterns)
    is_bearish = any(bear_pat in pattern_name for bear_pat in bearish_patterns)
    is_neutral = any(neut_pat in pattern_name for neut_pat in neutral_patterns)
    
    # Generate recommendations based on pattern type
    if is_bullish:
        # Strong bullish patterns
        if "Morning Star" in pattern_name or "Three White Soldiers" in pattern_name:
            recommendation = {
                "strategy": "Long Call",
                "options_type": "Call",
                "strike_price": round(current_price * 1.01, 2),  # Slightly OTM
                "expiration": "30-45 days",
                "risk_level": "Medium-High"
            }
        # Moderate bullish patterns
        elif "Bullish Engulfing" in pattern_name:
            recommendation = {
                "strategy": "Bull Call Spread",
                "options_type": "Call Spread",
                "strike_price": f"{round(current_price, 2)} / {round(current_price * 1.05, 2)}",
                "expiration": "30-45 days",
                "risk_level": "Medium"
            }
        # Weaker bullish patterns
        else:
            recommendation = {
                "strategy": "Bull Call Spread (Conservative)",
                "options_type": "Call Spread",
                "strike_price": f"{round(current_price * 0.98, 2)} / {round(current_price * 1.03, 2)}",
                "expiration": "15-30 days",
                "risk_level": "Low-Medium"
            }
    
    elif is_bearish:
        # Strong bearish patterns
        if "Evening Star" in pattern_name or "Three Black Crows" in pattern_name:
            recommendation = {
                "strategy": "Long Put",
                "options_type": "Put",
                "strike_price": round(current_price * 0.99, 2),  # Slightly OTM
                "expiration": "30-45 days",
                "risk_level": "Medium-High"
            }
        # Moderate bearish patterns
        elif "Bearish Engulfing" in pattern_name:
            recommendation = {
                "strategy": "Bear Put Spread",
                "options_type": "Put Spread",
                "strike_price": f"{round(current_price, 2)} / {round(current_price * 0.95, 2)}",
                "expiration": "30-45 days",
                "risk_level": "Medium"
            }
        # Weaker bearish patterns
        else:
            recommendation = {
                "strategy": "Bear Put Spread (Conservative)",
                "options_type": "Put Spread",
                "strike_price": f"{round(current_price * 1.02, 2)} / {round(current_price * 0.97, 2)}",
                "expiration": "15-30 days",
                "risk_level": "Low-Medium"
            }
    
    elif is_neutral:
        # Doji pattern shows indecision
        recommendation = {
            "strategy": "Iron Condor or Straddle",
            "options_type": "Iron Condor or Straddle",
            "strike_price": f"{round(current_price * 0.95, 2)} / {round(current_price * 1.05, 2)}",
            "expiration": "15-30 days",
            "risk_level": "Medium"
        }
    
    # Adjust based on volatility
    if volatility > 0.2:  # High volatility
        recommendation["strategy"] += " (Consider shorter expiration due to high volatility)"
        recommendation["risk_level"] = "High" if recommendation["risk_level"] != "High" else "Very High"
    
    return recommendation
