import os

import orjson
from langgraph.checkpoint.serde.base import SerializerProtocol
from langgraph.checkpoint.serde.types import SendProtocol
import pandas as pd
import numpy as np
from datetime import datetime
import yfinance as yf
from langchain_core.tracers import langchain
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import uuid

from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# # Custom DataFrame serializer
# class DataFrameSerializer(SerializerProtocol):
#     def dumps(self, obj: pd.DataFrame) -> bytes:
#         return orjson.dumps({
#             "data": obj.to_dict(orient="list"),
#             "columns": obj.columns.tolist(),
#             "index": obj.index.astype(str).tolist()
#         })
#
#     def loads(self, data: bytes) -> pd.DataFrame:
#         obj = orjson.loads(data)
#         return pd.DataFrame(
#             data=obj["data"],
#             columns=obj["columns"],
#             index=pd.DatetimeIndex(obj["index"])
#         )
#
#     # Required typed methods
#     def dumps_typed(self, obj: any) -> tuple[str, bytes]:
#         if isinstance(obj, pd.DataFrame):
#             return ("dataframe", self.dumps(obj))
#         return ("json", orjson.dumps(obj))
#
#     def loads_typed(self, type_id: str, data: bytes) -> any:
#         if type_id == "dataframe":
#             return self.loads(data)
#         return orjson.loads(data)


# # Enhanced SQLite Saver with DataFrame support
# class CustomSqliteSaver(SqliteSaver):
#     def __init__(self, conn: sqlite3.Connection):
#         super().__init__(conn, serde=DataFrameSerializer())


# --- Persona mapping for all nodes ---
NODE_PERSONAS = {
    "api": ("📊 Data Curator", "Financial data acquisition specialist"),
    "analyze": ("📈 Trend Analyst", "Price movement pattern expert"),
    "indicators": ("📉 Indicator Specialist", "Technical indicator maestro"),
    "double_pattern_detector": ("🔍 Double Pattern Expert", "Double top/bottom authority"),
    "triple_pattern_detector": ("🔍 Triple Pattern Expert", "Triple formation specialist"),
    "hs_pattern_detector": ("👤 Head & Shoulders Analyst", "Reversal pattern diagnostician"),
    "wedge_pattern_detector": ("🔻 Wedge Pattern Analyst", "Consolidation pattern expert"),
    "pennant_pattern_detector": ("🚩 Pennant Pattern Analyst", "Continuation pattern specialist"),
    "flag_pattern_detector": ("🏁 Flag Pattern Analyst", "Trend continuation expert"),
    "triangle_pattern_detector": ("🔺 Triangle Pattern Analyst", "Symmetrical pattern detector"),
    "llm_reason": ("🤖 LLM Market Strategist", "AI-powered market analysis synthesizer")
}

class State(TypedDict):
    messages: Annotated[list, add_messages]
    symbol: str
    ohlcv: pd.DataFrame
    analysis: str
    indicator_summary: str
    double_pattern_signal: str
    double_pattern_details: dict
    triple_pattern_signal: str
    triple_pattern_details: dict
    hs_pattern_signal: str
    hs_pattern_details: dict
    wedge_pattern_signal: str
    wedge_pattern_details: dict
    pennant_pattern_signal: str
    pennant_pattern_details: dict
    flag_pattern_signal: str
    flag_pattern_details: dict
    triangle_pattern_signal: str
    triangle_pattern_details: dict
    llm_opinion: str
    llm_prompt: str
    response: str
    trace: list
    api_error: str



def log_trace(state, step_name, notes=None):
    state["trace"] = state.get("trace", [])
    persona, role = NODE_PERSONAS.get(step_name, ("🧑‍💻 Analyst", "Generalist"))
    state["trace"].append({
        "step": step_name,
        "persona": persona,
        "role": role,
        "timestamp": datetime.now().isoformat(),
        "input_keys": list(state.keys()),
        "summary": notes or "(no summary)"
    })

def extract_stock_symbol(user_input: str) -> str:
    prompt = f"What stock symbol is mentioned in this message: \"{user_input}\"? Respond with just the symbol, or say 'followup' if none."
    result = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[{"role": "user", "content": prompt}]
    ).choices[0].message.content.strip().upper()
    return result if result != "FOLLOWUP" else "FOLLOWUP"

# API Node with serializable DataFrame
def api_node(state):
    symbol = state["symbol"]
    # Get lookback_days from state, default to 100 if not provided
    lookback_days = state.get("lookback_days", 100)
    
    try:
        ticker = yf.Ticker(symbol)
        # Use the lookback_days parameter to determine the period
        # For daily data, we use '{lookback_days}d' as the period
        hist = ticker.history(period=f"{lookback_days}d", interval="1d")
        
        # Explicitly limit to the requested number of days
        # Yahoo sometimes returns more data than requested
        if len(hist) > lookback_days:
            hist = hist.tail(lookback_days)
        
        if hist.empty:
            raise ValueError("No data returned from Yahoo.")
            
        # Reset index to get date as a column
        hist.reset_index(inplace=True)
        
        # The column name might be 'Date' for daily data instead of 'index'
        # Make sure we have a consistent column name for dates
        if 'Date' in hist.columns:
            # For daily data, Yahoo returns a 'Date' column
            hist.rename(columns={"Date": "Datetime"}, inplace=True)
        else:
            # For hourly data, Yahoo returns an 'index' column
            hist.rename(columns={"index": "Datetime"}, inplace=True)
            
        state["ohlcv"] = hist  # Will be serialized by CustomSqliteSaver
        log_trace(state, "api", f"Fetched {len(hist)} OHLCV rows for {symbol} with lookback of {lookback_days} days")
    except Exception as e:
        state["ohlcv"] = pd.DataFrame()
        state["api_error"] = str(e)
        log_trace(state, "api", f"Error: {e}")
    return state

def analyze_node(state):
    persona, role = NODE_PERSONAS["analyze"]
    df = state.get("ohlcv")
    if df is None or df.empty:
        state["analysis"] = "❌ No price data"
        log_trace(state, "analyze", f"{persona} - No price data")
        return state
    closes = df["Close"].tail(5).tolist()
    trend = "↑" if closes[-1] > closes[0] else "↓"
    state["analysis"] = f"{persona} 5-hour trend: {trend} ({closes[0]:.2f} → {closes[-1]:.2f})"
    log_trace(state, "analyze", state["analysis"])
    return state

def indicator_node(state):
    persona, role = NODE_PERSONAS["indicators"]
    df = state.get("ohlcv")
    if df is None or df.empty:
        state["indicator_summary"] = "❌ No data"
        log_trace(state, "indicators", f"{persona} - No data")
        return state
    df["EMA_9"] = df["Close"].ewm(span=9, adjust=False).mean()
    df["EMA_21"] = df["Close"].ewm(span=21, adjust=False).mean()
    delta = df["Close"].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    rs = pd.Series(gain).rolling(14).mean() / pd.Series(loss).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + rs))
    df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["VWAP"] = (df["Volume"] * (df["High"] + df["Low"] + df["Close"]) / 3).cumsum() / df["Volume"].cumsum()

    latest = df.dropna().iloc[-1]
    price = latest["Close"]

    state["indicator_summary"] = f"""
{persona} Indicators:

Price: {price:.2f}
VWAP: {latest['VWAP']:.2f} → {'Above' if price > latest['VWAP'] else 'Below'}
EMA 9/21: {latest['EMA_9']:.2f} / {latest['EMA_21']:.2f} → {'Bullish' if latest['EMA_9'] > latest['EMA_21'] else 'Bearish'}
RSI: {latest['RSI']:.2f} → {'Overbought' if latest['RSI'] > 70 else 'Oversold' if latest['RSI'] < 30 else 'Neutral'}
MACD: {latest['MACD']:.2f}, Signal: {latest['MACD_Signal']:.2f} → {'Bullish' if latest['MACD'] > latest['MACD_Signal'] else 'Bearish'}
""".strip()

    log_trace(state, "indicators", "Indicators calculated")
    return state

def double_pattern_detector_node(state):
    persona, role = NODE_PERSONAS["double_pattern_detector"]
    df = state.get("ohlcv")
    if df is None or df.empty:
        state["double_pattern_signal"] = "❌ No pattern data"
        log_trace(state, "double_pattern_detector", f"{persona} - No pattern data")
        return state

    prices = df["Close"].values
    df["local_max"] = df["Close"].iloc[argrelextrema(prices, np.greater_equal, order=3)[0]]
    df["local_min"] = df["Close"].iloc[argrelextrema(prices, np.less_equal, order=3)[0]]

    tops = df.dropna(subset=["local_max"]).tail(3)
    bottoms = df.dropna(subset=["local_min"]).tail(3)
    signal = "⚠️ No clear double pattern"
    details = {}
    close = df["Close"].iloc[-1]

    try:
        if len(tops) >= 2:
            top1, top2 = tops["local_max"].values[-2:]
            peak_diff = abs(top2 - top1) / top1
            neckline = df["Close"].iloc[tops.index[-1]+1:].min()
            if peak_diff < 0.03 and close < neckline:
                signal = "📉 Double Top → Bearish"
                details = {"top1": top1, "top2": top2, "neckline": neckline, "close": close}
        if len(bottoms) >= 2:
            bot1, bot2 = bottoms["local_min"].values[-2:]
            bottom_diff = abs(bot2 - bot1) / bot1
            neckline = df["Close"].iloc[bottoms.index[-1]+1:].max()
            if bottom_diff < 0.03 and close > neckline:
                signal = "📈 Double Bottom → Bullish"
                details = {"bot1": bot1, "bot2": bot2, "neckline": neckline, "close": close}
    except:
        pass

    state["double_pattern_signal"] = signal
    state["double_pattern_details"] = details
    log_trace(state, "double_pattern_detector", f"{persona} {signal} | {details}")
    return state

def triple_pattern_detector_node(state):
    persona, role = NODE_PERSONAS["triple_pattern_detector"]
    df = state.get("ohlcv")
    if df is None or df.empty:
        state["triple_pattern_signal"] = "❌ No pattern data"
        log_trace(state, "triple_pattern_detector", f"{persona} - No pattern data")
        return state

    prices = df["Close"].values
    df["local_max"] = df["Close"].iloc[argrelextrema(prices, np.greater_equal, order=3)[0]]
    df["local_min"] = df["Close"].iloc[argrelextrema(prices, np.less_equal, order=3)[0]]

    tops = df.dropna(subset=["local_max"]).tail(5)
    bottoms = df.dropna(subset=["local_min"]).tail(5)

    signal = "⚠️ No triple pattern"
    details = {}
    close = df["Close"].iloc[-1]

    try:
        if len(tops) >= 3:
            top1, top2, top3 = tops["local_max"].values[-3:]
            resistance = np.mean([top1, top2, top3])
            peak_dev = max(abs(top1 - resistance), abs(top2 - resistance), abs(top3 - resistance)) / resistance
            support = df["Close"].iloc[tops.index[-1]+1:].min()
            if peak_dev < 0.02 and close < support:
                signal = "📉 Triple Top → Bearish"
                details = {"top1": top1, "top2": top2, "top3": top3, "resistance": resistance, "support_broken": support, "close": close}
        if len(bottoms) >= 3:
            bot1, bot2, bot3 = bottoms["local_min"].values[-3:]
            support = np.mean([bot1, bot2, bot3])
            trough_dev = max(abs(bot1 - support), abs(bot2 - support), abs(bot3 - support)) / support
            resistance = df["Close"].iloc[bottoms.index[-1]+1:].max()
            if trough_dev < 0.02 and close > resistance:
                signal = "📈 Triple Bottom → Bullish"
                details = {"bot1": bot1, "bot2": bot2, "bot3": bot3, "support": support, "resistance_broken": resistance, "close": close}
    except:
        pass

    state["triple_pattern_signal"] = signal
    state["triple_pattern_details"] = details
    log_trace(state, "triple_pattern_detector", f"{persona} {signal} | {details}")
    return state

def head_shoulders_pattern_node(state):
    persona, role = NODE_PERSONAS["hs_pattern_detector"]
    df = state.get("ohlcv")
    if df is None or df.empty:
        state["hs_pattern_signal"] = "❌ No pattern data"
        state["hs_pattern_details"] = {}
        log_trace(state, "hs_pattern_detector", f"{persona} - No pattern data")
        return state

    # First, determine prior trend (needed to classify as reversal)
    df_full = df.copy()
    df_slice = df_full.tail(50)  # Focus on recent price action for pattern detection
    df_prior = df_full.iloc[-75:-50] if len(df_full) >= 75 else df_full.iloc[0:max(len(df_full)//2, 1)]
    
    # Get average price for prior period vs current period
    prior_avg = df_prior["Close"].mean()
    current_avg = df_slice["Close"].mean()
    prior_trend = "uptrend" if current_avg > prior_avg else "downtrend"
    
    # Detect local maxima and minima for pattern finding
    prices = df_slice["Close"].values
    df_slice["local_max"] = df_slice["Close"].iloc[argrelextrema(prices, np.greater_equal, order=3)[0]]
    df_slice["local_min"] = df_slice["Close"].iloc[argrelextrema(prices, np.less_equal, order=3)[0]]

    maxes = df_slice.dropna(subset=["local_max"]).tail(7)
    mins = df_slice.dropna(subset=["local_min"]).tail(7)

    signal = "⚠️ No head & shoulders pattern"
    details = {}
    close = df_slice["Close"].iloc[-1]

    try:
        # Regular Head & Shoulders (bearish reversal)
        if len(maxes) >= 3 and prior_trend == "uptrend":
            # Get the three most recent peaks
            l, h, r = maxes["local_max"].values[-3:]
            # Head should be higher than shoulders; shoulders should be roughly similar
            shoulder_symmetry = abs(l - r)/max(l, r) < 0.07  # Allow 7% difference in shoulders
            proper_formation = h > l and h > r and shoulder_symmetry
            
            if proper_formation:
                # Check if pattern is completed (price breaks neckline)
                # Neckline is formed by connecting lows between shoulders and head
                if len(mins) >= 2:
                    # Find troughs between peaks
                    troughs_between = []
                    for trough_idx in mins.index:
                        min_peak_idx = min(maxes.index)
                        max_peak_idx = max(maxes.index)
                        if min_peak_idx < trough_idx < max_peak_idx:
                            troughs_between.append(trough_idx)
                    
                    if len(troughs_between) >= 2:
                        # Draw neckline connecting these troughs
                        trough_indices = np.array(troughs_between)
                        trough_values = df_slice.loc[trough_indices, "local_min"].values
                        
                        # A proper neckline should be relatively flat
                        neckline_slope = np.abs(np.polyfit(range(len(trough_values)), trough_values, 1)[0])
                        is_flat_neckline = neckline_slope < 0.1  # Relatively flat
                        
                        # Confirm price has broken the neckline
                        neckline_level = np.mean(trough_values)  # Simplification: use average
                        neckline_broken = close < neckline_level
                        
                        if is_flat_neckline and neckline_broken:
                            signal = "📉 Head & Shoulders → Bearish Reversal"
                            details = {
                                "pattern": "Head & Shoulders", 
                                "classification": "Reversal (Bearish)",
                                "left": l, 
                                "head": h, 
                                "right": r, 
                                "neckline": neckline_level, 
                                "close": close,
                                "prior_trend": prior_trend
                            }
        
        # Inverse Head & Shoulders (bullish reversal)
        if len(mins) >= 3 and prior_trend == "downtrend" and "Bearish" not in signal:
            # Get the three most recent troughs
            l, h, r = mins["local_min"].values[-3:]
            # Head should be lower than shoulders; shoulders should be roughly similar
            shoulder_symmetry = abs(l - r)/min(l, r) < 0.07  # Allow 7% difference in shoulders
            proper_formation = h < l and h < r and shoulder_symmetry
            
            if proper_formation:
                # Check if pattern is completed (price breaks neckline)
                # Neckline is formed by connecting highs between shoulders and head
                if len(maxes) >= 2:
                    # Find peaks between troughs
                    peaks_between = []
                    for peak_idx in maxes.index:
                        min_trough_idx = min(mins.index)
                        max_trough_idx = max(mins.index)
                        if min_trough_idx < peak_idx < max_trough_idx:
                            peaks_between.append(peak_idx)
                    
                    if len(peaks_between) >= 2:
                        # Draw neckline connecting these peaks
                        peak_indices = np.array(peaks_between)
                        peak_values = df_slice.loc[peak_indices, "local_max"].values
                        
                        # A proper neckline should be relatively flat
                        neckline_slope = np.abs(np.polyfit(range(len(peak_values)), peak_values, 1)[0])
                        is_flat_neckline = neckline_slope < 0.1  # Relatively flat
                        
                        # Confirm price has broken the neckline
                        neckline_level = np.mean(peak_values)  # Simplification: use average
                        neckline_broken = close > neckline_level
                        
                        if is_flat_neckline and neckline_broken:
                            signal = "📈 Inverted Head & Shoulders → Bullish Reversal"
                            details = {
                                "pattern": "Inverted Head & Shoulders", 
                                "classification": "Reversal (Bullish)",
                                "left": l, 
                                "head": h, 
                                "right": r, 
                                "neckline": neckline_level, 
                                "close": close,
                                "prior_trend": prior_trend
                            }
    except Exception as e:
        signal = f"⚠️ H&S pattern detection error: {str(e)}"

    state["hs_pattern_signal"] = signal
    state["hs_pattern_details"] = details
    log_trace(state, "hs_pattern_detector", f"{persona} {signal} | {details}")
    return state

def wedge_pattern_detector_node(state):
    persona, role = NODE_PERSONAS["wedge_pattern_detector"]
    df = state.get("ohlcv")
    if df is None or df.empty or len(df) < 30:
        state["wedge_pattern_signal"] = "❌ No wedge pattern data"
        state["wedge_pattern_details"] = {}
        log_trace(state, "wedge_pattern_detector", f"{persona} - No wedge pattern data")
        return state

    # First, determine prior trend (needed to classify as reversal or continuation)
    df_full = df.copy()
    df_slice = df_full.tail(50)
    df_prior = df_full.iloc[-75:-50] if len(df_full) >= 75 else df_full.iloc[0:max(len(df_full)//2, 1)]
    
    # Get average price for prior period vs current period
    prior_avg = df_prior["Close"].mean()
    current_avg = df_slice["Close"].mean()
    prior_trend = "uptrend" if current_avg > prior_avg else "downtrend"
    
    # Linear regression on highs and lows of recent price action
    highs = df_slice["High"].values
    lows = df_slice["Low"].values
    idx = np.arange(len(df_slice))

    # Fit linear regression lines to highs and lows
    top_fit = np.polyfit(idx, highs, 1)
    bot_fit = np.polyfit(idx, lows, 1)

    top_slope = top_fit[0]
    bot_slope = bot_fit[0]
    slope_diff = abs(top_slope - bot_slope)

    # Calculate R² to ensure lines are good fits
    top_line = top_fit[0] * idx + top_fit[1]
    bot_line = bot_fit[0] * idx + bot_fit[1]
    
    top_r2 = 1 - (np.sum((highs - top_line)**2) / np.sum((highs - np.mean(highs))**2))
    bot_r2 = 1 - (np.sum((lows - bot_line)**2) / np.sum((lows - np.mean(lows))**2))
    
    
    # Check for convergence (wedges converge)
    is_converging = slope_diff > 0.01 and ((top_slope < 0 and bot_slope > top_slope) or 
                                      (top_slope > 0 and bot_slope < top_slope))
    
    # Check if price is near the pattern completion (near apex)
    apex_x = (bot_fit[1] - top_fit[1]) / (top_fit[0] - bot_fit[0]) if abs(top_fit[0] - bot_fit[0]) > 0.001 else 0
    near_completion = apex_x > 0 and apex_x < len(df_slice) * 1.5  # Near apex but not past it
    
    close = df_slice["Close"].iloc[-1]
    signal = "⚠️ No clear wedge pattern"
    details = {}

    try:
        # Rising Wedge (both slopes positive, top steeper)
        if (top_slope > 0 and bot_slope > 0 and bot_slope < top_slope and slope_diff > 0.01 and 
            top_r2 > 0.5 and bot_r2 > 0.5 and near_completion):
            
            if prior_trend == "uptrend":
                signal = "📉 Rising Wedge → Bearish Reversal"
                details = {"pattern": "Rising Wedge", "classification": "Reversal (Bearish)", 
                           "prior_trend": "Uptrend", "top_slope": top_slope, "bot_slope": bot_slope, 
                           "r2_values": [top_r2, bot_r2], "close": close}
            else:
                signal = "📉 Rising Wedge → Bearish Continuation"
                details = {"pattern": "Rising Wedge", "classification": "Continuation (Bearish)", 
                           "prior_trend": "Downtrend", "top_slope": top_slope, "bot_slope": bot_slope, 
                           "r2_values": [top_r2, bot_r2], "close": close}
                
        # Falling Wedge (both slopes negative, bottom less steep)
        elif (top_slope < 0 and bot_slope < 0 and bot_slope > top_slope and slope_diff > 0.01 and 
              top_r2 > 0.5 and bot_r2 > 0.5 and near_completion):
            
            if prior_trend == "downtrend":
                signal = "📈 Falling Wedge → Bullish Reversal"
                details = {"pattern": "Falling Wedge", "classification": "Reversal (Bullish)", 
                           "prior_trend": "Downtrend", "top_slope": top_slope, "bot_slope": bot_slope, 
                           "r2_values": [top_r2, bot_r2], "close": close}
            else:
                signal = "📈 Falling Wedge → Bullish Continuation"
                details = {"pattern": "Falling Wedge", "classification": "Continuation (Bullish)", 
                           "prior_trend": "Uptrend", "top_slope": top_slope, "bot_slope": bot_slope, 
                           "r2_values": [top_r2, bot_r2], "close": close}
    except Exception as e:
        signal = f"⚠️ Wedge detection error: {e}"

    state["wedge_pattern_signal"] = signal
    state["wedge_pattern_details"] = details
    log_trace(state, "wedge_pattern_detector", f"{persona} {signal} | {details}")
    return state

def pennant_pattern_detector_node(state):
    persona, role = NODE_PERSONAS["pennant_pattern_detector"]
    df = state.get("ohlcv")
    if df is None or df.empty or len(df) < 50:
        state["pennant_pattern_signal"] = "❌ No pennant pattern data"
        state["pennant_pattern_details"] = {}
        log_trace(state, "pennant_pattern_detector", f"{persona} - No pennant pattern data")
        return state

    df_slice = df.tail(50).copy()
    prices = df_slice["Close"].values
    highs = df_slice["High"].values
    lows = df_slice["Low"].values
    idx = np.arange(len(df_slice))

    pole_start = prices[0]
    pole_end = prices[10]
    pole_change = (pole_end - pole_start) / pole_start

    top_fit = np.polyfit(idx, highs, 1)
    bot_fit = np.polyfit(idx, lows, 1)
    top_slope = top_fit[0]
    bot_slope = bot_fit[0]

    close = prices[-1]
    details = {}
    signal = "⚠️ No clear pennant pattern"

    try:
        if pole_change > 0.05 and top_slope < 0 and bot_slope > 0 and abs(top_slope - bot_slope) > 0.01:
            signal = "📈 Bullish Pennant → Continuation Up"
            details = {
                "pole_change": round(pole_change * 100, 2),
                "top_slope": top_slope,
                "bot_slope": bot_slope,
                "close": close,
                "type": "bullish"
            }
        elif pole_change < -0.05 and top_slope < 0 and bot_slope > 0 and abs(top_slope - bot_slope) > 0.01:
            signal = "📉 Bearish Pennant → Continuation Down"
            details = {
                "pole_change": round(pole_change * 100, 2),
                "top_slope": top_slope,
                "bot_slope": bot_slope,
                "close": close,
                "type": "bearish"
            }
    except Exception as e:
        signal = f"⚠️ Pennant detection error: {e}"

    state["pennant_pattern_signal"] = signal
    state["pennant_pattern_details"] = details
    log_trace(state, "pennant_pattern_detector", f"{persona} {signal} | {details}")
    return state

def flag_pattern_detector_node(state):
    persona, role = NODE_PERSONAS["flag_pattern_detector"]
    df = state.get("ohlcv")
    if df is None or df.empty or len(df) < 50:
        state["flag_pattern_signal"] = "❌ No flag pattern data"
        state["flag_pattern_details"] = {}
        log_trace(state, "flag_pattern_detector", f"{persona} - No flag pattern data")
        return state

    df_slice = df.tail(50).copy()
    prices = df_slice["Close"].values
    highs = df_slice["High"].values
    lows = df_slice["Low"].values
    idx = np.arange(len(df_slice))

    pole_start = prices[0]
    pole_end = prices[10]
    pole_change = (pole_end - pole_start) / pole_start

    top_fit = np.polyfit(idx, highs, 1)
    bot_fit = np.polyfit(idx, lows, 1)
    top_slope = top_fit[0]
    bot_slope = bot_fit[0]
    close = prices[-1]
    details = {}
    signal = "⚠️ No flag pattern detected"

    try:
        if pole_change > 0.05 and top_slope < 0 and bot_slope < 0 and abs(top_slope - bot_slope) < 0.01:
            signal = "📈 Bullish Flag → Continuation Up"
            details = {
                "pole_change": round(pole_change * 100, 2),
                "top_slope": top_slope,
                "bot_slope": bot_slope,
                "close": close,
                "type": "bullish"
            }
        elif pole_change < -0.05 and top_slope > 0 and bot_slope > 0 and abs(top_slope - bot_slope) < 0.01:
            signal = "📉 Bearish Flag → Continuation Down"
            details = {
                "pole_change": round(pole_change * 100, 2),
                "top_slope": top_slope,
                "bot_slope": bot_slope,
                "close": close,
                "type": "bearish"
            }
    except Exception as e:
        signal = f"⚠️ Flag detection error: {e}"

    state["flag_pattern_signal"] = signal
    state["flag_pattern_details"] = details
    log_trace(state, "flag_pattern_detector", f"{persona} {signal} | {details}")
    return state

def triangle_pattern_detector_node(state):
    persona, role = NODE_PERSONAS["triangle_pattern_detector"]
    df = state.get("ohlcv")
    if df is None or df.empty or len(df) < 50:
        state["triangle_pattern_signal"] = "❌ No triangle pattern data"
        state["triangle_pattern_details"] = {}
        log_trace(state, "triangle_pattern_detector", f"{persona} - No triangle pattern data")
        return state

    # First, determine prior trend (needed to classify as continuation)
    df_full = df.copy()
    df_slice = df_full.tail(50).copy()
    df_prior = df_full.iloc[-75:-50] if len(df_full) >= 75 else df_full.iloc[0:max(len(df_full)//2, 1)]
    
    # Get average price for prior period vs current period
    prior_avg = df_prior["Close"].mean()
    current_avg = df_slice["Close"].mean()
    prior_trend = "uptrend" if current_avg > prior_avg else "downtrend"
    
    # Get price data
    highs = df_slice["High"].values
    lows = df_slice["Low"].values
    closes = df_slice["Close"].values
    volumes = df_slice["Volume"].values
    idx = np.arange(len(df_slice))

    # Linear regression for top and bottom trendlines
    top_fit = np.polyfit(idx, highs, 1)
    bot_fit = np.polyfit(idx, lows, 1)
    top_slope = top_fit[0]
    bot_slope = bot_fit[0]
    
    # Calculate R² to ensure lines are good fits
    top_line = top_fit[0] * idx + top_fit[1]
    bot_line = bot_fit[0] * idx + bot_fit[1]
    
    top_r2 = 1 - (np.sum((highs - top_line)**2) / np.sum((highs - np.mean(highs))**2))
    bot_r2 = 1 - (np.sum((lows - bot_line)**2) / np.sum((lows - np.mean(lows))**2))
    
    # Check for triangle convergence point
    apex_x = (bot_fit[1] - top_fit[1]) / (top_fit[0] - bot_fit[0]) if abs(top_fit[0] - bot_fit[0]) > 0.001 else 0
    near_apex = apex_x > 0 and apex_x < len(df_slice) * 1.5  # Within reasonable distance of the apex
    
    # Check for breakout conditions
    last_close = closes[-1]
    top_value = top_fit[0] * (len(idx) - 1) + top_fit[1]  # Value of top trendline at most recent point
    bot_value = bot_fit[0] * (len(idx) - 1) + bot_fit[1]  # Value of bottom trendline at most recent point
    
    # Increasing volume over pattern is often seen in real triangles
    recent_vol_avg = np.mean(volumes[-5:]) 
    earlier_vol_avg = np.mean(volumes[0:5])
    vol_increasing = recent_vol_avg > earlier_vol_avg
    
    # Identify pattern details for proper classification
    close = df_slice["Close"].iloc[-1]
    details = {}
    signal = "⚠️ No clear triangle pattern"

    try:
        # 1. Symmetrical Triangle - converging trendlines with similar absolute slopes
        if (abs(top_slope) > 0.001 and abs(bot_slope) > 0.001 and  # Both lines have slope
            ((top_slope < 0 and bot_slope > 0) or (top_slope > 0 and bot_slope < 0)) and  # One negative, one positive
            abs(abs(top_slope) - abs(bot_slope)) < 0.01 and  # Similar absolute slopes
            top_r2 > 0.4 and bot_r2 > 0.4 and  # Reasonable fit
            near_apex):  # Approaching convergence
            
            if prior_trend == "uptrend":
                signal = "📈 Bullish Triangle → Continuation"
                details = {
                    "pattern": "Symmetrical Triangle", 
                    "classification": "Continuation (Bullish)",
                    "top_slope": top_slope,
                    "bot_slope": bot_slope,
                    "r2_values": [top_r2, bot_r2],
                    "close": close,
                    "prior_trend": prior_trend
                }
            elif prior_trend == "downtrend":
                signal = "📉 Bearish Triangle → Continuation"
                details = {
                    "pattern": "Symmetrical Triangle", 
                    "classification": "Continuation (Bearish)",
                    "top_slope": top_slope,
                    "bot_slope": bot_slope,
                    "r2_values": [top_r2, bot_r2],
                    "close": close,
                    "prior_trend": prior_trend
                }
        
        # 2. Ascending Triangle - flat top, rising bottom (usually bullish continuation)
        elif (abs(top_slope) < 0.005 and bot_slope > 0.01 and  # Flat top, rising bottom
              top_r2 > 0.4 and bot_r2 > 0.4 and  # Good fit
              near_apex):  # Approaching convergence
            
            signal = "📈 Bullish Triangle (Ascending) → Continuation"
            details = {
                "pattern": "Ascending Triangle",
                "classification": "Continuation (Bullish)",
                "top_slope": top_slope,
                "bot_slope": bot_slope,
                "r2_values": [top_r2, bot_r2],
                "close": close,
                "prior_trend": prior_trend
            }
        
        # 3. Descending Triangle - flat bottom, declining top (usually bearish continuation)
        elif (abs(bot_slope) < 0.005 and top_slope < -0.01 and  # Flat bottom, declining top
              top_r2 > 0.4 and bot_r2 > 0.4 and  # Good fit
              near_apex):  # Approaching convergence
            
            signal = "📉 Bearish Triangle (Descending) → Continuation"
            details = {
                "pattern": "Descending Triangle",
                "classification": "Continuation (Bearish)",
                "top_slope": top_slope,
                "bot_slope": bot_slope,
                "r2_values": [top_r2, bot_r2],
                "close": close,
                "prior_trend": prior_trend
            }
        elif top_slope < 0 and bot_slope > 0 and abs(top_slope - bot_slope) > 0.01:
            signal = "🔼 Symmetrical Triangle → Neutral (Watch Breakout)"
            details = {
                "top_slope": top_slope,
                "bot_slope": bot_slope,
                "type": "symmetrical",
                "close": close,
            }
    except Exception as e:
        signal = f"⚠️ Triangle detection error: {e}"

    state["triangle_pattern_signal"] = signal
    state["triangle_pattern_details"] = details
    log_trace(state, "triangle_pattern_detector", f"{persona} {signal} | {details}")
    return state

def llm_reason_node(state):
    persona, role = NODE_PERSONAS["llm_reason"]
    df = state.get("ohlcv")
    if df is None or df.empty:
        state["llm_opinion"] = "❌ No OHLCV data to analyze"
        state["llm_prompt"] = ""
        log_trace(state, "llm_reason", f"{persona} - Skipped due to missing data")
        return state
    
    # Get closing price information
    latest_close = df["Close"].iloc[-1]
    prev_close = df["Close"].iloc[-2]
    week_ago_close = df["Close"].iloc[-5] if len(df) >= 5 else df["Close"].iloc[0]
    price_change_day = (latest_close - prev_close) / prev_close * 100
    price_change_week = (latest_close - week_ago_close) / week_ago_close * 100
    
    # Get volume trends
    latest_vol = df["Volume"].iloc[-1]
    avg_vol = df["Volume"].tail(10).mean()
    vol_ratio = latest_vol / avg_vol

    # Get key indicators from the last row (most recent)
    latest_indicators = df[["EMA_9", "EMA_21", "RSI", "MACD", "MACD_Signal", "VWAP"]].dropna().iloc[-1].to_dict()
    
    # Collect all pattern signals
    pattern_signals = {
        "Double Pattern": state.get("double_pattern_signal", "No data"),
        "Triple Pattern": state.get("triple_pattern_signal", "No data"),
        "Head & Shoulders": state.get("hs_pattern_signal", "No data"),
        "Wedge Pattern": state.get("wedge_pattern_signal", "No data"),
        "Pennant Pattern": state.get("pennant_pattern_signal", "No data"),
        "Flag Pattern": state.get("flag_pattern_signal", "No data"),
        "Triangle Pattern": state.get("triangle_pattern_signal", "No data")
    }
    
    # Count pattern types
    bullish_patterns = sum(1 for signal in pattern_signals.values() if "Bullish" in signal)
    bearish_patterns = sum(1 for signal in pattern_signals.values() if "Bearish" in signal)
    
    # Create an enhanced, structured prompt
    prompt = f"""
You are {persona} ({role}).

📌 SYMBOL: {state['symbol']}

===== PRICE DATA =====
Latest Close: ${latest_close:.2f}
Daily Change: {price_change_day:.2f}%
Weekly Change: {price_change_week:.2f}%

===== VOLUME =====
Latest Volume: {latest_vol:.0f}
Volume Ratio to 10-day Average: {vol_ratio:.2f}x

===== KEY INDICATORS =====
EMA 9/21: {latest_indicators.get('EMA_9', 'N/A'):.2f} / {latest_indicators.get('EMA_21', 'N/A'):.2f} 
    → {'Bullish' if latest_indicators.get('EMA_9', 0) > latest_indicators.get('EMA_21', 0) else 'Bearish'} crossover
    
RSI: {latest_indicators.get('RSI', 'N/A'):.2f} 
    → {'Overbought' if latest_indicators.get('RSI', 0) > 70 else 'Oversold' if latest_indicators.get('RSI', 0) < 30 else 'Neutral'}
    
MACD: {latest_indicators.get('MACD', 'N/A'):.4f}, Signal: {latest_indicators.get('MACD_Signal', 'N/A'):.4f} 
    → {'Bullish' if latest_indicators.get('MACD', -999) > latest_indicators.get('MACD_Signal', 999) else 'Bearish'} crossover
    
VWAP: {latest_indicators.get('VWAP', 'N/A'):.2f} 
    → Price is {'Above VWAP (Bullish)' if latest_close > latest_indicators.get('VWAP', 999999) else 'Below VWAP (Bearish)'}

===== DETECTED PATTERNS =====
"""

    # Add each pattern with prefix icon
    for pattern, signal in pattern_signals.items():
        icon = "🟢" if "Bullish" in signal else "🔴" if "Bearish" in signal else "⚪"
        prompt += f"\n{icon} {pattern}: {signal}"

    # Summary of pattern counts
    prompt += f"\n\nBullish patterns: {bullish_patterns}, Bearish patterns: {bearish_patterns}"

    # Add clear instructions and a template to ensure consistent output
    prompt += """

===== YOUR TASK =====
Provide a concise market judgment with the following format:

Judgment: [BULLISH/BEARISH/NEUTRAL]

Explanation:

1. Provide a brief but detailed explanation for your judgment based on the evidence.
2. Consider the current price, indicators (EMA, RSI, MACD, VWAP), detected technical patterns, and volume.
3. Weigh contradicting signals by their reliability and importance.
4. Explain the most compelling evidence for your judgment.
5. If there's uncertainty or conflicting signals, acknowledge it.
6. DO NOT hedge your conclusion. Make a clear determination.

Provide only this output and nothing else.
""".strip()

    # Log the prompt for debugging
    with open("llm_prompts_log.txt", "a", encoding="utf-8") as f:
        f.write(f"\n\n[{datetime.now().isoformat()}] SYMBOL: {state['symbol']}\n{prompt}\n")
    
    # Call the LLM with temperature=0.3 for high consistency
    result = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        temperature=0.3,  # Lower temperature for more consistent output
        messages=[{"role": "user", "content": prompt}]
    ).choices[0].message.content.strip()
    
    state["llm_opinion"] = result
    state["llm_prompt"] = prompt
    log_trace(state, "llm_reason", f"{persona} LLM result: {result}")

    return state

builder = StateGraph(State)

builder.add_node("api", api_node)
builder.add_node("analyze", analyze_node)
builder.add_node("indicators", indicator_node)
builder.add_node("double_pattern_detector", double_pattern_detector_node)
builder.add_node("triple_pattern_detector", triple_pattern_detector_node)
builder.add_node("hs_pattern_detector", head_shoulders_pattern_node)
builder.add_node("wedge_pattern_detector", wedge_pattern_detector_node)
builder.add_node("pennant_pattern_detector", pennant_pattern_detector_node)
builder.add_node("flag_pattern_detector", flag_pattern_detector_node)
builder.add_node("triangle_pattern_detector", triangle_pattern_detector_node)
builder.add_node("llm_reason", llm_reason_node)

builder.set_entry_point("api")
builder.add_edge("api", "analyze")
builder.add_edge("analyze", "indicators")
builder.add_edge("indicators", "double_pattern_detector")
builder.add_edge("double_pattern_detector", "triple_pattern_detector")
builder.add_edge("triple_pattern_detector", "hs_pattern_detector")
builder.add_edge("hs_pattern_detector", "wedge_pattern_detector")
builder.add_edge("wedge_pattern_detector", "pennant_pattern_detector")
builder.add_edge("pennant_pattern_detector", "flag_pattern_detector")
builder.add_edge("flag_pattern_detector", "triangle_pattern_detector")
builder.add_edge("triangle_pattern_detector", "llm_reason")
builder.set_finish_point("llm_reason")
langchain.debug=True



graph = builder.compile(checkpointer=None)

history: list[BaseMessage] = []

if __name__ == "__main__":
    print("💬 Ask about a stock (e.g., 'Tell me about TSLA') or type 'exit'.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ("exit", "quit"):
            break

        symbol = extract_stock_symbol(user_input)
        history.append(HumanMessage(content=user_input))
        thread_id = str(uuid.uuid4())
        result = graph.invoke({"symbol": symbol}, config={"thread_id": thread_id})

        print("\n✅ Market Direction:")
        print(result.get("llm_opinion", "No response."))

        print("\n📜 Trace:")
        for step in result["trace"]:
            print(f"- {step['step']} ({step['persona']} | {step['role']}) @ {step['timestamp']}")
            print(f"  📋 {step['summary']}\n")