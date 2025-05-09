import os
import pandas as pd
import numpy as np
from datetime import datetime
import yfinance as yf
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
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

class State(TypedDict):
    messages: Annotated[list, add_messages]
    symbol: str
    ohlcv: pd.DataFrame
    analysis: str
    indicator_summary: str
    # Double pattern
    double_pattern_signal: str
    double_pattern_details: dict
    # Triple pattern
    triple_pattern_signal: str
    triple_pattern_details: dict
    # Head & Shoulders pattern
    hs_pattern_signal: str
    hs_pattern_details: dict
    # Wedge pattern
    wedge_pattern_signal: str
    wedge_pattern_details: dict
    # Pennant pattern
    pennant_pattern_signal: str
    pennant_pattern_details: dict
    # Flag pattern
    flag_pattern_signal: str
    flag_pattern_details: dict

    llm_opinion: str
    llm_prompt: str
    response: str
    trace: list
    api_error: str

llm = genai.GenerativeModel("gemini-1.5-pro")

def log_trace(state, step_name, notes=None):
    state["trace"] = state.get("trace", [])
    state["trace"].append({
        "step": step_name,
        "timestamp": datetime.now().isoformat(),
        "input_keys": list(state.keys()),
        "summary": notes or "(no summary)"
    })

def extract_stock_symbol(user_input: str) -> str:
    prompt = f"What stock symbol is mentioned in this message: \"{user_input}\"? Respond with just the symbol, or say 'followup' if none."
    response = llm.generate_content(prompt).text.strip().upper()
    return response if response != "FOLLOWUP" else "FOLLOWUP"

def api_node(state):
    symbol = state["symbol"]
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1mo", interval="1h")
        if hist.empty:
            raise ValueError("No data returned from Yahoo.")
        hist.reset_index(inplace=True)
        hist.rename(columns={"index": "Datetime"}, inplace=True)
        state["ohlcv"] = hist
        log_trace(state, "api", f"Fetched {len(hist)} OHLCV rows for {symbol}")
    except Exception as e:
        state["ohlcv"] = None
        state["api_error"] = str(e)
        log_trace(state, "api", f"Error: {e}")
    return state

def analyze_node(state):
    df = state.get("ohlcv")
    if df is None or df.empty:
        state["analysis"] = "❌ No price data"
        return state
    closes = df["Close"].tail(5).tolist()
    trend = "↑" if closes[-1] > closes[0] else "↓"
    state["analysis"] = f"5-hour trend: {trend} ({closes[0]:.2f} → {closes[-1]:.2f})"
    log_trace(state, "analyze", state["analysis"])
    return state

def indicator_node(state):
    df = state.get("ohlcv")
    if df is None or df.empty:
        state["indicator_summary"] = "❌ No data"
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
📈 Indicators:

Price: {price:.2f}
VWAP: {latest['VWAP']:.2f} → {'Above' if price > latest['VWAP'] else 'Below'}
EMA 9/21: {latest['EMA_9']:.2f} / {latest['EMA_21']:.2f} → {'Bullish' if latest['EMA_9'] > latest['EMA_21'] else 'Bearish'}
RSI: {latest['RSI']:.2f} → {'Overbought' if latest['RSI'] > 70 else 'Oversold' if latest['RSI'] < 30 else 'Neutral'}
MACD: {latest['MACD']:.2f}, Signal: {latest['MACD_Signal']:.2f} → {'Bullish' if latest['MACD'] > latest['MACD_Signal'] else 'Bearish'}
""".strip()

    log_trace(state, "indicators", "Indicators calculated")
    return state

def double_pattern_detector_node(state):
    df = state.get("ohlcv")
    if df is None or df.empty:
        state["double_pattern_signal"] = "❌ No pattern data"
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
    log_trace(state, "double_pattern_detector", f"{signal} | {details}")
    return state

def triple_pattern_detector_node(state):
    df = state.get("ohlcv")
    if df is None or df.empty:
        state["triple_pattern_signal"] = "❌ No pattern data"
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
    log_trace(state, "triple_pattern_detector", f"{signal} | {details}")
    return state

def head_shoulders_pattern_node(state):
    df = state.get("ohlcv")
    if df is None or df.empty:
        state["hs_pattern_signal"] = "❌ No pattern data"
        state["hs_pattern_details"] = {}
        return state

    prices = df["Close"].values
    df["local_max"] = df["Close"].iloc[argrelextrema(prices, np.greater_equal, order=3)[0]]
    df["local_min"] = df["Close"].iloc[argrelextrema(prices, np.less_equal, order=3)[0]]

    maxes = df.dropna(subset=["local_max"]).tail(7)
    mins = df.dropna(subset=["local_min"]).tail(7)

    signal = "⚠️ No head & shoulders pattern"
    details = {}
    close = df["Close"].iloc[-1]

    try:
        # Bearish Head & Shoulders: three peaks (L-H-R) with head higher, R ~ L, then break of neckline
        if len(maxes) >= 3:
            l, h, r = maxes["local_max"].values[-3:]
            if h > l and h > r and abs(l - r)/h < 0.05:
                neckline = df["Close"].iloc[maxes.index[-1]+1:].min()
                if close < neckline:
                    signal = "📉 Head & Shoulders → Bearish"
                    details = {"left": l, "head": h, "right": r, "neckline": neckline, "close": close}

        # Bullish Inverse Head & Shoulders: three troughs (L-H-R) with head lower, R ~ L, then break of neckline
        if len(mins) >= 3:
            l, h, r = mins["local_min"].values[-3:]
            if h < l and h < r and abs(l - r)/h < 0.05:
                neckline = df["Close"].iloc[mins.index[-1]+1:].max()
                if close > neckline:
                    signal = "📈 Inverse Head & Shoulders → Bullish"
                    details = {"left": l, "head": h, "right": r, "neckline": neckline, "close": close}
    except:
        pass

    state["hs_pattern_signal"] = signal
    state["hs_pattern_details"] = details
    log_trace(state, "hs_pattern_detector", f"{signal} | {details}")
    return state

def wedge_pattern_detector_node(state):
    df = state.get("ohlcv")
    if df is None or df.empty or len(df) < 30:
        state["wedge_pattern_signal"] = "❌ No wedge pattern data"
        state["wedge_pattern_details"] = {}
        return state

    # Use the last 50 candles for wedge detection
    df_slice = df.tail(50)
    highs = df_slice["High"].values
    lows = df_slice["Low"].values
    idx = np.arange(len(df_slice))

    # Linear regression: top and bottom lines
    top_fit = np.polyfit(idx, highs, 1)   # slope, intercept
    bot_fit = np.polyfit(idx, lows, 1)

    top_slope = top_fit[0]
    bot_slope = bot_fit[0]
    slope_diff = abs(top_slope - bot_slope)

    close = df_slice["Close"].iloc[-1]
    signal = "⚠️ No clear wedge pattern"
    details = {}

    try:
        if top_slope > 0 and bot_slope > 0 and bot_slope < top_slope and slope_diff > 0.01:
            signal = "📉 Rising Wedge → Bearish"
            details = {
                "top_slope": top_slope,
                "bot_slope": bot_slope,
                "type": "rising",
                "close": close
            }
        elif top_slope < 0 and bot_slope < 0 and bot_slope > top_slope and slope_diff > 0.01:
            signal = "📈 Falling Wedge → Bullish"
            details = {
                "top_slope": top_slope,
                "bot_slope": bot_slope,
                "type": "falling",
                "close": close
            }
    except Exception as e:
        signal = f"⚠️ Wedge detection error: {e}"

    state["wedge_pattern_signal"] = signal
    state["wedge_pattern_details"] = details
    log_trace(state, "wedge_pattern_detector", f"{signal} | {details}")
    return state

def pennant_pattern_detector_node(state):
    df = state.get("ohlcv")
    if df is None or df.empty or len(df) < 50:
        state["pennant_pattern_signal"] = "❌ No pennant pattern data"
        state["pennant_pattern_details"] = {}
        return state

    df_slice = df.tail(50).copy()
    prices = df_slice["Close"].values
    highs = df_slice["High"].values
    lows = df_slice["Low"].values
    idx = np.arange(len(df_slice))

    # Check for prior strong move (flagpole)
    pole_start = prices[0]
    pole_end = prices[10]
    pole_change = (pole_end - pole_start) / pole_start

    # Fit lines to highs and lows in consolidation zone
    top_fit = np.polyfit(idx, highs, 1)
    bot_fit = np.polyfit(idx, lows, 1)
    top_slope = top_fit[0]
    bot_slope = bot_fit[0]

    close = prices[-1]
    details = {}
    signal = "⚠️ No clear pennant pattern"

    try:
        # Bullish pennant: strong up pole, converging lines, low volume, breakout above top line
        if pole_change > 0.05 and top_slope < 0 and bot_slope > 0 and abs(top_slope - bot_slope) > 0.01:
            signal = "📈 Bullish Pennant → Continuation Up"
            details = {
                "pole_change": round(pole_change * 100, 2),
                "top_slope": top_slope,
                "bot_slope": bot_slope,
                "close": close,
                "type": "bullish"
            }
        # Bearish pennant: strong down pole, converging lines, breakout below bottom line
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
    log_trace(state, "pennant_pattern_detector", f"{signal} | {details}")
    return state

def flag_pattern_detector_node(state):
    df = state.get("ohlcv")
    if df is None or df.empty or len(df) < 50:
        state["flag_pattern_signal"] = "❌ No flag pattern data"
        state["flag_pattern_details"] = {}
        return state

    df_slice = df.tail(50).copy()
    prices = df_slice["Close"].values
    highs = df_slice["High"].values
    lows = df_slice["Low"].values
    idx = np.arange(len(df_slice))

    # Estimate flagpole (prior trend)
    pole_start = prices[0]
    pole_end = prices[10]
    pole_change = (pole_end - pole_start) / pole_start

    # Fit linear regression to highs and lows
    top_fit = np.polyfit(idx, highs, 1)
    bot_fit = np.polyfit(idx, lows, 1)
    top_slope = top_fit[0]
    bot_slope = bot_fit[0]
    close = prices[-1]
    details = {}
    signal = "⚠️ No flag pattern detected"

    try:
        # Bullish flag: strong up pole, downward parallel consolidation
        if pole_change > 0.05 and top_slope < 0 and bot_slope < 0 and abs(top_slope - bot_slope) < 0.01:
            signal = "📈 Bullish Flag → Continuation Up"
            details = {
                "pole_change": round(pole_change * 100, 2),
                "top_slope": top_slope,
                "bot_slope": bot_slope,
                "close": close,
                "type": "bullish"
            }
        # Bearish flag: strong down pole, upward parallel consolidation
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
    log_trace(state, "flag_pattern_detector", f"{signal} | {details}")
    return state

def llm_reason_node(state):
    df = state.get("ohlcv")
    if df is None or df.empty:
        state["llm_opinion"] = "❌ No OHLCV data to analyze"
        state["llm_prompt"] = ""
        log_trace(state, "llm_reason", "Skipped due to missing data")
        return state

    ohlcv_data = df.tail(50).to_dict(orient="records")

    pattern_debug = f"""
Double: {state.get("double_pattern_signal", "")}
Triple: {state.get("triple_pattern_signal", "")}
Head & Shoulders: {state.get("hs_pattern_signal", "")}
Wedge: {state.get("wedge_pattern_signal", "")}
Pennant: {state.get("pennant_pattern_signal", "")}
Flag: {state.get("flag_pattern_signal", "")}
"""

    indicators_raw = df[["EMA_9", "EMA_21", "RSI", "MACD", "MACD_Signal", "VWAP"]].dropna().tail(50).to_dict(orient="list")

    prompt = f"""
You are a technical market analyst.

📌 SYMBOL: {state['symbol']}

OHLCV (last 50):
{ohlcv_data}

Indicators:
{indicators_raw}

Patterns:
{pattern_debug}

Make a judgment: BULLISH, BEARISH, or NEUTRAL, and explain why.
""".strip()
    with open("llm_prompts_log.txt", "a", encoding="utf-8") as f:
        f.write(f"\n\n[{datetime.now().isoformat()}] SYMBOL: {state['symbol']}\n{prompt}\n")
    result = llm.generate_content(prompt).text.strip()
    state["llm_opinion"] = result
    state["llm_prompt"] = prompt
    log_trace(state, "llm_reason", f"LLM result: {result}")
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
builder.add_edge("flag_pattern_detector", "llm_reason")
builder.set_finish_point("llm_reason")

graph = builder.compile(checkpointer=None)

# Persistent message history
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
            print(f"- {step['step']} @ {step['timestamp']}")
            print(f"  📋 {step['summary']}\n")