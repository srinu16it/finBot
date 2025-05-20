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
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1mo", interval="1h")
        if hist.empty:
            raise ValueError("No data returned from Yahoo.")
        hist.reset_index(inplace=True)
        hist.rename(columns={"index": "Datetime"}, inplace=True)
        state["ohlcv"] = hist  # Will be serialized by CustomSqliteSaver
        log_trace(state, "api", f"Fetched {len(hist)} OHLCV rows for {symbol}")
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

    prices = df["Close"].values
    df["local_max"] = df["Close"].iloc[argrelextrema(prices, np.greater_equal, order=3)[0]]
    df["local_min"] = df["Close"].iloc[argrelextrema(prices, np.less_equal, order=3)[0]]

    maxes = df.dropna(subset=["local_max"]).tail(7)
    mins = df.dropna(subset=["local_min"]).tail(7)

    signal = "⚠️ No head & shoulders pattern"
    details = {}
    close = df["Close"].iloc[-1]

    try:
        if len(maxes) >= 3:
            l, h, r = maxes["local_max"].values[-3:]
            if h > l and h > r and abs(l - r)/h < 0.05:
                neckline = df["Close"].iloc[maxes.index[-1]+1:].min()
                if close < neckline:
                    signal = "📉 Head & Shoulders → Bearish"
                    details = {"left": l, "head": h, "right": r, "neckline": neckline, "close": close}
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

    df_slice = df.tail(50)
    highs = df_slice["High"].values
    lows = df_slice["Low"].values
    idx = np.arange(len(df_slice))

    top_fit = np.polyfit(idx, highs, 1)
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
            details = {"top_slope": top_slope, "bot_slope": bot_slope, "type": "rising", "close": close}
        elif top_slope < 0 and bot_slope < 0 and bot_slope > top_slope and slope_diff > 0.01:
            signal = "📈 Falling Wedge → Bullish"
            details = {"top_slope": top_slope, "bot_slope": bot_slope, "type": "falling", "close": close}
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

    df_slice = df.tail(50).copy()
    highs = df_slice["High"].values
    lows = df_slice["Low"].values
    idx = np.arange(len(df_slice))

    top_fit = np.polyfit(idx, highs, 1)
    bot_fit = np.polyfit(idx, lows, 1)
    top_slope = top_fit[0]
    bot_slope = bot_fit[0]

    close = df_slice["Close"].iloc[-1]
    details = {}
    signal = "⚠️ No clear triangle pattern"

    try:
        if abs(top_slope) < 0.005 and bot_slope > 0.01:
            signal = "🔺 Ascending Triangle → Bullish"
            details = {
                "top_slope": top_slope,
                "bot_slope": bot_slope,
                "type": "ascending",
                "close": close,
            }
        elif abs(bot_slope) < 0.005 and top_slope < -0.01:
            signal = "🔻 Descending Triangle → Bearish"
            details = {
                "top_slope": top_slope,
                "bot_slope": bot_slope,
                "type": "descending",
                "close": close,
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

    ohlcv_data = df.tail(50).to_dict(orient="records")

    pattern_debug = f"""
Double: {state.get("double_pattern_signal", "")}
Triple: {state.get("triple_pattern_signal", "")}
Head & Shoulders: {state.get("hs_pattern_signal", "")}
Wedge: {state.get("wedge_pattern_signal", "")}
Pennant: {state.get("pennant_pattern_signal", "")}
Flag: {state.get("flag_pattern_signal", "")}
Triangle: {state.get("triangle_pattern_signal", "")}
"""

    indicators_raw = df[["EMA_9", "EMA_21", "RSI", "MACD", "MACD_Signal", "VWAP"]].dropna().tail(50).to_dict(orient="list")

    prompt = f"""
You are {persona} ({role}).

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
    result = client.chat.completions.create(
        model="gpt-4-turbo-preview",
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