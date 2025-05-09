import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import yfinance as yf
import google.generativeai as genai
from langgraph.graph import StateGraph
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
llm = genai.GenerativeModel("gemini-1.5-pro")

memory_log = []

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

def branch_node(state):
    log_trace(state, "branch", f"Routing based on symbol: {state.get('symbol')}")
    return state

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

def pattern_detector_node(state):
    df = state.get("ohlcv")
    if df is None or df.empty:
        state["pattern_signal"] = "❌ No pattern data"
        return state

    prices = df["Close"].values
    df["local_max"] = df["Close"].iloc[argrelextrema(prices, np.greater_equal, order=3)[0]]
    df["local_min"] = df["Close"].iloc[argrelextrema(prices, np.less_equal, order=3)[0]]

    tops = df.dropna(subset=["local_max"]).tail(3)
    bottoms = df.dropna(subset=["local_min"]).tail(3)

    signal = "⚠️ No clear pattern"
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

    state["pattern_signal"] = signal
    state["pattern_details"] = details
    log_trace(state, "pattern_detector", f"{signal} | {details}")
    return state

def llm_reason_node(state):
    df = state["ohlcv"]
    latest = df.iloc[-1]
    ohlcv_data = df.tail(50).to_dict(orient="records")
    pattern_details = state.get("pattern_details", {})
    pattern_debug = {
        "tops": df["local_max"].dropna().tail(5).tolist(),
        "bottoms": df["local_min"].dropna().tail(5).tolist(),
        "neckline": pattern_details.get("neckline"),
        "close": pattern_details.get("close"),
        "breakout_confirmed": (
            pattern_details.get("close") > pattern_details.get("neckline")
            if "close" in pattern_details and "neckline" in pattern_details else False
        )
    }
    indicators_raw = df[["EMA_9", "EMA_21", "RSI", "MACD", "MACD_Signal", "VWAP"]].dropna().tail(50).to_dict(orient="list")

    prompt = f"""
You are a technical market analyst. Analyze the market direction (BULLISH, BEARISH, or NEUTRAL) based on the following complete dataset.

📌 SYMBOL: {state['symbol']}

📉 Raw OHLCV (last 50 points):
{ohlcv_data}

📈 Indicators (last 50 points):
{indicators_raw}

📀 Pattern Detection:
Pattern: {state['pattern_signal']}
Pattern Metrics:
{pattern_debug}

Your job is to analyze if the market is showing signs of bullish or bearish pressure based on:
- Price action
- Indicator alignment
- Breakout confirmation
- Historical support/resistance shape
- Any divergence or overbought/oversold conditions

Return a well-reasoned conclusion. End your answer with a single line that is one of: BULLISH, BEARISH, NEUTRAL
""".strip()

    result = llm.generate_content(prompt).text.strip()
    state["llm_opinion"] = result
    state["llm_prompt"] = prompt
    log_trace(state, "llm_reason", f"LLM input length: {len(prompt)} chars\nResponse:\n{result}")
    return state

def memory_node(state):
    memory_log.append({
        "timestamp": datetime.now().isoformat(),
        "symbol": state["symbol"],
        "trend": state["analysis"],
        "pattern": state["pattern_signal"],
        "llm_opinion": state["llm_opinion"]
    })
    log_trace(state, "memory", "Saved to memory")
    return state

def followup_node(state):
    last = memory_log[-1] if memory_log else {}
    prompt = f"""
Previous stock: {last.get('symbol')}
Trend: {last.get('trend')}
Pattern: {last.get('pattern')}
Opinion: {last.get('llm_opinion')}
User asked: {state['user_input']}

Respond accordingly.
"""
    result = llm.generate_content(prompt).text.strip()
    state["response"] = result
    log_trace(state, "followup", "Handled follow-up")
    return state

def plot_pattern_debug(df, symbol, pattern_details, pattern_signal):
    plt.figure(figsize=(12, 6))
    plt.plot(df["Datetime"], df["Close"], label="Close Price", color="black")

    if "local_max" in df:
        plt.scatter(df["Datetime"], df["local_max"], label="Peaks", color="red", marker="^")
    if "local_min" in df:
        plt.scatter(df["Datetime"], df["local_min"], label="Troughs", color="blue", marker="v")

    if "neckline" in pattern_details:
        plt.axhline(pattern_details["neckline"], color="orange", linestyle="--", label="Neckline")
    if "close" in pattern_details:
        plt.axhline(pattern_details["close"], color="green", linestyle="--", label="Close Price")

    plt.title(f"{symbol} Chart: {pattern_signal}")
    plt.xlabel("Datetime")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def branching(state):
    return "followup" if state["symbol"] == "FOLLOWUP" else "api"

graph = StateGraph(state_schema=dict)
graph.add_node("branch", branch_node)
graph.add_node("api", api_node)
graph.add_node("analyze", analyze_node)
graph.add_node("indicators", indicator_node)
graph.add_node("pattern_detector", pattern_detector_node)
graph.add_node("llm_reason", llm_reason_node)
graph.add_node("memory", memory_node)
graph.add_node("followup", followup_node)

graph.set_entry_point("branch")
graph.add_conditional_edges("branch", branching)
graph.add_edge("api", "analyze")
graph.add_edge("analyze", "indicators")
graph.add_edge("indicators", "pattern_detector")
graph.add_edge("pattern_detector", "llm_reason")
graph.add_edge("llm_reason", "memory")
graph.set_finish_point("memory")
graph.set_finish_point("followup")

flow = graph.compile()

def run_chat():
    print("💬 Ask about a stock (e.g., 'Tell me about TSLA') or type 'exit'.")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        symbol = extract_stock_symbol(user_input)
        state = {
            "user_input": user_input,
            "symbol": symbol,
            "trace": []
        }

        result = flow.invoke(state)

        print("\n✅ Market Direction:")
        print(result.get("response", result.get("llm_opinion", "No response.")))

        print("\n🧠 LLM Prompt Sent:")
        print(result.get("llm_prompt", "Not available"))

        print("\n📜 Trace:")
        for step in result["trace"]:
            print(f"- {step['step']} @ {step['timestamp']}")
            print(f"  📋 {step['summary']}\n")

        if result.get("ohlcv") is not None and result.get("pattern_details"):
            plot_pattern_debug(
                df=result["ohlcv"],
                symbol=result["symbol"],
                pattern_details=result["pattern_details"],
                pattern_signal=result["pattern_signal"]
            )

run_chat()