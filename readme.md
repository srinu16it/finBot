# LangGraph Stock Pattern Analyzer

A modular, AI-enhanced stock analysis tool built with Python, LangGraph, and Gemini 1.5. It fetches live stock data, detects technical chart patterns, computes indicators, and uses an LLM to summarize market outlook in natural language.

---

## 🚀 Features

- ✅ Live OHLCV data via Yahoo Finance (`yfinance`)
- ✅ Technical indicators (EMA, RSI, MACD, VWAP)
- ✅ Pattern detection nodes:
  - Double Top / Bottom
  - Triple Top / Bottom
  - Head & Shoulders / Inverse H&S
  - Rising / Falling Wedge
  - Bullish / Bearish Pennants
  - Bullish / Bearish Flags
  - Ascending / Descending / Symmetrical Triangles
- ✅ Gemini 1.5 LLM summary with reasoning
- ✅ LangGraph-powered execution with memory and trace
- ✅ Transparent LLM prompt logging to files

---

## 🧱 Architecture

```
User Input
   ↓
Extract Symbol
   ↓
[LangGraph Flow]
   → API Fetch (OHLCV)
   → Analysis Summary
   → Indicator Calculation
   → Pattern Detection (7 types)
   → LLM Reasoning (Gemini 1.5)
   ↓
Final Report (Market Direction + Trace)
```

All nodes operate within a shared `State` dictionary using LangGraph’s `StateGraph` abstraction. LangGraph provides:

- Type-safe flow definitions using `TypedDict`
- Visualizable graph structure
- Built-in memory management (`MemorySaver` for trace)
- Sequential and conditional edge control

Each node modifies the shared state and passes it along to the next. The flow ends with the LLM reason node that reads all indicators and patterns.

---

## 🥪 Setup

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/finBot.git
cd finBot
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Environment variables

Create a `.env` file:

```env
GEMINI_API_KEY=your_google_generative_ai_key
```

---

## 📦 File Structure

```
.
├── finBot.py              # Main execution file
├── prompts/               # Saved LLM prompts
├── llm_prompts_log.txt    # Flat log of all LLM prompt texts
├── README.md              # This file
├── .env                   # API key config
└── requirements.txt       # Python dependencies
```

---

## 🧠 Prompt Logging

LLM prompts are automatically:

- Printed to console
- Saved to `prompts/` folder as:
  ```
  llm_prompt_<SYMBOL>_<TIMESTAMP>.txt
  ```
- Appended to `llm_prompts_log.txt`

---

## 🧠 Example Usage

```
You: Tell me about TSLA
✅ Market Direction:
BULLISH — Indicators and triangle support continuation.
📜 Trace:
- api @ 2025-05-08T...
  📋 Fetched 720 rows for TSLA
- analyze ...
- indicators ...
- pattern_detector ...
- wedge ...
- triangle ...
- llm_reason ...
```

---

## 📘 References

- Yahoo Finance API via `yfinance`
- Google Gemini 1.5 API (via `google.generativeai`)
- LangGraph for orchestrated memory-safe reasoning
- Chart pattern logic based on Investopedia, IG Academy, and trading guides

---

## 🤛 Author

Built by Nikhil T. For questions or contributions, open an issue or PR!
