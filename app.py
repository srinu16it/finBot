import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import numpy as np
import finBot
import datetime

# Configure page
st.set_page_config(page_title="Options Trading Advisor", page_icon="📈", layout="wide")

# Title and description
st.title("💸 Options Trading Advisor")
st.write("""
Get actionable options trading recommendations based on technical patterns and AI analysis.

Enter a stock symbol to receive short-term trading signals for options strategies.
""")

# Input section with expiry selection
with st.form(key='stock_form'):
    col1, col2 = st.columns([3, 2])
    with col1:
        symbol = st.text_input("Stock Symbol (e.g., AAPL, MSFT)", "AAPL")
    with col2:
        # Generate next 4 Fridays for options expiry
        today = datetime.datetime.now()
        friday_delta = (4 - today.weekday()) % 7  # Days until next Friday
        fridays = [(today + datetime.timedelta(days=friday_delta + i*7)).strftime("%b %d") for i in range(4)]
        expiry = st.selectbox("Options Expiry", fridays)
    submit_button = st.form_submit_button("🔍 Analyze Trading Opportunity")

if submit_button:
    with st.spinner("Analyzing stock data..."):
        try:
            # Run the analysis through the graph
            result = finBot.graph.invoke({"symbol": symbol})
            
            # Display results
            st.success("Analysis Complete!")
            
            # Main container for the app
            ohlcv = result.get("ohlcv")
            direction = result.get("llm_opinion", "No response").split("\n")[0].split(":")[-1].strip()
            indicator_summary = result.get("indicator_summary", "")
            
            if ohlcv is not None and not isinstance(ohlcv, str) and not ohlcv.empty:
                latest_price = ohlcv["Close"].iloc[-1]
                recent_high = ohlcv["High"].tail(20).max()
                recent_low = ohlcv["Low"].tail(20).min()
                atr = ohlcv["High"].tail(14).max() - ohlcv["Low"].tail(14).min()  # Simple ATR approximation
                daily_volatility = ohlcv["Close"].tail(5).pct_change().std() * 100
                
                # Determine confidence based on pattern detection and indicators
                confidence = 50  # Default moderate confidence
                if "No clear pattern" not in result.get("double_pattern_signal", "") or \
                   "No pattern" not in result.get("triple_pattern_signal", "") or \
                   "No pattern" not in result.get("hs_pattern_signal", ""):
                    confidence += 20  # Pattern detected
                
                # Adjust for strong direction in llm opinion
                if "strong" in direction.lower() or "clear" in direction.lower():
                    confidence += 15
                    
                # Set time frame based on data
                time_frame = "Short-term (1-5 days)"
                
                # Trading signal background color
                signal_color = "#FFB300" # Default yellow (wait)
                if direction.upper().startswith("BULLISH"):
                    signal_color = "#4CAF50" # Green
                elif direction.upper().startswith("BEARISH"):
                    signal_color = "#F44336" # Red
                
                # --- TRADING SIGNAL PANEL ---
                st.markdown("<h2 style='text-align: center;'>TRADING SIGNAL</h2>", unsafe_allow_html=True)
                
                cols = st.columns([2, 3, 2])
                with cols[1]:
                    if direction.upper().startswith("BULLISH"):
                        signal = "BUY CALL OPTIONS"
                        direction_text = "BULLISH"
                    elif direction.upper().startswith("BEARISH"):
                        signal = "BUY PUT OPTIONS"
                        direction_text = "BEARISH"
                    else:
                        signal = "WAIT"
                        direction_text = "NEUTRAL"
                    
                    st.markdown(f"<div style='background-color:{signal_color}; padding:20px; border-radius:10px; text-align:center;'>"
                              f"<h1 style='color:white; margin:0;'>{signal}</h1>"
                              f"<h3 style='color:white; margin:10px 0 0 0;'>{direction_text} • {confidence}% Confidence • {time_frame}</h3>"
                              f"</div>", unsafe_allow_html=True)
                
                # --- PRICE ACTION & ENTRY/EXIT STRATEGY ---
                st.markdown("<h2 style='margin-top:30px;'>ENTRY/EXIT STRATEGY</h2>", unsafe_allow_html=True)
                
                col1, col2 = st.columns([3, 2])
                with col1:
                    # Price chart with key levels
                    price_fig = go.Figure()
                    price_fig.add_trace(go.Candlestick(
                        x=ohlcv["Datetime"].tail(30),
                        open=ohlcv["Open"].tail(30),
                        high=ohlcv["High"].tail(30),
                        low=ohlcv["Low"].tail(30),
                        close=ohlcv["Close"].tail(30),
                        name="Price Action"
                    ))
                    
                    # Add key levels
                    target_price = 0
                    stop_price = 0
                    if direction.upper().startswith("BULLISH"):
                        target_price = round(latest_price + (atr * 1.5), 2)
                        stop_price = round(max(latest_price - (atr * 0.8), recent_low), 2)
                        price_fig.add_hline(y=target_price, line_dash="dash", line_color="green", annotation_text="Target")
                        price_fig.add_hline(y=stop_price, line_dash="dash", line_color="red", annotation_text="Stop Loss")
                        price_fig.add_hline(y=latest_price, line_dash="solid", line_color="blue", annotation_text="Entry")
                    elif direction.upper().startswith("BEARISH"):
                        target_price = round(latest_price - (atr * 1.5), 2)
                        stop_price = round(min(latest_price + (atr * 0.8), recent_high), 2)
                        price_fig.add_hline(y=target_price, line_dash="dash", line_color="green", annotation_text="Target")
                        price_fig.add_hline(y=stop_price, line_dash="dash", line_color="red", annotation_text="Stop Loss")
                        price_fig.add_hline(y=latest_price, line_dash="solid", line_color="blue", annotation_text="Entry")
                    
                    price_fig.update_layout(
                        title=f"{symbol} Price Action with Key Levels",
                        height=400,
                        xaxis_rangeslider_visible=False
                    )
                    st.plotly_chart(price_fig, use_container_width=True)
                
                with col2:
                    # Entry/Exit Strategy card
                    st.markdown("#### Key Price Levels")
                    
                    if direction.upper().startswith("BULLISH") or direction.upper().startswith("BEARISH"):
                        reward_risk = abs((target_price - latest_price) / (latest_price - stop_price))
                        win_rate_needed = 1 / (1 + reward_risk)  # Required win rate for profitable strategy
                        
                        levels_markdown = f"""
                        💰 **Current Price:** ${latest_price:.2f}  
                        🎯 **Target Price:** ${target_price:.2f} ({abs(target_price - latest_price) / latest_price * 100:.1f}% move)  
                        🛑 **Stop Loss:** ${stop_price:.2f} ({abs(stop_price - latest_price) / latest_price * 100:.1f}% move)  
                        ⚖️ **Reward/Risk Ratio:** {reward_risk:.2f}  
                        📊 **Required Win Rate:** {win_rate_needed*100:.1f}%  
                        ⏱️ **Time Frame:** {time_frame}  
                        🎲 **Confidence:** {confidence}%
                        """
                        st.markdown(levels_markdown)
                        
                        # TRADING PLAN
                        if direction.upper().startswith("BULLISH"):
                            strategy = f"Buy call options on {symbol} with strike near ${round(latest_price, 1)}"
                            exit_plan = f"- Take profits at ${target_price:.2f}\n- Cut losses at ${stop_price:.2f}"
                            ideal_scenario = "Price breaks above recent resistance with increasing volume"
                        else:
                            strategy = f"Buy put options on {symbol} with strike near ${round(latest_price, 1)}"
                            exit_plan = f"- Take profits at ${target_price:.2f}\n- Cut losses at ${stop_price:.2f}"
                            ideal_scenario = "Price breaks below recent support with increasing volume"
                        
                        st.markdown("#### TRADING PLAN")
                        st.markdown(f"**Strategy:** {strategy}")
                        st.markdown(f"**Expiry:** {expiry}")
                        st.markdown(f"**Exit Plan:**\n{exit_plan}")
                        st.markdown(f"**Ideal Scenario:** {ideal_scenario}")
                    else:
                        st.warning("No clear trading signal at this time. Recommendation: WAIT for better setup.")

            # --- PATTERN EVIDENCE & TECHNICAL SIGNALS ---
            st.markdown("<h2 style='margin-top:30px;'>PATTERN EVIDENCE & TECHNICAL SIGNALS</h2>", unsafe_allow_html=True)
            
            col1, col2 = st.columns([1, 1])
            with col1:
                # Technical Indicators Summary
                st.subheader("Technical Indicators")
                indicators = result.get("indicator_summary", "No indicators available")
                st.markdown(f"```{indicators}```")
                
            with col2:
                # Pattern Evidence
                st.subheader("Pattern Detection")
                patterns = {
                    "Double Pattern": result.get("double_pattern_signal", "No pattern"),
                    "Triple Pattern": result.get("triple_pattern_signal", "No pattern"),
                    "Head & Shoulders": result.get("hs_pattern_signal", "No pattern"),
                    "Wedge Pattern": result.get("wedge_pattern_signal", "No pattern"),
                    "Pennant Pattern": result.get("pennant_pattern_signal", "No pattern"),
                    "Flag Pattern": result.get("flag_pattern_signal", "No pattern"),
                    "Triangle Pattern": result.get("triangle_pattern_signal", "No pattern")
                }
                
                # Create a nice visual table of patterns
                for pattern, signal in patterns.items():
                    if "No" not in signal:
                        st.markdown(f"✅ **{pattern}:** {signal}")
                    else:
                        st.markdown(f"❌ **{pattern}:** {signal}")
            
            # --- RISK ASSESSMENT & AI REASONING ---
            st.markdown("<h2 style='margin-top:30px;'>RISK ASSESSMENT & AI REASONING</h2>", unsafe_allow_html=True)
            
            col1, col2 = st.columns([3, 1])
            with col1:
                # AI Analysis
                st.subheader("AI Market Analysis")
                st.write(result.get("llm_opinion", "No analysis available"))  # Full AI analysis
            
            with col2:
                # Risk factors
                st.subheader("Risk Factors")
                
                # Calculate volatility
                if ohlcv is not None and not isinstance(ohlcv, str) and not ohlcv.empty:
                    volatility = ohlcv["Close"].tail(14).pct_change().std() * 100
                    avg_volume = ohlcv["Volume"].tail(5).mean()
                    recent_volume = ohlcv["Volume"].tail(1).values[0]
                    rsi_signal = "Overbought" if "Overbought" in result.get("indicator_summary", "") else "Oversold" if "Oversold" in result.get("indicator_summary", "") else "Neutral"
                
                    # Show risk factors
                    st.markdown(f"🎯 **Volatility:** {volatility:.2f}% daily")
                    st.markdown(f"📊 **Volume:** {recent_volume:.0f} ({recent_volume/avg_volume:.1f}x avg)")
                    st.markdown(f"📈 **RSI Signal:** {rsi_signal}")
                    
                    # Options-specific risk
                    st.markdown(f"⏳ **Theta Risk:** {expiry} expiry")
                    
                    # Create a risk meter
                    risk_level = 3  # Default moderate
                    if volatility > 5:
                        risk_level = 5  # High
                    elif volatility < 2:
                        risk_level = 1  # Low
                    
                    st.markdown("#### Risk Meter")
                    risk_meter = ["🟢", "🟢", "🟡", "🟡", "🔴"]
                    st.markdown("".join(risk_meter[0:risk_level]) + "⚪"*(5-risk_level))
                    st.caption(f"Level {risk_level}/5: {'Low' if risk_level <= 2 else 'Moderate' if risk_level <= 3 else 'High'} risk trade")
                else:
                    st.warning("No data available for risk assessment")

            # Detailed trace is available as an expandable section
            with st.expander("View Analysis Process Trace"):
                trace = result.get("trace", [])
                if trace:
                    for step in trace:
                        st.markdown(f"- {step['step']} ({step['persona']} | {step['role']}) @ {step['timestamp']}<br> &nbsp;&nbsp;&nbsp;{step['summary']}", unsafe_allow_html=True)
                else:
                    st.info("No trace information available.")

            # Display pattern analysis
            st.header("Pattern Analysis")

            ohlcv = result.get("ohlcv")
            if ohlcv is not None and not isinstance(ohlcv, str) and not ohlcv.empty:
                # Focus on last 100 bars for clarity
                plot_df = ohlcv.copy().tail(100).reset_index(drop=True)

                # --- Double Pattern ---
                st.subheader("Double Pattern")
                double_signal = result.get("double_pattern_signal", "No pattern")
                double_details = result.get("double_pattern_details", {})
                # Find local peaks/troughs
                plot_df["local_max"] = plot_df["Close"].iloc[plot_df["Close"].argrelextrema(np.greater_equal, order=3)[0]] if hasattr(plot_df["Close"], 'argrelextrema') else None
                plot_df["local_min"] = plot_df["Close"].iloc[plot_df["Close"].argrelextrema(np.less_equal, order=3)[0]] if hasattr(plot_df["Close"], 'argrelextrema') else None

                # Plot price and peaks/troughs
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=plot_df["Datetime"], y=plot_df["Close"], mode="lines", name="Close"))
                # Mark peaks
                peaks = plot_df.dropna(subset=["local_max"])
                if not peaks.empty:
                    fig.add_trace(go.Scatter(x=peaks["Datetime"], y=peaks["local_max"], mode="markers", marker=dict(color="red", size=10), name="Peaks"))
                # Mark troughs
                troughs = plot_df.dropna(subset=["local_min"])
                if not troughs.empty:
                    fig.add_trace(go.Scatter(x=troughs["Datetime"], y=troughs["local_min"], mode="markers", marker=dict(color="blue", size=10), name="Troughs"))
                fig.update_layout(title="Double Pattern - Peaks and Troughs", xaxis_title="Date/Time", yaxis_title="Price")
                st.plotly_chart(fig, use_container_width=True)
                st.write(f"**Result:** {double_signal}")
                # Explanation
                if double_signal.startswith("⚠️"):
                    st.caption("No two peaks or troughs within 3% of each other and proper neckline found in the last 100 bars.")
                elif double_signal.startswith("📉") or double_signal.startswith("📈"):
                    st.caption("Pattern detected based on peaks/troughs and neckline criteria.")

                # --- Triple Pattern ---
                st.subheader("Triple Pattern")
                triple_signal = result.get("triple_pattern_signal", "No pattern")
                triple_details = result.get("triple_pattern_details", {})
                # Find local peaks/troughs
                plot_df["local_max"] = plot_df["Close"].iloc[plot_df["Close"].argrelextrema(np.greater_equal, order=3)[0]] if hasattr(plot_df["Close"], 'argrelextrema') else None
                plot_df["local_min"] = plot_df["Close"].iloc[plot_df["Close"].argrelextrema(np.less_equal, order=3)[0]] if hasattr(plot_df["Close"], 'argrelextrema') else None
                # Plot price and peaks/troughs
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=plot_df["Datetime"], y=plot_df["Close"], mode="lines", name="Close"))
                # Mark peaks
                peaks2 = plot_df.dropna(subset=["local_max"])
                if not peaks2.empty:
                    fig2.add_trace(go.Scatter(x=peaks2["Datetime"], y=peaks2["local_max"], mode="markers", marker=dict(color="red", size=10), name="Peaks"))
                # Mark troughs
                troughs2 = plot_df.dropna(subset=["local_min"])
                if not troughs2.empty:
                    fig2.add_trace(go.Scatter(x=troughs2["Datetime"], y=troughs2["local_min"], mode="markers", marker=dict(color="blue", size=10), name="Troughs"))
                fig2.update_layout(title="Triple Pattern - Peaks and Troughs", xaxis_title="Date/Time", yaxis_title="Price")
                st.plotly_chart(fig2, use_container_width=True)
                st.write(f"**Result:** {triple_signal}")
                # Explanation
                if triple_signal.startswith("⚠️"):
                    st.caption("No three peaks or troughs within 2% of each other and proper neckline found in the last 100 bars.")
                elif triple_signal.startswith("📉") or triple_signal.startswith("📈"):
                    st.caption("Pattern detected based on peaks/troughs and neckline criteria.")
            else:
                st.info("No price data available for pattern visualization.")

            # Head & Shoulders, Wedge, etc. can be added similarly


        except Exception as e:
            st.error(f"Error analyzing stock: {str(e)}")
