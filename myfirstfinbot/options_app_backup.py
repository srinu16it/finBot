import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import numpy as np
import finBot
import myfirstfinbot.candlestick_patterns as cp
from datetime import datetime, timedelta

# Configure page
st.set_page_config(page_title="Options Trading Advisor", page_icon="📈", layout="wide")

# Title and description
st.title("Options Trading Advisor")
st.markdown("""
This app analyzes stock patterns and provides clear options trading signals based on technical analysis.
Enter a stock symbol to get started.
""")

# Sidebar for user input
with st.sidebar:
    st.header("Stock Selection")
    symbol = st.text_input("Enter Stock Symbol (e.g., AAPL, MSFT)", "AAPL").upper()
    
    # Add time period selection
    st.header("Analysis Settings")
    lookback_days = st.slider("Lookback Period (days)", 30, 365, 100)
    
    # Add a submit button
    submit_button = st.button("Analyze Stock")

# Main app logic
if submit_button:
    with st.spinner("Analyzing stock data..."):
        try:
            # Run the analysis through the graph with the selected lookback period
            result = finBot.graph.invoke({"symbol": symbol, "lookback_days": lookback_days})
            
            # Display results
            st.success("Analysis Complete!")
            
            # Main container for the app
            ohlcv = result.get("ohlcv")
            direction = result.get("llm_opinion", "No response").split("\n")[0].split(":")[-1].strip()
            indicator_summary = result.get("indicator_summary", "")
            
            # Double-check that we're only using the requested number of days
            # This ensures the UI and the backend are in sync
            if ohlcv is not None and not isinstance(ohlcv, str) and not ohlcv.empty:
                if len(ohlcv) > lookback_days:
                    ohlcv = ohlcv.tail(lookback_days)
            
            if ohlcv is not None and not isinstance(ohlcv, str) and not ohlcv.empty:
                # Get key price data
                latest_price = ohlcv["Close"].iloc[-1]
                recent_high = ohlcv["High"].tail(20).max()
                recent_low = ohlcv["Low"].tail(20).min()
                atr = ohlcv["High"].tail(14).max() - ohlcv["Low"].tail(14).min()  # Simple ATR approximation
                daily_volatility = ohlcv["Close"].tail(5).pct_change().std() * 100
                
                # Create recent dataframe for visualization using all available data
                recent_df = ohlcv.copy()
                recent_df = recent_df.reset_index()
                
                # --- STOCK OVERVIEW ---
                st.markdown("<h2>STOCK OVERVIEW</h2>", unsafe_allow_html=True)
                
                # Create two columns for the overview
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Calculate the display period based on actual data length
                    actual_days = len(ohlcv)
                    display_period = f"Last {actual_days} Days"
                    
                    # Price chart with key levels - use all available data
                    price_fig = go.Figure()
                    
                    # Check which date column is available (could be 'Datetime', 'Date', or index)
                    if 'Datetime' in ohlcv.columns:
                        date_col = 'Datetime'
                    elif 'Date' in ohlcv.columns:
                        date_col = 'Date'
                    else:
                        # If neither exists, reset index to get the date as a column
                        ohlcv = ohlcv.reset_index()
                        date_col = 'Date' if 'Date' in ohlcv.columns else 'index'
                    
                    price_fig.add_trace(go.Candlestick(
                        x=ohlcv[date_col],
                        open=ohlcv["Open"],
                        high=ohlcv["High"],
                        low=ohlcv["Low"],
                        close=ohlcv["Close"],
                        name="Price"
                    ))
                    
                    # Add key levels
                    price_fig.add_hline(y=latest_price, line_dash="dash", line_color="blue", annotation_text="Current")
                    price_fig.add_hline(y=recent_high, line_dash="dash", line_color="green", annotation_text="Recent High")
                    price_fig.add_hline(y=recent_low, line_dash="dash", line_color="red", annotation_text="Recent Low")
                    
                    # Improve layout - use the actual display period in the title
                    price_fig.update_layout(
                        title=f"{symbol} - {display_period}",
                        xaxis_title="Date",
                        yaxis_title="Price ($)",
                        height=400,
                        xaxis_rangeslider_visible=False
                    )
                    
                    st.plotly_chart(price_fig, use_container_width=True)
                
                with col2:
                    # Key metrics
                    st.subheader("Key Metrics")
                    st.metric("Current Price", f"${latest_price:.2f}")
                    
                    # Calculate daily change
                    daily_change = ((ohlcv["Close"].iloc[-1] / ohlcv["Close"].iloc[-2]) - 1) * 100
                    st.metric("Daily Change", f"{daily_change:.2f}%", f"{daily_change:.2f}%")
                    
                    # Calculate weekly change
                    if len(ohlcv) >= 5:
                        weekly_change = ((ohlcv["Close"].iloc[-1] / ohlcv["Close"].iloc[-5]) - 1) * 100
                        st.metric("Weekly Change", f"{weekly_change:.2f}%", f"{weekly_change:.2f}%")
                    
                    # Calculate monthly change
                    if len(ohlcv) >= 20:
                        monthly_change = ((ohlcv["Close"].iloc[-1] / ohlcv["Close"].iloc[-20]) - 1) * 100
                        st.metric("Monthly Change", f"{monthly_change:.2f}%", f"{monthly_change:.2f}%")
                    
                    # Volatility
                    st.metric("Daily Volatility", f"{daily_volatility:.2f}%")
                    
                    # Market Direction
                    st.subheader("Market Direction")
                    if "bullish" in direction.lower():
                        st.markdown("🟢 **BULLISH**")
                    elif "bearish" in direction.lower():
                        st.markdown("🔴 **BEARISH**")
                    else:
                        st.markdown("⚪ **NEUTRAL**")
                
                # --- OPTIONS TRADING DASHBOARD ---
                st.markdown("<h2 style='margin-top:30px;'>OPTIONS TRADING DASHBOARD</h2>", unsafe_allow_html=True)
                
                # Get all patterns from the analysis
                patterns = {
                    "Double Pattern": result.get("double_pattern_signal", "No pattern"),
                    "Triple Pattern": result.get("triple_pattern_signal", "No pattern"),
                    "Head & Shoulders": result.get("hs_pattern_signal", "No pattern"),
                    "Wedge Pattern": result.get("wedge_pattern_signal", "No pattern"),
                    "Pennant Pattern": result.get("pennant_pattern_signal", "No pattern"),
                    "Flag Pattern": result.get("flag_pattern_signal", "No pattern"),
                    "Triangle Pattern": result.get("triangle_pattern_signal", "No pattern")
                }
                
                # Determine if we have bullish or bearish patterns
                bullish_signals = []
                bearish_signals = []
                
                # Check all patterns and classify them as bullish or bearish
                for pattern_type, signal in patterns.items():
                    if "No pattern" not in signal:
                        if any(term in signal for term in ["Bullish", "Bottom", "Support", "Inverse", "Inverted", "Ascending"]):
                            # Store pattern name, type, and a confidence score
                            pattern_name = pattern_type.replace(" Pattern", "")
                            confidence = 3  # Default medium confidence
                            if "Triple" in pattern_type: confidence = 4  # More reliable
                            if "Double" in pattern_type or "Head & Shoulders" in pattern_type: confidence = 3  # Reliable
                            bullish_signals.append((pattern_name, signal, confidence))
                        elif any(term in signal for term in ["Bearish", "Top", "Resistance", "Descending"]):
                            # Store pattern name, type, and a confidence score
                            pattern_name = pattern_type.replace(" Pattern", "")
                            confidence = 3  # Default medium confidence
                            if "Triple" in pattern_type: confidence = 4  # More reliable
                            if "Double" in pattern_type or "Head & Shoulders" in pattern_type: confidence = 3  # Reliable
                            bearish_signals.append((pattern_name, signal, confidence))
                
                # Create tabs for pattern summary and options strategies
                tabs = st.tabs(["Pattern Summary", "CALL Options", "PUT Options"])
                
                # --- Pattern Summary Tab ---
                with tabs[0]:
                    # Print the actual number of days in the data for debugging
                    st.sidebar.text(f"Data retrieved: {len(ohlcv)} days")
                    
                    # Use the new candlestick pattern detection module
                    candlestick_patterns = cp.detect_patterns(ohlcv)
                    
                    # Display detected patterns in a more organized way
                    st.subheader("📊 Candlestick Pattern Analysis")
                    
                    # Create a two-column layout for the patterns
                    col1, col2 = st.columns(2)
                    
                    # Display the candlestick patterns by category
                    with col1:
                        st.markdown("### Single & Double Candle Patterns")
                        
                        # Single candle pattern
                        single_pattern = candlestick_patterns["Single Candle"]
                        if "No significant" not in single_pattern:
                            st.markdown(f"**🕯️ {single_pattern}**")
                            # Show pattern description
                            description = cp.get_pattern_description(single_pattern)
                            with st.expander("What does this pattern mean?"):
                                st.markdown(description)
                        else:
                            st.markdown("*No significant single candle patterns detected*")
                        
                        # Double candle pattern
                        double_pattern = candlestick_patterns["Double Candle"]
                        if "No significant" not in double_pattern:
                            st.markdown(f"**🕯️🕯️ {double_pattern}**")
                            # Show pattern description
                            description = cp.get_pattern_description(double_pattern)
                            with st.expander("What does this pattern mean?"):
                                st.markdown(description)
                        else:
                            st.markdown("*No significant double candle patterns detected*")
                    
                    with col2:
                        st.markdown("### Triple & Confirmation Patterns")
                        
                        # Triple candle pattern
                        triple_pattern = candlestick_patterns["Triple Candle"]
                        if "No significant" not in triple_pattern:
                            st.markdown(f"**🕯️🕯️🕯️ {triple_pattern}**")
                            # Show pattern description
                            description = cp.get_pattern_description(triple_pattern)
                            with st.expander("What does this pattern mean?"):
                                st.markdown(description)
                        else:
                            st.markdown("*No significant triple candle patterns detected*")
                        
                        # Confirmation pattern
                        confirmation_pattern = candlestick_patterns["Confirmation"]
                        if "No confirmation" not in confirmation_pattern:
                            st.markdown(f"**✓ {confirmation_pattern}**")
                            # Show pattern description
                            description = cp.get_pattern_description(confirmation_pattern)
                            with st.expander("What does this pattern mean?"):
                                st.markdown(description)
                        else:
                            st.markdown("*No confirmation patterns detected*")
                    
                    # Overall pattern-based recommendation
                    st.markdown("### 🎯 Pattern-Based Options Strategy")
                    
                    # Find the most significant pattern for recommendation
                    significant_pattern = None
                    for pattern_type in ["Triple Candle", "Double Candle", "Single Candle", "Neutral"]:
                        if "No" not in candlestick_patterns[pattern_type]:
                            significant_pattern = candlestick_patterns[pattern_type]
                            break
                    
                    if significant_pattern:
                        # Get options recommendation
                        volatility = daily_volatility / 100  # Convert to decimal
                        recommendation = cp.get_options_recommendation(significant_pattern, latest_price, volatility)
                        
                        # Display recommendation
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"**Based on {significant_pattern.split('→')[0].strip()}:**")
                            st.markdown(f"**Strategy:** {recommendation['strategy']}")
                            st.markdown(f"**Options Type:** {recommendation['options_type']}")
                        with col2:
                            st.markdown(f"**Strike Price:** {recommendation['strike_price']}")
                            st.markdown(f"**Expiration:** {recommendation['expiration']}")
                            st.markdown(f"**Risk Level:** {recommendation['risk_level']}")
                    else:
                        st.info("No significant patterns detected to generate options strategy recommendations.")
                    
                    # Show technical indicators summary
                    st.markdown("### Technical Indicators")
                    indicators = result.get("indicator_summary", "No indicators available")
                    with st.expander("View Technical Indicators"):
                        st.markdown(f"```{indicators}```")
                        
                    # Check for patterns from finBot's pattern detection
                    st.markdown("### Additional Pattern Analysis")
                    found_patterns = []
                    for pattern_type, signal in patterns.items():
                        if "No pattern" not in signal:
                            found_patterns.append(f"**{pattern_type}**: {signal}")
                    
                    if found_patterns:
                        for pattern in found_patterns:
                            st.markdown(f"✅ {pattern}")
                    else:
                        st.info("No additional patterns detected in current price action.")
                
                # --- CALL Options Tab ---
                with tabs[1]:
                    # Check if we have any bullish signals
                    if bullish_signals:
                        st.success("✅ Bullish patterns detected! Displaying CALL options setup")
                        
                        # Sort signals by confidence
                        bullish_signals.sort(key=lambda x: x[2], reverse=True)
                        
                        # Display the detected patterns in a clean list
                        st.markdown("### Detected Bullish Patterns")
                        for pattern_name, signal, conf in bullish_signals:
                            st.markdown(f"**{pattern_name}**: {signal} (Confidence: {conf}/5)")
                        
                        # Calculate target price based on pattern confidence
                        strongest_pattern = bullish_signals[0]
                        pattern_name, signal, confidence = strongest_pattern
                        
                        # Set multiple targets based on pattern confidence and technical levels
                        stop_percent = 3.0  # Standard 3% stop loss
                        target1_percent = confidence * 1.0  # Short-term target
                        target2_percent = confidence * 2.0  # Medium-term target
                        
                        # Calculate actual price levels with precision
                        entry_low = round(latest_price * 0.995, 2)  # Slight buffer for entry zone
                        entry_high = round(latest_price * 1.005, 2)
                        stop_loss = round(latest_price * (1 - (stop_percent / 100)), 2)
                        target1 = round(latest_price * (1 + (target1_percent / 100)), 2)
                        target2 = round(latest_price * (1 + (target2_percent / 100)), 2)
                        
                        # Calculate profit percentages for options (leveraged)
                        option_multiplier = 5  # Approximate leverage for ATM options
                        option_target1_percent = target1_percent * option_multiplier
                        option_target2_percent = target2_percent * option_multiplier
                        option_stop_percent = stop_percent * option_multiplier
                        
                        # Show detailed entry/target/stop metrics
                        st.markdown("### 🎯 Precise Price Levels")
                        st.markdown(f"""                        
                        **Entry Zone**: ${entry_low}–{entry_high} (current levels)
                        
                        **Stop-loss**: ${stop_loss} ({stop_percent:.1f}% below entry)
                        
                        **Target 1**: ${target1} (short-term, +{target1_percent:.1f}%)
                        
                        **Target 2**: ${target2} (medium-term, +{target2_percent:.1f}%)
                        """)
                        
                        # Show position sizing recommendations
                        risk_per_trade = 1.0  # 1% account risk per trade
                        st.markdown("### 💰 Position Management")
                        st.markdown(f"""
                        **Initial Position**: 50-60% of planned allocation
                        
                        **Scaling Plan**: 
                        - Add 20-25% more if price holds above entry for 2 days
                        - Add final 20-25% if price breaks above ${round(latest_price * 1.02, 2)}
                        
                        **Risk Management**:
                        - Risk {risk_per_trade:.1f}% of account per trade
                        - Expected option return: +{option_target1_percent:.1f}% to +{option_target2_percent:.1f}%
                        - Maximum option loss: {option_stop_percent:.1f}%
                        - Risk/Reward: 1:{round(target1_percent/stop_percent, 1)} (minimum acceptable)
                        """)
                        
                        # Calculate option parameters
                        days_to_expiry = 30  # Default for short-term options
                        strike_price = round(latest_price * 1.01, 0)  # Slightly OTM
                        
                        # Calculate optimal strike and expiration
                        atm_strike = round(latest_price, 0)
                        otm_strike = round(latest_price * 1.03, 0)
                        
                        # Determine optimal expiration based on target timeframe
                        short_term_days = 30
                        medium_term_days = 60
                        
                        # Show detailed options strategy recommendation
                        st.markdown("### 🧠 CALL Options Strategy")
                        
                        # Calculate conviction level based on pattern strength and confirmation factors
                        conviction_factors = []
                        
                        # Check price action
                        if latest_price > recent_df["Close"].iloc[-5]:  # Price trending up
                            conviction_factors.append("✅ Price action")
                        
                        # Check volume confirmation
                        recent_volume = recent_df["Volume"].iloc[-5:].mean()
                        if recent_df["Volume"].iloc[-1] > recent_volume:
                            conviction_factors.append("✅ Volume confirmation")
                        
                        # Check breakout (if price is above recent resistance)
                        if latest_price > recent_df["High"].iloc[-20:-5].max():
                            conviction_factors.append("✅ Clean breakout")
                        
                        # Final verdict based on conviction factors
                        conviction_level = "Medium-Conviction Buy"
                        if len(conviction_factors) >= 3:
                            conviction_level = "High-Conviction Buy"
                        if len(conviction_factors) <= 1:
                            conviction_level = "Low-Conviction Buy"
                        
                        # Create a more visual, scannable conviction indicator
                        cols = st.columns([1, 2])
                        with cols[0]:
                            # Display conviction level with appropriate color
                            if "High" in conviction_level:
                                st.markdown(f"<h3 style='color:green;'>{conviction_level}</h3>", unsafe_allow_html=True)
                            elif "Medium" in conviction_level:
                                st.markdown(f"<h3 style='color:orange;'>{conviction_level}</h3>", unsafe_allow_html=True)
                            else:
                                st.markdown(f"<h3 style='color:gray;'>{conviction_level}</h3>", unsafe_allow_html=True)
                        
                        with cols[1]:
                            # Create a more compact checklist
                            checks = {
                                "Price Action": "✅" if "✅ Price action" in conviction_factors else "❌",
                                "Volume": "✅" if "✅ Volume confirmation" in conviction_factors else "❌",
                                "Breakout": "✅" if "✅ Clean breakout" in conviction_factors else "❌"
                            }
                            
                            # Display as a compact table
                            check_data = [[k, v] for k, v in checks.items()]
                            st.table(pd.DataFrame(check_data, columns=["Factor", "Status"]))
                        
                        st.markdown(f"""
                        ### 📈 Specific Options Recommendation
                        
                        **Primary Strategy:**
                        - Buy {symbol} {atm_strike} CALL
                        - Expiration: {short_term_days} days
                        - Target Gain: +{option_target1_percent:.1f}% (at Target 1)
                        - Max Loss: {option_stop_percent:.1f}%
                        
                        **Alternative Strategy:**
                        - Buy {symbol} {otm_strike} CALL (more aggressive)
                        - Expiration: {medium_term_days} days
                        - Target Gain: +{option_target2_percent:.1f}% (at Target 2)
                        - Max Loss: 100% (smaller position size recommended)
                        
                        **Scaling Strategy:**
                        - Start with 50-60% of planned allocation
                        - Add remaining position in 1-2 additional tranches if momentum continues
                        """)
                        
                        # Create a simple visualization
                        fig = go.Figure()
                        fig.add_trace(go.Candlestick(
                            x=recent_df.index,
                            open=recent_df['Open'],
                            high=recent_df['High'],
                            low=recent_df['Low'],
                            close=recent_df['Close'],
                            name="Price"
                        ))
                        
                        # Add entry, target and stop lines
                        fig.add_hline(y=latest_price, line_color="green", line_dash="dash", annotation_text="Entry")
                        fig.add_hline(y=target_price, line_color="blue", annotation_text="Target")
                        fig.add_hline(y=stop_loss, line_color="red", line_dash="dash", annotation_text="Stop")
                        
                        # Improve layout
                        fig.update_layout(
                            title=f"{symbol} - CALL Options Trading Setup ({pattern_name})",
                            height=400,
                            xaxis_rangeslider_visible=False
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No bullish patterns detected. Consider PUT options instead.")
                
                # --- PUT Options Tab ---
                with tabs[2]:
                    # Check if we have any bearish signals
                    if bearish_signals:
                        st.success("✅ Bearish patterns detected! Displaying PUT options setup")
                        
                        # Sort signals by confidence
                        bearish_signals.sort(key=lambda x: x[2], reverse=True)
                        
                        # Display the detected patterns in a clean list
                        st.markdown("### Detected Bearish Patterns")
                        for pattern_name, signal, conf in bearish_signals:
                            st.markdown(f"**{pattern_name}**: {signal} (Confidence: {conf}/5)")
                        
                        # Calculate target price based on pattern confidence
                        strongest_pattern = bearish_signals[0]
                        pattern_name, signal, confidence = strongest_pattern
                        
                        # Set multiple targets based on pattern confidence and technical levels
                        stop_percent = 3.0  # Standard 3% stop loss
                        target1_percent = confidence * 1.0  # Short-term target
                        target2_percent = confidence * 2.0  # Medium-term target
                        
                        # Calculate actual price levels with precision for bearish setup
                        entry_low = round(latest_price * 0.995, 2)  # Slight buffer for entry zone
                        entry_high = round(latest_price * 1.005, 2)
                        stop_loss = round(latest_price * (1 + (stop_percent / 100)), 2)  # Stop is higher for puts
                        target1 = round(latest_price * (1 - (target1_percent / 100)), 2)  # Price decreases
                        target2 = round(latest_price * (1 - (target2_percent / 100)), 2)  # Price decreases more
                        
                        # Calculate profit percentages for options (leveraged)
                        option_multiplier = 5  # Approximate leverage for ATM options
                        option_target1_percent = target1_percent * option_multiplier
                        option_target2_percent = target2_percent * option_multiplier
                        option_stop_percent = stop_percent * option_multiplier
                        
                        # Show detailed entry/target/stop metrics
                        st.markdown("### 🎯 Precise Price Levels")
                        st.markdown(f"""                        
                        **Entry Zone**: ${entry_low}–{entry_high} (current levels)
                        
                        **Stop-loss**: ${stop_loss} ({stop_percent:.1f}% above entry)
                        
                        **Target 1**: ${target1} (short-term, -{target1_percent:.1f}%)
                        
                        **Target 2**: ${target2} (medium-term, -{target2_percent:.1f}%)
                        """)
                        
                        # Show position sizing recommendations
                        risk_per_trade = 1.0  # 1% account risk per trade
                        st.markdown("### 💰 Position Management")
                        st.markdown(f"""
                        **Initial Position**: 50-60% of planned allocation
                        
                        **Scaling Plan**: 
                        - Add 20-25% more if price holds below entry for 2 days
                        - Add final 20-25% if price breaks below ${round(latest_price * 0.98, 2)}
                        
                        **Risk Management**:
                        - Risk {risk_per_trade:.1f}% of account per trade
                        - Expected option return: +{option_target1_percent:.1f}% to +{option_target2_percent:.1f}%
                        - Maximum option loss: {option_stop_percent:.1f}%
                        - Risk/Reward: 1:{round(target1_percent/stop_percent, 1)} (minimum acceptable)
                        """)
                        
                        # Calculate option parameters
                        days_to_expiry = 30  # Default for short-term options
                        strike_price = round(latest_price * 0.99, 0)  # Slightly OTM put
                        
                        # Calculate optimal strike and expiration
                        atm_strike = round(latest_price, 0)
                        otm_strike = round(latest_price * 0.97, 0)
                        
                        # Determine optimal expiration based on target timeframe
                        short_term_days = 30
                        medium_term_days = 60
                        
                        # Show detailed options strategy recommendation
                        st.markdown("### 🧠 PUT Options Strategy")
                        
                        # Calculate conviction level based on pattern strength and confirmation factors
                        conviction_factors = []
                        
                        # Check price action
                        if latest_price < recent_df["Close"].iloc[-5]:  # Price trending down
                            conviction_factors.append("✅ Price action")
                        
                        # Check volume confirmation
                        recent_volume = recent_df["Volume"].iloc[-5:].mean()
                        if recent_df["Volume"].iloc[-1] > recent_volume:
                            conviction_factors.append("✅ Volume confirmation")
                        
                        # Check breakdown (if price is below recent support)
                        if latest_price < recent_df["Low"].iloc[-20:-5].min():
                            conviction_factors.append("✅ Clean breakdown")
                        
                        # Final verdict based on conviction factors
                        conviction_level = "Medium-Conviction Sell"
                        if len(conviction_factors) >= 3:
                            conviction_level = "High-Conviction Sell"
                        if len(conviction_factors) <= 1:
                            conviction_level = "Low-Conviction Sell"
                        
                        # Create a more visual, scannable conviction indicator
                        cols = st.columns([1, 2])
                        with cols[0]:
                            # Display conviction level with appropriate color
                            if "High" in conviction_level:
                                st.markdown(f"<h3 style='color:red;'>{conviction_level}</h3>", unsafe_allow_html=True)
                            elif "Medium" in conviction_level:
                                st.markdown(f"<h3 style='color:orange;'>{conviction_level}</h3>", unsafe_allow_html=True)
                            else:
                                st.markdown(f"<h3 style='color:gray;'>{conviction_level}</h3>", unsafe_allow_html=True)
                        
                        with cols[1]:
                            # Create a more compact checklist
                            checks = {
                                "Price Action": "✅" if "✅ Price action" in conviction_factors else "❌",
                                "Volume": "✅" if "✅ Volume confirmation" in conviction_factors else "❌",
                                "Breakdown": "✅" if "✅ Clean breakdown" in conviction_factors else "❌"
                            }
                            
                            # Display as a compact table
                            check_data = [[k, v] for k, v in checks.items()]
                            st.table(pd.DataFrame(check_data, columns=["Factor", "Status"]))
                        
                        st.markdown(f"""
                        ### 📉 Specific Options Recommendation
                        
                        **Primary Strategy:**
                        - Buy {symbol} {atm_strike} PUT
                        - Expiration: {short_term_days} days
                        - Target Gain: +{option_target1_percent:.1f}% (at Target 1)
                        - Max Loss: {option_stop_percent:.1f}%
                        
                        **Alternative Strategy:**
                        - Buy {symbol} {otm_strike} PUT (more aggressive)
                        - Expiration: {medium_term_days} days
                        - Target Gain: +{option_target2_percent:.1f}% (at Target 2)
                        - Max Loss: 100% (smaller position size recommended)
                        
                        **Scaling Strategy:**
                        - Start with 50-60% of planned allocation
                        - Add remaining position in 1-2 additional tranches if momentum continues
                        """)
                        
                        # Create a simple visualization
                        fig = go.Figure()
                        fig.add_trace(go.Candlestick(
                            x=recent_df.index,
                            open=recent_df['Open'],
                            high=recent_df['High'],
                            low=recent_df['Low'],
                            close=recent_df['Close'],
                            name="Price"
                        ))
                        
                        # Add entry, target and stop lines
                        fig.add_hline(y=latest_price, line_color="red", line_dash="dash", annotation_text="Entry")
                        fig.add_hline(y=target_price, line_color="blue", annotation_text="Target")
                        fig.add_hline(y=stop_loss, line_color="green", line_dash="dash", annotation_text="Stop")
                        
                        # Improve layout
                        fig.update_layout(
                            title=f"{symbol} - PUT Options Trading Setup ({pattern_name})",
                            height=400,
                            xaxis_rangeslider_visible=False
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No bearish patterns detected. Consider CALL options instead.")
                
                # --- AI MARKET SUMMARY ---
                st.markdown("<h2 style='margin-top:30px;'>AI MARKET SUMMARY</h2>", unsafe_allow_html=True)
                
                # Get LLM reasoning and parse key points
                llm_opinion = result.get("llm_opinion", "No analysis available")
                
                # Extract the market direction from the first line
                market_direction = "NEUTRAL"
                if "bullish" in llm_opinion.lower():
                    market_direction = "BULLISH"
                elif "bearish" in llm_opinion.lower():
                    market_direction = "BEARISH"
                
                # Create a more concise, scannable summary
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("Quick Take")
                    
                    # Display market direction with color
                    if market_direction == "BULLISH":
                        st.markdown("<h3 style='color:green;'>🟢 BULLISH</h3>", unsafe_allow_html=True)
                    elif market_direction == "BEARISH":
                        st.markdown("<h3 style='color:red;'>🔴 BEARISH</h3>", unsafe_allow_html=True)
                    else:
                        st.markdown("<h3 style='color:gray;'>⚪ NEUTRAL</h3>", unsafe_allow_html=True)
                    
                    # Extract key points from the LLM opinion
                    key_points = []
                    for line in llm_opinion.split("\n"):
                        if line.strip() and ":" not in line and len(line) > 15 and not line.startswith("#"):
                            key_points.append(line.strip())
                    
                    # Display 2-3 key points
                    st.markdown("**Key Points:**")
                    for i, point in enumerate(key_points[:3]):
                        st.markdown(f"• {point}")
                
                with col2:
                    # Show a condensed version of technical indicators
                    st.subheader("Technical Signals")
                    
                    # Extract trend signals
                    trend_signals = []
                    if "uptrend" in llm_opinion.lower() or "bullish trend" in llm_opinion.lower():
                        trend_signals.append("🟢 Uptrend")
                    if "downtrend" in llm_opinion.lower() or "bearish trend" in llm_opinion.lower():
                        trend_signals.append("🔴 Downtrend")
                    if "consolidation" in llm_opinion.lower() or "sideways" in llm_opinion.lower():
                        trend_signals.append("⚪ Consolidation")
                    
                    # Display trend signals
                    for signal in trend_signals:
                        st.markdown(signal)
                    
                    # Extract support/resistance levels
                    support_resistance = []
                    for line in llm_opinion.split("\n"):
                        if "support" in line.lower() or "resistance" in line.lower():
                            support_resistance.append(line.strip())
                    
                    # Display support/resistance levels
                    if support_resistance:
                        st.markdown("**Key Levels:**")
                        for level in support_resistance[:2]:  # Show max 2 levels
                            st.markdown(f"• {level}")
                
                # Add a "View Full Analysis" expander for those who want details
                with st.expander("View Full AI Analysis", expanded=False):
                    st.markdown(f"```{llm_opinion}```")
                
            else:
                st.error(f"Could not fetch data for {symbol}. Please check the symbol and try again.")
                
        except Exception as e:
            st.error(f"Error analyzing stock: {str(e)}")
