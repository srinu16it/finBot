"""
Streamlit UI for testing AlphaVantage data provider.

This application demonstrates the enhanced AlphaVantage provider
with caching and rate limiting capabilities.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import os
import sys

# Add enhancements to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from enhancements.data_providers.alpha_provider import AlphaVantageProvider
from enhancements.data_access.cache import CacheManager


# Page configuration
st.set_page_config(
    page_title="AlphaVantage Data Provider Test",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("🔬 AlphaVantage Data Provider Testing")
st.markdown("""
This app tests the enhanced AlphaVantage data provider with:
- 🚦 Automatic rate limiting (5 calls/minute for free tier)
- 💾 DuckDB caching to reduce API calls
- 📊 Multiple data types (daily, intraday, indicators, quotes)
""")

# Check for API key
if "ALPHAVANTAGE_API_KEY" not in os.environ:
    st.error("⚠️ Please set the ALPHAVANTAGE_API_KEY environment variable!")
    st.info("Get your free API key at: https://www.alphavantage.co/support/#api-key")
    
    # Allow user to input API key temporarily
    api_key = st.text_input("Enter your AlphaVantage API key:", type="password")
    if api_key:
        os.environ["ALPHAVANTAGE_API_KEY"] = api_key
        st.success("✅ API key set for this session!")
        st.rerun()
    else:
        st.stop()

# Initialize provider
@st.cache_resource
def get_provider():
    """Get or create AlphaVantage provider instance."""
    cache_manager = CacheManager()
    return AlphaVantageProvider(cache_manager)

provider = get_provider()

# Sidebar controls
st.sidebar.header("📋 Data Request Configuration")

# Symbol input
symbol = st.sidebar.text_input("Stock Symbol", value="AAPL").upper()

# Data type selection
data_type = st.sidebar.selectbox(
    "Data Type",
    ["Daily", "Intraday", "Quote", "Technical Indicator"]
)

# Cache control
use_cache = st.sidebar.checkbox("Use Cache", value=True)
if st.sidebar.button("Clear All Cache"):
    provider.cache_manager.clear_all()
    st.sidebar.success("Cache cleared!")

# Show cache stats
with st.sidebar.expander("📊 Cache Statistics"):
    stats = provider.cache_manager.get_stats()
    st.json(stats)

# Main content area
col1, col2 = st.columns([3, 1])

with col1:
    st.header(f"📈 {symbol} - {data_type} Data")

# Handle different data types
if data_type == "Daily":
    outputsize = st.sidebar.selectbox("Output Size", ["compact", "full"])
    
    if st.button("Fetch Daily Data", type="primary"):
        with st.spinner("Fetching data..."):
            try:
                df = provider.get_daily(symbol, outputsize=outputsize, use_cache=use_cache)
                
                if df is not None and not df.empty:
                    # Store in session state
                    st.session_state.daily_data = df
                    st.success(f"✅ Fetched {len(df)} days of data")
                else:
                    st.error("No data returned")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # Display data if available
    if "daily_data" in st.session_state:
        df = st.session_state.daily_data
        
        # Plot
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='OHLC'
        ))
        fig.update_layout(
            title=f"{symbol} Daily Price Chart",
            yaxis_title="Price ($)",
            xaxis_title="Date",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Volume subplot
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume'))
        fig_vol.update_layout(
            title="Volume",
            height=200,
            xaxis_title="Date",
            yaxis_title="Volume"
        )
        st.plotly_chart(fig_vol, use_container_width=True)
        
        # Show data table
        with st.expander("📊 View Raw Data"):
            st.dataframe(df)

elif data_type == "Intraday":
    interval = st.sidebar.selectbox(
        "Interval",
        ["1min", "5min", "15min", "30min", "60min"]
    )
    outputsize = st.sidebar.selectbox("Output Size", ["compact", "full"])
    
    if st.button("Fetch Intraday Data", type="primary"):
        with st.spinner("Fetching data..."):
            try:
                df = provider.get_intraday(
                    symbol, 
                    interval=interval, 
                    outputsize=outputsize, 
                    use_cache=use_cache
                )
                
                if df is not None and not df.empty:
                    st.session_state.intraday_data = df
                    st.success(f"✅ Fetched {len(df)} data points")
                else:
                    st.error("No data returned")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # Display data if available
    if "intraday_data" in st.session_state:
        df = st.session_state.intraday_data
        
        # Line chart for intraday
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['Close'],
            mode='lines',
            name='Close Price',
            line=dict(color='blue', width=2)
        ))
        fig.update_layout(
            title=f"{symbol} Intraday Price ({interval})",
            yaxis_title="Price ($)",
            xaxis_title="Time",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Show data table
        with st.expander("📊 View Raw Data"):
            st.dataframe(df)

elif data_type == "Quote":
    if st.button("Fetch Real-time Quote", type="primary"):
        with st.spinner("Fetching quote..."):
            try:
                quote = provider.get_quote(symbol, use_cache=use_cache)
                
                if quote:
                    st.session_state.quote = quote
                    st.success("✅ Quote fetched successfully")
                else:
                    st.error("No quote data returned")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # Display quote if available
    if "quote" in st.session_state:
        quote = st.session_state.quote
        
        # Quote metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Current Price",
                f"${quote['price']:.2f}",
                f"{quote['change']:.2f} ({quote['change_percent']})"
            )
        
        with col2:
            st.metric("Open", f"${quote['open']:.2f}")
        
        with col3:
            st.metric("High", f"${quote['high']:.2f}")
        
        with col4:
            st.metric("Low", f"${quote['low']:.2f}")
        
        # Additional info
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Volume", f"{quote['volume']:,}")
        
        with col2:
            st.metric("Previous Close", f"${quote['previous_close']:.2f}")
        
        with col3:
            st.metric("Trading Day", quote['latest_trading_day'])
        
        # Full quote data
        with st.expander("📊 Full Quote Data"):
            st.json(quote)

elif data_type == "Technical Indicator":
    indicator = st.sidebar.selectbox(
        "Indicator",
        ["SMA", "EMA", "RSI", "MACD", "BBANDS", "STOCH", "ADX", "CCI", "AROON", "MFI"]
    )
    
    interval = st.sidebar.selectbox(
        "Interval",
        ["daily", "weekly", "monthly", "1min", "5min", "15min", "30min", "60min"]
    )
    
    time_period = st.sidebar.number_input(
        "Time Period",
        min_value=2,
        max_value=200,
        value=14
    )
    
    series_type = st.sidebar.selectbox(
        "Series Type",
        ["close", "open", "high", "low"]
    )
    
    if st.button(f"Fetch {indicator} Data", type="primary"):
        with st.spinner(f"Fetching {indicator} data..."):
            try:
                df = provider.get_technical_indicator(
                    symbol,
                    indicator,
                    interval=interval,
                    time_period=time_period,
                    series_type=series_type,
                    use_cache=use_cache
                )
                
                if df is not None and not df.empty:
                    st.session_state.indicator_data = df
                    st.session_state.indicator_name = indicator
                    st.success(f"✅ Fetched {indicator} data")
                else:
                    st.error("No indicator data returned")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # Display indicator data if available
    if "indicator_data" in st.session_state:
        df = st.session_state.indicator_data
        indicator_name = st.session_state.indicator_name
        
        # Plot indicator
        fig = go.Figure()
        
        # Add traces for each column in the indicator data
        for col in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[col],
                mode='lines',
                name=col
            ))
        
        fig.update_layout(
            title=f"{symbol} - {indicator_name} ({interval})",
            yaxis_title="Value",
            xaxis_title="Date/Time",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Show data table
        with st.expander("📊 View Raw Data"):
            st.dataframe(df)

# Rate limit status in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 🚦 Rate Limit Status")
st.sidebar.info("""
Free tier: 5 calls/minute
Premium: Higher limits available
""")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center;">
    <p>Data provided by <a href="https://www.alphavantage.co">Alpha Vantage</a> | 
    Enhanced with caching and rate limiting</p>
</div>
""", unsafe_allow_html=True) 