#!/usr/bin/env python3
"""
Interactive Pattern Analysis and Options Recommendation UI.

This Streamlit app provides a comprehensive interface for:
- Fetching data from multiple sources (AlphaVantage, Yahoo Finance)
- Running pattern detection with confidence scoring
- Generating options recommendations
- Visualizing patterns and technical indicators

Usage:
    ./venv_test/bin/streamlit run enhancements/examples/pattern_analysis_ui.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import the enhanced pattern analyzer
from pattern_analysis_with_alphavantage import EnhancedPatternAnalyzer

# Import additional components
from enhancements.data_providers.yahoo_provider import YahooProvider
from enhancements.patterns.confidence import PatternConfidenceEngine, update_pattern_outcome


# Page configuration
st.set_page_config(
    page_title="FinBot Pattern Analysis & Options Recommendations",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}
if 'historical_analyses' not in st.session_state:
    st.session_state.historical_analyses = []


def initialize_analyzer():
    """Initialize the pattern analyzer."""
    if st.session_state.analyzer is None:
        try:
            st.session_state.analyzer = EnhancedPatternAnalyzer()
            return True
        except Exception as e:
            st.error(f"Failed to initialize analyzer: {str(e)}")
            return False
    return True


def create_candlestick_chart(df: pd.DataFrame, patterns: list = None):
    """Create an interactive candlestick chart with pattern annotations."""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=('Price Action & Patterns', 'Volume', 'Technical Indicators')
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='OHLC'
        ),
        row=1, col=1
    )
    
    # Add EMAs if available
    if 'EMA_9' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['EMA_9'],
                name='EMA 9',
                line=dict(color='orange', width=1)
            ),
            row=1, col=1
        )
    
    if 'EMA_21' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['EMA_21'],
                name='EMA 21',
                line=dict(color='blue', width=1)
            ),
            row=1, col=1
        )
    
    # Add pattern annotations
    if patterns:
        for pattern in patterns:
            if 'start_date' in pattern and 'end_date' in pattern:
                fig.add_vrect(
                    x0=pattern['start_date'],
                    x1=pattern['end_date'],
                    fillcolor="LightSalmon" if 'bearish' in pattern.get('type', '') else "LightGreen",
                    opacity=0.3,
                    layer="below",
                    line_width=0,
                    annotation_text=pattern.get('pattern', ''),
                    annotation_position="top left",
                    row=1, col=1
                )
    
    # Volume bars
    colors = ['red' if row['Open'] > row['Close'] else 'green' for idx, row in df.iterrows()]
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['Volume'],
            name='Volume',
            marker_color=colors
        ),
        row=2, col=1
    )
    
    # RSI if available
    if 'RSI' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['RSI'],
                name='RSI',
                line=dict(color='purple', width=2)
            ),
            row=3, col=1
        )
        
        # Add RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    
    # Update layout
    fig.update_layout(
        title='Technical Analysis Chart',
        xaxis_title='Date',
        yaxis_title='Price',
        height=800,
        showlegend=True,
        xaxis_rangeslider_visible=False
    )
    
    return fig


def display_pattern_confidence(patterns: list):
    """Display pattern confidence metrics."""
    if not patterns:
        st.info("No patterns detected in the current analysis.")
        return
    
    # Create a DataFrame for better display
    pattern_data = []
    for pattern in patterns:
        pattern_data.append({
            'Pattern': pattern.get('name', 'Unknown'),
            'Type': pattern.get('type', 'Unknown'),
            'Confidence': f"{pattern.get('confidence_score', 0):.1%}",
            'Historical Win Rate': f"{pattern.get('historical_win_rate', 0):.1%}",
            'Significance': pattern.get('significance', 'Medium')
        })
    
    df = pd.DataFrame(pattern_data)
    
    # Use column configuration for better display
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Pattern": st.column_config.TextColumn("Pattern Name", width="medium"),
            "Type": st.column_config.TextColumn("Direction", width="small"),
            "Confidence": st.column_config.TextColumn("Confidence Score", width="small"),
            "Historical Win Rate": st.column_config.TextColumn("Win Rate", width="small"),
            "Significance": st.column_config.TextColumn("Significance", width="small"),
        }
    )
    
    # Show average metrics
    if pattern_data:
        avg_confidence = sum(p.get('confidence_score', 0) for p in patterns) / len(patterns)
        avg_win_rate = sum(p.get('historical_win_rate', 0) for p in patterns) / len(patterns)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Confidence", f"{avg_confidence:.1%}")
        with col2:
            st.metric("Average Win Rate", f"{avg_win_rate:.1%}")
        with col3:
            strength = "Strong" if avg_confidence > 0.7 else "Moderate" if avg_confidence > 0.5 else "Weak"
            st.metric("Signal Strength", strength)


def display_options_strategies(strategies: list):
    """Display recommended options strategies."""
    if not strategies:
        st.info("No specific options strategies recommended based on current patterns.")
        return
    
    for i, strategy in enumerate(strategies, 1):
        with st.expander(f"Strategy {i}: {strategy.get('strategy_type', 'Unknown')}", expanded=True):
            st.write(f"**Description:** {strategy.get('description', 'N/A')}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                risk_level = strategy.get('risk_level', 'Unknown')
                risk_color = "🟢" if risk_level == "low" else "🟡" if risk_level == "medium" else "🔴"
                st.write(f"**Risk Level:** {risk_color} {risk_level.capitalize()}")
            
            with col2:
                if strategy.get('potential_profit'):
                    st.write(f"**Potential Profit:** {strategy['potential_profit']}")
                else:
                    st.write("**Potential Profit:** N/A")
            
            with col3:
                if strategy.get('max_loss'):
                    st.write(f"**Maximum Loss:** {strategy['max_loss']}")
                else:
                    st.write("**Maximum Loss:** N/A")
            
            if strategy.get('breakeven'):
                st.write(f"**Breakeven:** {strategy['breakeven']}")
            
            # Add implementation details if available
            if strategy.get('implementation'):
                st.write("**Implementation Details:**")
                st.code(strategy['implementation'], language='text')


# Main UI
def main():
    st.title("🎯 FinBot Pattern Analysis & Options Recommendations")
    st.markdown("""
    This advanced analysis tool combines:
    - **Multiple data sources** (AlphaVantage, Yahoo Finance)
    - **Pattern detection** with confidence scoring
    - **Options strategy recommendations**
    - **Historical performance tracking**
    """)
    
    # Check for API key
    if "ALPHAVANTAGE_API_KEY" not in os.environ:
        st.warning("⚠️ AlphaVantage API key not set. Some features may be limited.")
        with st.expander("Set API Key"):
            api_key = st.text_input("Enter AlphaVantage API key:", type="password")
            if st.button("Set API Key"):
                if api_key:
                    os.environ["ALPHAVANTAGE_API_KEY"] = api_key
                    st.success("API key set for this session!")
                    st.rerun()
    
    # Initialize analyzer
    if not initialize_analyzer():
        st.error("Failed to initialize the analysis system. Please check your configuration.")
        return
    
    # Sidebar controls
    with st.sidebar:
        st.header("📊 Analysis Configuration")
        
        # Symbol input
        symbol = st.text_input("Stock Symbol", value="AAPL", help="Enter the stock ticker symbol").upper()
        
        # Analysis period
        period_days = st.slider("Analysis Period (days)", min_value=30, max_value=365, value=90, step=30)
        
        # Data source selection
        data_source = st.radio(
            "Data Source",
            ["AlphaVantage", "Yahoo Finance", "Auto (AlphaVantage → Yahoo)"],
            index=2,
            help="Select data provider or use Auto for fallback"
        )
        
        # Pattern sensitivity
        st.subheader("Pattern Detection Settings")
        min_confidence = st.slider(
            "Minimum Confidence Threshold",
            min_value=0.3,
            max_value=0.9,
            value=0.5,
            step=0.1,
            help="Only show patterns above this confidence level"
        )
        
        # Run analysis button
        analyze_button = st.button("🔍 Run Analysis", type="primary", use_container_width=True)
        
        # Additional options
        st.subheader("Additional Options")
        show_raw_data = st.checkbox("Show Raw Data", value=False)
        export_report = st.checkbox("Export Report", value=False)
    
    # Main content area
    if analyze_button:
        with st.spinner(f"Analyzing {symbol}..."):
            try:
                # Run analysis
                report = st.session_state.analyzer.analyze_symbol(symbol, period_days)
                
                # Store results
                st.session_state.analysis_results[symbol] = report
                st.session_state.historical_analyses.append({
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'report': report
                })
                
                # Check for errors
                if 'error' in report:
                    st.error(f"Analysis failed: {report['error']}")
                    return
                
                # Display results in tabs
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "📈 Chart & Patterns",
                    "🎯 Pattern Analysis",
                    "💡 Options Strategies",
                    "📊 Market Analysis",
                    "📋 Full Report"
                ])
                
                with tab1:
                    st.subheader("Technical Chart with Pattern Overlays")
                    
                    # Get the data for charting
                    try:
                        df = st.session_state.analyzer.fetch_enhanced_data(symbol, period_days)
                        fig = create_candlestick_chart(df, report.get('patterns_detected', []))
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Failed to create chart: {str(e)}")
                
                with tab2:
                    st.subheader("Detected Patterns & Confidence Analysis")
                    
                    patterns = report.get('patterns_detected', [])
                    # Filter by minimum confidence
                    filtered_patterns = [p for p in patterns if p.get('confidence_score', 0) >= min_confidence]
                    
                    if filtered_patterns:
                        display_pattern_confidence(filtered_patterns)
                        
                        # Pattern outcome tracking
                        st.subheader("Track Pattern Outcomes")
                        st.info("Help improve predictions by tracking pattern outcomes")
                        
                        selected_pattern = st.selectbox(
                            "Select pattern to track",
                            [f"{p['name']} ({p['type']})" for p in filtered_patterns]
                        )
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            outcome = st.radio("Outcome", ["Successful", "Failed"])
                        with col2:
                            if st.button("Record Outcome"):
                                # Extract pattern details
                                pattern_idx = [f"{p['name']} ({p['type']})" for p in filtered_patterns].index(selected_pattern)
                                pattern = filtered_patterns[pattern_idx]
                                
                                # Update outcome
                                update_pattern_outcome(
                                    pattern['name'],
                                    'bullish' if 'bullish' in pattern['type'] else 'bearish',
                                    outcome.lower() == 'successful'
                                )
                                st.success("Outcome recorded! This will improve future predictions.")
                    else:
                        st.info(f"No patterns detected with confidence ≥ {min_confidence:.0%}")
                
                with tab3:
                    st.subheader("Recommended Options Strategies")
                    strategies = report.get('options_recommendations', [])
                    display_options_strategies(strategies)
                    
                    # Risk analysis
                    if strategies:
                        st.subheader("Risk Analysis Summary")
                        risk_levels = [s.get('risk_level', 'unknown') for s in strategies]
                        risk_counts = pd.Series(risk_levels).value_counts()
                        
                        fig = go.Figure(data=[
                            go.Bar(x=risk_counts.index, y=risk_counts.values)
                        ])
                        fig.update_layout(
                            title="Strategy Risk Distribution",
                            xaxis_title="Risk Level",
                            yaxis_title="Count"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                with tab4:
                    st.subheader("Market Analysis Summary")
                    market = report.get('market_analysis', {})
                    
                    if market:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            trend = market.get('trend', 'Unknown')
                            trend_icon = "📈" if trend == 'bullish' else "📉" if trend == 'bearish' else "➡️"
                            st.metric("Market Trend", f"{trend_icon} {trend.capitalize()}")
                        
                        with col2:
                            sentiment = market.get('sentiment', 'Unknown')
                            sentiment_icon = "😊" if sentiment == 'positive' else "😟" if sentiment == 'negative' else "😐"
                            st.metric("Sentiment", f"{sentiment_icon} {sentiment.capitalize()}")
                        
                        with col3:
                            confidence = report.get('confidence_metrics', {})
                            strength = confidence.get('recommendation_strength', 'Unknown')
                            st.metric("Signal Strength", strength.upper())
                        
                        # Key levels
                        if market.get('key_levels'):
                            st.subheader("Key Price Levels")
                            levels = market['key_levels']
                            df_levels = pd.DataFrame([
                                {"Level": k, "Price": f"${v:.2f}"} for k, v in levels.items()
                            ])
                            st.table(df_levels)
                        
                        # Recommendation
                        if market.get('recommendation'):
                            st.subheader("Recommendation")
                            st.info(market['recommendation'])
                
                with tab5:
                    st.subheader("Complete Analysis Report")
                    
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Symbol", report['symbol'])
                    with col2:
                        st.metric("Current Price", f"${report['current_price']:.2f}")
                    with col3:
                        st.metric("Patterns Found", len(report.get('patterns_detected', [])))
                    with col4:
                        st.metric("Strategies", len(report.get('options_recommendations', [])))
                    
                    # Full JSON report
                    with st.expander("View Full JSON Report"):
                        st.json(report)
                    
                    # Export options
                    if export_report:
                        report_json = json.dumps(report, indent=2)
                        st.download_button(
                            label="📥 Download Report (JSON)",
                            data=report_json,
                            file_name=f"{symbol}_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                
                # Show raw data if requested
                if show_raw_data:
                    with st.expander("View Raw Data"):
                        df = st.session_state.analyzer.fetch_enhanced_data(symbol, period_days)
                        st.dataframe(df)
                
            except Exception as e:
                st.error(f"Analysis error: {str(e)}")
                st.exception(e)
    
    # Display cached results if available
    elif symbol in st.session_state.analysis_results:
        st.info(f"Showing cached results for {symbol}. Click 'Run Analysis' to refresh.")
        report = st.session_state.analysis_results[symbol]
        
        # Display cached results (similar structure as above)
        # ... (abbreviated for brevity)
    
    # Historical analyses
    with st.sidebar:
        if st.session_state.historical_analyses:
            st.subheader("📜 Analysis History")
            for analysis in st.session_state.historical_analyses[-5:]:  # Show last 5
                st.text(f"{analysis['timestamp'].strftime('%H:%M')} - {analysis['symbol']}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center;">
        <p>FinBot Enhanced Pattern Analysis | Powered by AlphaVantage & Yahoo Finance</p>
        <p>Remember: This is not financial advice. Always do your own research.</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main() 