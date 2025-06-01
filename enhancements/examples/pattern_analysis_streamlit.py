#!/usr/bin/env python3
"""
Streamlit UI for Enhanced Pattern Analysis.

A simplified interface that integrates with the enhanced pattern analyzer.

Usage:
    ./venv_test/bin/streamlit run enhancements/examples/pattern_analysis_streamlit.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
from pathlib import Path
from datetime import datetime
import json

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Load .env file from project root
    env_path = Path(__file__).parent.parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"Loaded .env from {env_path}")
except ImportError:
    print("python-dotenv not installed, skipping .env load")

# Handle both API key naming conventions
if "ALPHA_VANTAGE_API_KEY" in os.environ and "ALPHAVANTAGE_API_KEY" not in os.environ:
    os.environ["ALPHAVANTAGE_API_KEY"] = os.environ["ALPHA_VANTAGE_API_KEY"]

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import the enhanced analyzer
from enhanced_pattern_analyzer import run_enhanced_analysis, EnhancedPatternDetector, OptionsRecommendationEngine
from advanced_options_analyzer import AdvancedOptionsAnalyzer
from enhancements.data_providers.alpha_provider import AlphaVantageProvider
from enhancements.data_providers.yahoo_provider import YahooProvider
from enhancements.data_access.cache import CacheManager
from enhancements.patterns.confidence import update_pattern_outcome

# Page configuration
st.set_page_config(
    page_title="FinBot Pattern Analysis",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []


def create_price_chart(df: pd.DataFrame, patterns: list):
    """Create an interactive price chart with pattern annotations."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=('Price & Patterns', 'Volume')
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # Add technical indicators
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
    for pattern in patterns:
        annotation_text = f"{pattern['pattern'].replace('_', ' ').title()}<br>Confidence: {pattern['confidence_score']:.1%}"
        
        # Add a marker at the pattern location
        if 'neckline' in pattern:
            fig.add_hline(
                y=pattern['neckline'],
                line_dash="dash",
                line_color="red" if pattern['type'] == 'bearish' else "green",
                annotation_text=annotation_text,
                row=1, col=1
            )
    
    # Volume
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
    
    # Update layout
    fig.update_layout(
        title='Technical Analysis',
        xaxis_title='Date',
        height=600,
        showlegend=True,
        xaxis_rangeslider_visible=False
    )
    
    return fig


def display_pattern_cards(patterns: list):
    """Display patterns as cards."""
    if not patterns:
        st.info("No patterns detected in the current timeframe.")
        return
    
    # Create columns for pattern cards
    cols = st.columns(min(3, len(patterns)))
    
    for i, pattern in enumerate(patterns[:3]):  # Show top 3 patterns
        with cols[i % 3]:
            # Card styling
            card_color = "#90EE90" if pattern['type'] == 'bullish' else "#FFB6C1"
            
            # Get pattern details
            pattern_name = pattern['pattern'].replace('_', ' ').title()
            
            # Try to get price info for better context
            if 'neckline' in pattern:
                price_info = f"Neckline: ${pattern['neckline']:.2f}"
            elif 'resistance' in pattern:
                price_info = f"Resistance: ${pattern['resistance']:.2f}"
            elif 'support' in pattern:
                price_info = f"Support: ${pattern['support']:.2f}"
            else:
                price_info = ""
            
            # Get pattern timing if available
            if 'end_idx' in pattern:
                days_ago = len(patterns[0].get('prices', [])) - pattern.get('end_idx', 0) if 'prices' in patterns[0] else pattern.get('end_idx', 0)
                timing = f"~{days_ago} bars ago" if days_ago > 0 else "Current"
            else:
                timing = "Recent"
            
            st.markdown(f"""
            <div style="
                background-color: {card_color};
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin-bottom: 10px;
                height: 180px;
            ">
                <h4 style="margin: 0; color: #333;">{pattern_name}</h4>
                <p style="margin: 5px 0; color: #555;">Type: <strong>{pattern['type'].upper()}</strong></p>
                <p style="margin: 5px 0; color: #555;">Confidence: <strong>{pattern['confidence_score']:.1%}</strong></p>
                <p style="margin: 5px 0; color: #555;">Win Rate: <strong>{pattern['historical_win_rate']:.1%}</strong></p>
                <p style="margin: 5px 0; color: #666; font-size: 0.9em;">{price_info}</p>
                <p style="margin: 5px 0; color: #666; font-size: 0.85em; font-style: italic;">{timing}</p>
            </div>
            """, unsafe_allow_html=True)


def display_options_strategies(recommendations: list):
    """Display options strategy recommendations."""
    if not recommendations:
        st.info("No options strategies recommended.")
        return
    
    # Check if this is a NO TRADE recommendation
    if recommendations[0].get('strategy_type') == 'NO TRADE':
        st.warning("⚠️ **NO TRADE RECOMMENDED**")
        st.write(f"**Reason:** {recommendations[0].get('description', 'Entry conditions not met')}")
        if 'detailed_explanation' in recommendations[0]:
            st.info(recommendations[0]['detailed_explanation'])
        return
    
    # Add educational content
    with st.expander("📚 Options Trading Basics", expanded=False):
        st.markdown("""
        ### Key Terms:
        - **Strike Price**: The price at which you can buy (call) or sell (put) the underlying stock
        - **Premium**: The cost of the option contract
        - **Expiration Date**: The last day the option can be exercised
        - **DTE (Days to Expiration)**: Number of days until option expires
        - **ATM (At The Money)**: Strike price near current stock price
        - **OTM (Out of The Money)**: Call above / Put below current price
        - **ITM (In The Money)**: Call below / Put above current price
        - **IV (Implied Volatility)**: Market's expectation of price movement
        
        ### Risk Levels:
        - **Low**: Limited, defined maximum loss
        - **Medium**: Moderate risk with manageable downside
        - **High**: Significant risk, requires careful management
        """)
    
    # Get current price from report (passed through st.session_state)
    current_price = st.session_state.get('current_price', 100)
    
    for i, strategy in enumerate(recommendations):
        # Skip NO TRADE recommendations
        if strategy.get('strategy_type') == 'NO TRADE':
            continue
            
        with st.expander(f"{strategy['strategy_type']} - {strategy.get('market_outlook', 'Neutral').title()} Strategy", expanded=(i==0)):
            
            # Overview section
            st.markdown("### 📋 Overview")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                risk_level = strategy.get('risk_level', 'medium')
                st.write(f"**Risk Level:** {risk_level.title()}")
                # Add risk indicator
                risk_color = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(risk_level, "⚪")
                st.write(f"Risk Indicator: {risk_color}")
            
            with col2:
                complexity = strategy.get('complexity', 'moderate')
                st.write(f"**Complexity:** {complexity.title()}")
            
            with col3:
                st.write(f"**Market Outlook:** {strategy.get('market_outlook', 'neutral').title()}")
            
            # Detailed explanation
            if 'detailed_explanation' in strategy:
                st.markdown("### 📖 Strategy Explanation")
                st.info(strategy['detailed_explanation'].strip())
            
            # NEW: Simple Action Plan
            st.markdown("### 🎯 Your Action Plan")
            
            # Calculate example strikes based on current price
            atm_strike = round(current_price / 5) * 5
            otm_5_percent = round(current_price * 1.05 / 5) * 5
            otm_10_percent = round(current_price * 1.10 / 5) * 5
            otm_put_5_percent = round(current_price * 0.95 / 5) * 5
            otm_put_10_percent = round(current_price * 0.90 / 5) * 5
            
            if "Bull Call Spread" in strategy['strategy_type']:
                strike_width = otm_5_percent - atm_strike
                entry_cost = 3.00  # Example net debit per share
                
                st.markdown(f"""
                ### 📍 ENTRY
                - **Buy:** ${atm_strike} Call
                - **Sell:** ${otm_5_percent} Call
                - **Cost:** ~${entry_cost * 100:.0f} per contract
                - **When:** Enter on next green day or pullback
                
                ### 🎯 TARGETS
                - **Target 1 (25% profit):** Stock at ${atm_strike + (strike_width * 0.25 + entry_cost):.2f} → Exit for ${entry_cost * 100 * 0.25:.0f} profit
                - **Target 2 (50% profit):** Stock at ${atm_strike + (strike_width * 0.50 + entry_cost):.2f} → Exit for ${entry_cost * 100 * 0.50:.0f} profit
                - **Target 3 (75% profit):** Stock at ${atm_strike + (strike_width * 0.75 + entry_cost):.2f} → Exit for ${entry_cost * 100 * 0.75:.0f} profit
                
                ### 🛑 STOP LOSS
                - **If stock drops below:** ${current_price - 1.5 * 10:.2f}
                - **Or if losing:** 50% of entry cost (${entry_cost * 100 * 0.5:.0f})
                - **Time stop:** Exit 2 weeks before expiration
                """)
                
            elif "Bull Put Spread" in strategy['strategy_type']:
                strike_width = otm_put_5_percent - otm_put_10_percent
                credit = 2.00  # Example net credit per share
                
                st.markdown(f"""
                ### 📍 ENTRY
                - **Sell:** ${otm_put_5_percent} Put
                - **Buy:** ${otm_put_10_percent} Put
                - **Credit:** ~${credit * 100:.0f} per contract (you receive this)
                - **When:** Enter on red days when IV is high
                
                ### 🎯 TARGETS
                - **Target 1 (25% profit):** Let time pass, buy back for ${credit * 100 * 0.75:.0f} → Keep ${credit * 100 * 0.25:.0f}
                - **Target 2 (50% profit):** Let time pass, buy back for ${credit * 100 * 0.50:.0f} → Keep ${credit * 100 * 0.50:.0f}
                - **Target 3 (75% profit):** Let time pass, buy back for ${credit * 100 * 0.25:.0f} → Keep ${credit * 100 * 0.75:.0f}
                
                ### 🛑 STOP LOSS
                - **If stock drops below:** ${otm_put_5_percent - (strike_width * 0.5):.2f}
                - **Or if losing:** Buy back if loss exceeds credit received
                - **Time stop:** Close at 21 days to expiration
                """)
                
            elif "Bear Call Spread" in strategy['strategy_type']:
                strike_width = otm_10_percent - otm_5_percent
                credit = 2.00  # Example net credit per share
                
                st.markdown(f"""
                ### 📍 ENTRY
                - **Sell:** ${otm_5_percent} Call
                - **Buy:** ${otm_10_percent} Call
                - **Credit:** ~${credit * 100:.0f} per contract (you receive this)
                - **When:** Enter on green days at resistance
                
                ### 🎯 TARGETS
                - **Target 1 (25% profit):** Let time pass, buy back for ${credit * 100 * 0.75:.0f} → Keep ${credit * 100 * 0.25:.0f}
                - **Target 2 (50% profit):** Let time pass, buy back for ${credit * 100 * 0.50:.0f} → Keep ${credit * 100 * 0.50:.0f}
                - **Target 3 (75% profit):** Let time pass, buy back for ${credit * 100 * 0.25:.0f} → Keep ${credit * 100 * 0.75:.0f}
                
                ### 🛑 STOP LOSS
                - **If stock rises above:** ${otm_5_percent + (strike_width * 0.5):.2f}
                - **Or if losing:** Buy back if loss exceeds credit received
                - **Time stop:** Close at 21 days to expiration
                """)
                
            elif "Bear Put Spread" in strategy['strategy_type']:
                strike_width = atm_strike - otm_put_5_percent
                entry_cost = 3.00  # Example net debit per share
                
                st.markdown(f"""
                ### 📍 ENTRY
                - **Buy:** ${atm_strike} Put
                - **Sell:** ${otm_put_5_percent} Put
                - **Cost:** ~${entry_cost * 100:.0f} per contract
                - **When:** Enter on failed breakouts or at resistance
                
                ### 🎯 TARGETS
                - **Target 1 (25% profit):** Stock at ${atm_strike - (strike_width * 0.25 + entry_cost):.2f} → Exit for ${entry_cost * 100 * 0.25:.0f} profit
                - **Target 2 (50% profit):** Stock at ${atm_strike - (strike_width * 0.50 + entry_cost):.2f} → Exit for ${entry_cost * 100 * 0.50:.0f} profit
                - **Target 3 (75% profit):** Stock at ${atm_strike - (strike_width * 0.75 + entry_cost):.2f} → Exit for ${entry_cost * 100 * 0.75:.0f} profit
                
                ### 🛑 STOP LOSS
                - **If stock rises above:** ${current_price + 1.5 * 10:.2f}
                - **Or if losing:** 50% of entry cost (${entry_cost * 100 * 0.5:.0f})
                - **Time stop:** Exit 2 weeks before expiration
                """)
            
            # Quick summary
            st.markdown("### 📋 Quick Summary")
            if "Bull" in strategy['strategy_type']:
                if "Call" in strategy['strategy_type']:
                    st.success("**BULLISH:** Pay to enter. Make money if stock goes UP. Exit at targets or stop loss.")
                else:
                    st.success("**BULLISH:** Get paid to enter. Keep money if stock stays UP. Buy back cheaper or let expire.")
            else:  # Bear
                if "Put" in strategy['strategy_type']:
                    st.success("**BEARISH:** Pay to enter. Make money if stock goes DOWN. Exit at targets or stop loss.")
                else:
                    st.success("**BEARISH:** Get paid to enter. Keep money if stock stays DOWN. Buy back cheaper or let expire.")
            
            # Risk reminder
            st.markdown("---")
            st.caption("⚠️ **Risk:** Never risk more than 1-2% of your account per trade. Options expire worthless if wrong.")


# Main UI
def main():
    st.title("📊 FinBot Enhanced Pattern Analysis")
    st.markdown("Advanced pattern detection with options strategy recommendations")
    
    # Sidebar
    with st.sidebar:
        st.header("Analysis Settings")
        
        # API Key check
        if "ALPHAVANTAGE_API_KEY" not in os.environ:
            st.warning("AlphaVantage API key not set")
            api_key = st.text_input("Enter API Key (optional):", type="password")
            if api_key:
                os.environ["ALPHAVANTAGE_API_KEY"] = api_key
        
        # Symbol input
        symbol = st.text_input("Stock Symbol:", value="AAPL").upper()
        
        # Timeframe selection
        st.subheader("📅 Timeframe Settings")
        timeframe = st.radio(
            "Analysis Timeframe:",
            ["Daily", "Weekly"],
            index=0,
            help="Daily: Best for short-term patterns and options\nWeekly: Better for longer-term trends"
        )
        
        period_options = {
            "Daily": {"30 days": 30, "60 days": 60, "90 days": 90, "180 days": 180},
            "Weekly": {"3 months": 90, "6 months": 180, "1 year": 365}
        }
        
        selected_period = st.selectbox(
            "Analysis Period:",
            list(period_options[timeframe].keys()),
            index=2 if timeframe == "Daily" else 1
        )
        period_days = period_options[timeframe][selected_period]
        
        # Data source
        data_source = st.radio(
            "Data Source:",
            ["Yahoo Finance", "AlphaVantage (if available)"],
            index=0
        )
        
        # Check if AlphaVantage is actually available
        alphavantage_available = "ALPHAVANTAGE_API_KEY" in os.environ and os.environ["ALPHAVANTAGE_API_KEY"]
        
        if data_source == "AlphaVantage (if available)" and not alphavantage_available:
            st.warning("⚠️ AlphaVantage API key not set. Will use Yahoo Finance instead.")
        
        use_alphavantage = data_source == "AlphaVantage (if available)" and alphavantage_available
        
        # Analysis button
        analyze_button = st.button("🔍 Run Analysis", type="primary", use_container_width=True)
        
        # Show analysis context
        st.markdown("---")
        st.info(f"""
        **Analysis Requirements:**
        - Pattern bias: Bullish/Bearish
        - ADX ≥ 20 (trend strength)
        - Weekly Close vs 20-SMA alignment
        - HV/IV analysis for strategy
        - ATR-based stop loss (1.5x)
        """)
        
        # History
        if st.session_state.analysis_history:
            st.markdown("---")
            st.subheader("Recent Analyses")
            for item in st.session_state.analysis_history[-5:]:
                st.text(f"{item['time']} - {item['symbol']}")
    
    # Main content
    if analyze_button and symbol:
        with st.spinner(f"Analyzing {symbol}..."):
            try:
                # Run standard analysis with advanced features
                report = run_enhanced_analysis(
                    symbol, 
                    use_alphavantage=use_alphavantage,
                    timeframe=timeframe.lower(),
                    period_days=period_days
                )
                
                if report:
                    # Save to history
                    st.session_state.analysis_history.append({
                        'time': datetime.now().strftime("%H:%M"),
                        'symbol': symbol,
                        'report': report
                    })
                    
                    # Save current price to session state for options display
                    st.session_state.current_price = report['current_price']
                    
                    # Try to get real-time quote for display
                    real_time_price = None
                    real_time_change = None
                    if use_alphavantage:
                        try:
                            provider = AlphaVantageProvider(CacheManager())
                            quote = provider.get_quote(symbol)
                            if quote:
                                real_time_price = quote.get('price')
                                real_time_change = quote.get('change_percent')
                                # Update session state with real-time price if available
                                if real_time_price:
                                    st.session_state.current_price = real_time_price
                        except:
                            pass
                    
                    # Display summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        if real_time_price:
                            st.metric(
                                "Real-Time Price", 
                                f"${real_time_price:.2f}",
                                delta=f"{real_time_change}%" if real_time_change else None
                            )
                        else:
                            st.metric("Current Price", f"${report['current_price']:.2f}")
                    with col2:
                        st.metric("Patterns Found", report['patterns_detected'])
                    with col3:
                        outlook = report['market_outlook']
                        if outlook == 'no_patterns':
                            outlook_icon = "❓"
                            outlook_text = "No Patterns"
                        elif outlook == 'bullish':
                            outlook_icon = "📈"
                            outlook_text = "Bullish"
                        elif outlook == 'bearish':
                            outlook_icon = "📉"
                            outlook_text = "Bearish"
                        else:
                            outlook_icon = "➖"
                            outlook_text = "Neutral"
                        st.metric("Market Outlook", f"{outlook_icon} {outlook_text}")
                    with col4:
                        entry_icon = "✅" if report.get('advanced_conditions', {}).get('entry_conditions_met', False) else "❌"
                        st.metric("Entry Signal", f"{entry_icon} {'GO' if report.get('advanced_conditions', {}).get('entry_conditions_met', False) else 'NO'}")
                    
                    # Add a note about data freshness
                    if real_time_price:
                        st.success("✅ Using real-time market data")
                    else:
                        st.info("ℹ️ Using end-of-day data")
                    
                    # Advanced conditions display
                    if 'advanced_conditions' in report:
                        with st.expander("🎯 Advanced Entry Conditions", expanded=True):
                            adv = report['advanced_conditions']
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.subheader("Pattern & ADX")
                                st.write(f"**Pattern Bias:** {adv['pattern_bias'].upper()}")
                                adx_val = adv['ADX']
                                adx_display = f"{adx_val:.1f}" if adx_val > 0 else "Insufficient data"
                                st.write(f"**ADX:** {adx_display} {'✅' if adv['ADX_condition_met'] else '❌'}")
                                if not adv['ADX_condition_met']:
                                    st.caption("Need ADX ≥ 20")
                            
                            with col2:
                                st.subheader("Weekly Trend")
                                st.write(f"**Weekly Close:** ${adv['weekly_close']:.2f}")
                                if adv['weekly_SMA_20']:
                                    st.write(f"**Weekly SMA(20):** ${adv['weekly_SMA_20']:.2f}")
                                    st.write(f"**Trend OK:** {'✅' if adv['weekly_trend_condition_met'] else '❌'}")
                                else:
                                    st.write("**Weekly SMA(20):** Calculating...")
                            
                            with col3:
                                st.subheader("Volatility")
                                indicators = report['technical_indicators']
                                st.write(f"**HV(60):** {indicators['HV_60']:.1f}%")
                                st.write(f"**HV(30):** {indicators['HV_30']:.1f}%")
                                if indicators['IV']:
                                    st.write(f"**IV:** {indicators['IV']:.1f}%")
                                    iv_hv_ratio = indicators['IV'] / indicators['HV_60'] if indicators['HV_60'] > 0 else 0
                                    st.write(f"**IV/HV:** {iv_hv_ratio:.2f}")
                                    
                                    # Show IV skew if available
                                    iv_data = indicators.get('IV_data', {})
                                    if iv_data and 'iv_skew' in iv_data:
                                        skew = iv_data['iv_skew']
                                        if skew > 5:
                                            st.caption("📉 Put skew - Bearish sentiment")
                                        elif skew < -5:
                                            st.caption("📈 Call skew - Bullish sentiment")
                                        else:
                                            st.caption("Balanced IV skew")
                            
                            # Entry decision with candlestick timing
                            if adv['entry_conditions_met']:
                                # Check candlestick timing
                                candlestick = adv.get('candlestick_timing', {})
                                if candlestick.get('timing') == 'confirmed':
                                    st.success(f"✅ All entry conditions met - Trade confirmed by {candlestick.get('pattern', 'candlestick')} pattern")
                                elif candlestick.get('timing') == 'wait':
                                    st.warning(f"⏳ Entry conditions met - {candlestick.get('description', 'Wait for candlestick confirmation')}")
                                else:
                                    st.success("✅ All entry conditions met - Trade recommended")
                            else:
                                missing = []
                                if not adv['ADX_condition_met']:
                                    missing.append("ADX < 20")
                                if not adv['weekly_trend_condition_met']:
                                    missing.append("Weekly trend not aligned")
                                if adv['pattern_bias'] == 'neutral':
                                    missing.append("No clear pattern bias")
                                st.error(f"❌ Entry conditions not met: {', '.join(missing)}")
                    
                    # Check for outlook conflicts
                    if 'pattern_based_outlook' in report and 'price_based_outlook' in report:
                        if report['pattern_based_outlook'] != report['price_based_outlook']:
                            st.warning(f"⚠️ **Divergence Alert**: Patterns suggest {report['pattern_based_outlook']} "
                                     f"outlook while price trend is {report['price_based_outlook']}. "
                                     f"This could indicate a potential reversal.")
                            
                            # Explain the divergence
                            with st.expander("Understanding the Divergence", expanded=True):
                                st.markdown(f"""
                                ### Pattern vs Price Analysis
                                
                                **Pattern Analysis**: {report['pattern_based_outlook'].upper()}
                                - Based on detected chart patterns
                                - Focuses on formation shapes and breakouts
                                - Forward-looking (predictive)
                                
                                **Price Trend**: {report['price_based_outlook'].upper()}
                                - Based on 20-day price movement
                                - Simple trend direction
                                - Backward-looking (historical)
                                
                                ### What This Means:
                                When patterns and price trends diverge, it often signals:
                                - **Potential Reversal**: The market may be setting up for a turn
                                - **Pattern Completion**: Chart patterns often form at trend reversals
                                - **Increased Volatility**: Be prepared for larger moves
                                
                                ### Recommended Action:
                                - Consider the pattern-based outlook for options strategies
                                - Use tighter risk management due to divergence
                                - Monitor for pattern confirmation/failure
                                """)
                    
                    # Show exit warnings if any
                    if 'exit_warnings' in report and report['exit_warnings']:
                        with st.expander("⚠️ Exit Warnings", expanded=True):
                            for warning in report['exit_warnings']:
                                st.warning(f"**{warning['pattern'].replace('_', ' ').title()}** detected on {warning['date'].strftime('%Y-%m-%d')}: {warning['description']}")
                                st.caption(f"Confidence: {warning['confidence']*100:.0f}% | Action: {warning['action'].replace('_', ' ')}")
                    
                    # Display analysis parameters
                    if 'analysis_parameters' in report:
                        with st.expander("📊 Analysis Parameters", expanded=False):
                            params = report['analysis_parameters']
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.write("**Timeframe:**", params['timeframe'].title())
                                st.write("**Data Points:**", params['data_points'])
                            
                            with col2:
                                st.write("**Period:**", f"{params['period_days']} days")
                                st.write("**Pattern Window:**", params['pattern_detection_window'])
                            
                            with col3:
                                st.write("**Date Range:**")
                                st.write(f"{params['date_range']['start']} to {params['date_range']['end']}")
                            
                            st.info(f"""
                            💡 **Why Our Analysis May Differ from TradingView:**
                            
                            1. **Timeframe**: We're analyzing {params['pattern_detection_window']} patterns vs TradingView's potential weekly/monthly view
                            2. **Indicators**: Different calculation methods or periods
                            3. **Pattern Recognition**: Our AI focuses on specific chart patterns
                            4. **Options Focus**: Our recommendations are optimized for {params['options_timeline']}
                            
                            For best results, compare multiple timeframes and sources before trading.
                            """)
                    
                    # Create tabs
                    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📈 Chart", "🎯 Patterns", "💡 Options", "📊 Indicators", "📰 News", "⏰ 4H Timing"])
                    
                    with tab1:
                        # Fetch data for charting
                        cache_manager = CacheManager()
                        if use_alphavantage:
                            provider = AlphaVantageProvider(cache_manager)
                            df = provider.get_daily(symbol, outputsize="compact")
                        else:
                            provider = YahooProvider(cache_manager)
                            df = provider.get_ohlcv(symbol, period="3mo", interval="1d")
                            if df is not None and 'Date' in df.columns:
                                df.set_index('Date', inplace=True)
                        
                        if df is not None:
                            # Calculate indicators
                            df["EMA_9"] = df["Close"].ewm(span=9, adjust=False).mean()
                            df["EMA_21"] = df["Close"].ewm(span=21, adjust=False).mean()
                            
                            # Create chart
                            fig = create_price_chart(df.tail(60), report['patterns'])
                            st.plotly_chart(fig, use_container_width=True)
                    
                    with tab2:
                        st.subheader("Detected Patterns")
                        display_pattern_cards(report['patterns'])
                        
                        # Pattern outcome tracking
                        if report['patterns']:
                            st.markdown("---")
                            st.subheader("Track Pattern Outcomes")
                            col1, col2, col3 = st.columns([2, 1, 1])
                            
                            with col1:
                                selected_pattern = st.selectbox(
                                    "Select pattern:",
                                    [f"{p['pattern']} ({p['type']})" for p in report['patterns']]
                                )
                            
                            with col2:
                                outcome = st.radio("Outcome:", ["Success", "Failure"])
                            
                            with col3:
                                if st.button("Record"):
                                    # Parse pattern info
                                    pattern_name = selected_pattern.split(" (")[0]
                                    pattern_type = selected_pattern.split(" (")[1].rstrip(")")
                                    
                                    # Update outcome
                                    update_pattern_outcome(
                                        pattern_name,
                                        pattern_type,
                                        outcome == "Success"
                                    )
                                    st.success("Outcome recorded!")
                    
                    with tab3:
                        st.subheader("Options Strategy Recommendations")
                        display_options_strategies(report['options_recommendations'])
                    
                    with tab4:
                        st.subheader("Technical Indicators")
                        indicators = report['technical_indicators']
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("EMA 9", f"{indicators['EMA_9']:.2f}")
                            st.metric("EMA 21", f"{indicators['EMA_21']:.2f}")
                            
                            # EMA Cross
                            if indicators['EMA_9'] > indicators['EMA_21']:
                                st.success("✅ Bullish EMA Cross")
                            else:
                                st.error("❌ Bearish EMA Cross")
                        
                        with col2:
                            rsi_value = indicators.get('RSI', 50)
                            # Handle NaN or invalid RSI values
                            if pd.isna(rsi_value) or not isinstance(rsi_value, (int, float)):
                                rsi_value = 50  # Default to neutral
                            
                            st.metric("RSI", f"{rsi_value:.2f}")
                            
                            # RSI Status
                            if rsi_value > 70:
                                st.warning("⚠️ Overbought")
                            elif rsi_value < 30:
                                st.warning("⚠️ Oversold")
                            else:
                                st.info("✓ Neutral")
                            
                            macd_value = indicators.get('MACD', 0)
                            macd_signal_value = indicators.get('MACD_Signal', 0)
                            
                            # Handle NaN values for MACD
                            if pd.isna(macd_value):
                                macd_value = 0
                            if pd.isna(macd_signal_value):
                                macd_signal_value = 0
                            
                            st.metric("MACD", f"{macd_value:.4f}")
                            st.metric("MACD Signal", f"{macd_signal_value:.4f}")
                    
                    with tab5:
                        st.subheader("News Sentiment Analysis")
                        
                        adv = report.get('advanced_conditions', {})
                        news = adv.get('news_sentiment', {})
                        
                        if news and news.get('articles_analyzed', 0) > 0:
                            # Sentiment score with visual indicator
                            sentiment_score = news.get('sentiment_score', 0)
                            relevance = news.get('relevance_score', 0)
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                # Color code sentiment
                                if sentiment_score > 0.3:
                                    st.success(f"Sentiment: {sentiment_score:.2f} 😊")
                                elif sentiment_score < -0.3:
                                    st.error(f"Sentiment: {sentiment_score:.2f} 😟")
                                else:
                                    st.info(f"Sentiment: {sentiment_score:.2f} 😐")
                            
                            with col2:
                                st.metric("Relevance", f"{relevance:.2f}")
                            
                            with col3:
                                st.metric("Articles", news.get('articles_analyzed', 0))
                            
                            # Show how it affects entry
                            if sentiment_score < -0.5:
                                st.error("⚠️ Very negative news sentiment - Entry blocked")
                            elif sentiment_score > 0.3:
                                st.success("✅ Positive news sentiment - Entry supported")
                            
                            # Recent articles if available
                            if 'recent_articles' in news and news['recent_articles']:
                                st.markdown("### Recent Headlines")
                                for article in news['recent_articles'][:3]:
                                    with st.expander(article.get('title', 'No title')):
                                        st.write(f"Sentiment: {article.get('sentiment', 0):.2f}")
                                        st.write(f"Relevance: {article.get('relevance', 0):.2f}")
                                        st.write(f"Time: {article.get('time', 'Unknown')}")
                        else:
                            st.info("No news sentiment data available. Using technical analysis only.")
                    
                    with tab6:
                        st.subheader("4-Hour Timing Analysis")
                        
                        four_hour_data = report.get('four_hour_timing')
                        
                        if four_hour_data and four_hour_data != None:
                            # Overall timing recommendation
                            timing = four_hour_data.get('timing', 'wait')
                            confidence = four_hour_data.get('confidence', 0.5)
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                # Timing signal with color
                                timing_color = {
                                    'immediate': '🟢',
                                    'soon': '🟡', 
                                    'wait': '🔴',
                                    'unavailable': '⚫'
                                }.get(timing, '⚪')
                                st.metric("Entry Timing", f"{timing_color} {timing.upper()}")
                            
                            with col2:
                                st.metric("Confidence", f"{confidence:.0%}")
                            
                            with col3:
                                st.metric("4H Trend", four_hour_data.get('4h_trend', 'N/A').upper())
                            
                            # Recommendation
                            st.info(f"**Recommendation:** {four_hour_data.get('recommendation', 'No recommendation available')}")
                            
                            # Entry zones
                            if 'entry_zones' in four_hour_data and four_hour_data['entry_zones']:
                                st.markdown("### 📍 Entry Zones")
                                for zone in four_hour_data['entry_zones']:
                                    st.write(f"• **{zone['type'].replace('_', ' ').title()}** at ${zone['level']:.2f}")
                                    st.caption(f"  {zone['action']}")
                            
                            # Support/Resistance levels
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                if 'support_levels' in four_hour_data and four_hour_data['support_levels']:
                                    st.markdown("### 🛡️ Support Levels")
                                    for level in four_hour_data['support_levels']:
                                        st.write(f"• ${level:.2f}")
                            
                            with col2:
                                if 'resistance_levels' in four_hour_data and four_hour_data['resistance_levels']:
                                    st.markdown("### 🚧 Resistance Levels")
                                    for level in four_hour_data['resistance_levels']:
                                        st.write(f"• ${level:.2f}")
                            
                            # Entry reasons
                            if 'entry_reasons' in four_hour_data and four_hour_data['entry_reasons']:
                                st.markdown("### ✅ Entry Signals")
                                for reason in four_hour_data['entry_reasons']:
                                    st.write(f"• {reason}")
                            
                            # Entry score visualization
                            if 'entry_score' in four_hour_data:
                                score = four_hour_data['entry_score']
                                st.markdown("### 📊 Entry Score")
                                
                                # Create a progress bar
                                progress_value = min(score / 10, 1.0)  # Normalize to 0-1
                                st.progress(progress_value)
                                st.caption(f"Score: {score}/10 - {'Strong' if score >= 6 else 'Moderate' if score >= 4 else 'Weak'} signal")
                            
                            # Suggested stop
                            if 'suggested_stop' in four_hour_data:
                                st.markdown("### 🛑 Suggested Stop Loss")
                                st.write(f"${four_hour_data['suggested_stop']:.2f} (based on 4H ATR)")
                            
                            # How to use this info
                            with st.expander("📚 How to Use 4-Hour Timing", expanded=False):
                                st.markdown("""
                                ### Understanding 4-Hour Timing
                                
                                The 4-hour chart provides more granular entry timing to complement daily patterns:
                                
                                **Timing Signals:**
                                - 🟢 **IMMEDIATE**: Strong setup, enter on next 4H candle
                                - 🟡 **SOON**: Setup forming, prepare to enter within 1-2 candles
                                - 🔴 **WAIT**: Not ideal, wait for better setup
                                
                                **Entry Zones:**
                                - **Support Bounce**: Buy when price tests support level
                                - **EMA Pullback**: Buy when price pulls back to moving average
                                - **Resistance Rejection**: Short when price fails at resistance
                                
                                **Best Practices:**
                                1. Wait for daily pattern confirmation first
                                2. Use 4H timing to fine-tune your entry
                                3. Enter during market hours for better fills
                                4. Set alerts at key 4H levels
                                5. Use 4H ATR for more precise stops
                                
                                **Note**: Each 4H bar represents ~2 trading days of behavior
                                """)
                        else:
                            st.info("4-hour timing analysis not available. This could be due to:")
                            st.write("• Limited intraday data availability")
                            st.write("• Weekend or after-hours analysis")
                            st.write("• Data provider limitations")
                            
                            st.caption("💡 Tip: 4-hour analysis works best during market hours with liquid stocks")
                    
                    # Export option
                    st.markdown("---")
                    if st.button("📥 Export Report"):
                        report_json = json.dumps(report, indent=2, default=str)
                        st.download_button(
                            label="Download JSON Report",
                            data=report_json,
                            file_name=f"{symbol}_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                else:
                    st.error("Analysis failed. Please try again.")
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.exception(e)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>FinBot Pattern Analysis | Not Financial Advice</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main() 