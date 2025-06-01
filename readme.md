# FinBot - Technical Analysis and Market Pattern Detection System

## Overview
FinBot is an advanced technical analysis system that combines traditional technical indicators with pattern recognition and LLM-powered market analysis. The system uses a state graph architecture to process market data through multiple specialized nodes, each responsible for different aspects of market analysis.

## System Architecture

### Core Components

1. **Data Collection (API Node)**
   - Fetches OHLCV data from Yahoo Finance
   - Configurable lookback period (default: 100 days)
   - Daily interval data
   - Error handling and data validation

2. **Technical Analysis Pipeline**
   - **Analyze Node**: Short-term trend analysis
   - **Indicator Node**: Technical indicator calculations
   - **Pattern Detection Nodes**: Specialized pattern recognition

3. **Pattern Detection System**
   - Double Pattern Detector
   - Triple Pattern Detector
   - Head & Shoulders Pattern Detector
   - Wedge Pattern Detector
   - Pennant Pattern Detector
   - Flag Pattern Detector
   - Triangle Pattern Detector

4. **LLM Integration**
   - Market analysis synthesis
   - Pattern interpretation
   - Market direction prediction

### Technical Indicators

The system calculates and monitors:
- EMA (9, 21 periods)
- RSI (14 periods)
- MACD (12, 26, 9)
- VWAP
- Price trends
- Volume analysis

### Pattern Detection

#### Double Patterns
- Double Top/Bottom detection
- Neckline calculation
- Breakout confirmation

#### Triple Patterns
- Triple Top/Bottom detection
- Resistance/Support levels
- Pattern completion validation

#### Head & Shoulders
- Prior trend analysis
- Pattern component identification
- Neckline calculation
- Breakout confirmation

#### Wedge Patterns
- Linear regression analysis
- Convergence calculation
- Pattern completion detection

#### Pennant Patterns
- Pole movement analysis
- Trendline slope calculation
- Continuation pattern validation

#### Flag Patterns
- Parallel trendline analysis
- Continuation pattern detection
- Volume confirmation

#### Triangle Patterns
- Symmetrical/Ascending/Descending detection
- Breakout point calculation
- Pattern completion validation

## Data Flow

1. **Data Collection**
   ```
   API Node → OHLCV Data → Analyze Node
   ```

2. **Technical Analysis**
   ```
   Analyze Node → Trend Analysis → Indicator Node
   ```

3. **Pattern Detection**
   ```
   Indicator Node → Technical Indicators → Pattern Detection Nodes
   ```

4. **LLM Analysis**
   ```
   Pattern Detection → Pattern Signals → LLM Node → Market Analysis
   ```

## State Management

The system uses a state graph architecture where:
- Each node enriches the state with its analysis
- Data consistency is maintained across nodes
- Intermediate calculations are preserved
- Pattern detection details are tracked

## Dependencies

- Python 3.x
- pandas
- numpy
- yfinance
- langgraph
- openai
- scipy
- matplotlib
- streamlit (for UI)

## Environment Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Enter a stock symbol and analysis parameters
3. View the comprehensive market analysis

## Data Curation Process

1. **Raw Data Collection**
   - OHLCV data fetching
   - Data validation
   - Time period standardization

2. **Technical Analysis**
   - Indicator calculations
   - Trend analysis
   - Volume analysis

3. **Pattern Detection**
   - Pattern identification
   - Confirmation signals
   - Breakout detection

4. **LLM Preparation**
   - Data structuring
   - Pattern summarization
   - Market context compilation

## Future Modifications

This system is designed to be modular and extensible. Key areas for modification include:

1. **Pattern Detection**
   - Add new pattern types
   - Modify detection algorithms
   - Adjust confirmation thresholds

2. **Technical Indicators**
   - Add new indicators
   - Modify calculation methods
   - Adjust parameters

3. **LLM Integration**
   - Modify prompt structure
   - Add new analysis aspects
   - Enhance reasoning capabilities

4. **Data Sources**
   - Add alternative data providers
   - Implement real-time data feeds
   - Add historical data storage

## Contributing

When modifying the system:
1. Document all changes
2. Maintain the state graph architecture
3. Follow the existing pattern detection methodology
4. Update the README with new features
5. Add appropriate error handling
6. Include unit tests for new functionality

## Development Rules

**Important**: This project follows strict development rules to ensure maintainability and reversibility of changes. All new features and modifications must adhere to these rules.

### Key Principles:
1. **All enhancements go in the `enhancements/` folder** - Never modify existing code
2. **Protected folders: `myfirstfinbot/`** - Do NOT modify any files in protected folders
3. **No mutation of public APIs** - Use wrappers, subclasses, or composition
4. **Mandatory caching layer** - All external API calls must check cache first
5. **80% test coverage minimum** - All new code requires comprehensive tests
6. **Environment-based configuration** - No hardcoded secrets or API keys

### Quick Reference:
- New data providers: `enhancements/data_providers/`
- Technical indicators: `enhancements/technical_indicators/`
- Pattern enhancements: `enhancements/patterns/`
- Options/Greeks: `enhancements/options/`
- Documentation: `enhancements/docs/`

**For complete development rules, see [DEVELOPMENT_RULES.md](DEVELOPMENT_RULES.md)**

## License

[Your License Here]

## Contact

[Your Contact Information]
