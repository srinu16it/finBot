# FinBot Development Rules

## Overview
These rules ensure that all enhancements to the FinBot system are modular, maintainable, and reversible. The core principle is to never modify existing code directly, instead extending functionality through a dedicated enhancements folder.

## Core Development Rules

### 1. Folder Policy
- **Rule**: Place _all_ new or modified code only inside the top-level folder `enhancements/`
- **Rationale**: Keeps original code pristine and makes rollbacks trivial
- **Implementation**: Never edit, move, or delete any existing files outside this folder

### 2. Protected Folders
- **Rule**: Do NOT touch or modify the `myfirstfinbot` folder and its subfolders
- **Scope**: This includes all files within `myfirstfinbot/` and any subdirectories
- **Rationale**: This folder contains protected legacy code that must remain untouched
- **Enforcement**: Any modifications to files in this folder will be rejected in code review

### 3. No Mutation of Public API
- **Rule**: Never change the signature or internal logic of existing functions/classes
- **Extension Methods**:
  - Create wrapper functions
  - Use subclasses
  - Implement composition helpers
- **Location**: All extensions must reside in `enhancements/`

### 4. Data Providers
- **Yahoo Finance**: 
  - Location: `enhancements/data_providers/yahoo_provider.py`
  - Implementation: Via yfinance library
- **AlphaVantage**:
  - Location: `enhancements/data_providers/alpha_provider.py`
  - API Key: Read from environment variable `ALPHAVANTAGE_API_KEY`
  - Rate Limit: ≤ 5 calls per minute (free-tier limit)
  - Also has premuim subscription hence leverage the API in some cases for real time and options data

### 5. Caching Layer
- **Implementation**: DuckDB cache in `enhancements/data_access/cache.py`
- **Usage**: Every provider must call the cache helper before making external API requests
- **Benefits**: Reduces API calls, improves performance, enables offline development

### 6. Technical Indicators
- **Location**: `enhancements/technical_indicators/`
- **Examples**: Accurate Wilder ATR implementation
- **Important**: Existing quick ATR approximations must remain untouched

### 7. Pattern Edge Engine
- **Storage**: Statistical win-rates in DuckDB
- **Interface**: `get_pattern_edge()` helper
- **Location**: `enhancements/patterns/confidence.py`

### 8. Volatility & Greeks
- **Location**: `enhancements/options/`
- **Rule**: Option-chain processing must be isolated
- **Important**: Do NOT pull option data in Streamlit layer directly

### 9. Environment & Secrets
- **Method**: python-dotenv (.env file)
- **Rules**:
  - Never hard-code secrets
  - Never commit secrets to VCS
  - Use environment variables for all sensitive data

### 10. Testing Requirements
- **Coverage**: ≥ 80% for all new functionality
- **Location**: `tests/` mirroring package structure
- **Framework**: pytest

### 11. Style & Linting
- **Standards**: PEP-8 compliant
- **Formatter**: black (line length 88)
- **Docstrings**: Google-style
- **Linter**: ruff on CI

### 12. Dependencies
- **Location**: requirements.txt
- **Section**: Clearly commented "Enhancements" section
- **Policy**: Keep new dependencies minimal
- **Allowed**: duckdb, pandas_ta

### 13. Documentation
- **Location**: `enhancements/docs/`
- **Format**: Markdown
- **Content**: Purpose and usage for each new module

### 14. Safety Checks
Before any pull request:
- Run `python -m pytest`
- Run `black --check`
- Run `ruff`
- Ensure CI passes

### 15. Rollback Plan
- **Method**: Simple folder delete
- **Requirement**: No global state mutation outside `enhancements/`
- **Result**: Instant reversion to original functionality

### 16. License & Attribution
- **Respect**: All data-provider Terms of Service
- **AlphaVantage**: Limited to free-tier constraints
- **Attribution**: Properly credit all data sources

## Folder Structure

```
enhancements/
├── __init__.py
├── data_providers/
│   ├── __init__.py
│   ├── yahoo_provider.py
│   └── alpha_provider.py
├── data_access/
│   ├── __init__.py
│   └── cache.py
├── technical_indicators/
│   ├── __init__.py
│   └── wilder_atr.py
├── patterns/
│   ├── __init__.py
│   └── confidence.py
├── options/
│   ├── __init__.py
│   └── greeks.py
└── docs/
    ├── data_providers.md
    ├── caching.md
    └── patterns.md
```

## Quick Start

1. Create your feature branch
2. Add code only in `enhancements/`
3. Write tests with >80% coverage
4. Run formatters and linters
5. Update documentation
6. Submit PR

## Example: Adding a New Data Provider

```python
# enhancements/data_providers/new_provider.py
from enhancements.data_access.cache import CacheManager

class NewDataProvider:
    def __init__(self):
        self.cache = CacheManager()
    
    def get_data(self, symbol: str):
        # Check cache first
        cached_data = self.cache.get(f"new_provider_{symbol}")
        if cached_data:
            return cached_data
        
        # Fetch from API
        data = self._fetch_from_api(symbol)
        
        # Store in cache
        self.cache.set(f"new_provider_{symbol}", data)
        return data
```

## Enforcement

These rules are enforced through:
1. Code reviews
2. CI/CD pipeline checks
3. Pre-commit hooks
4. Automated testing

## Questions?

For clarification on any rule, please consult the team lead or create an issue for discussion. 