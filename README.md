# Extending a factor model

We propose a method to extend a provided factor risk model and study its performance in portfolio construction problems (i.e., a
downstream task")

## Overview

`extending_factor_model` is a Python package for investigating how to extend a provided risk model. It provides:

1. **Convex Markowitz Portfolio Optimizer** - A flexible implementation of mean-variance portfolio optimization
2. **Backtester** - Tools to evaluate portfolio strategies over historical periods
3. **Interactive Visualization** - A Marimo notebook for exploring and visualizing results

## Installation

```bash
# Clone the repository
git clone https://github.com/alextzik/extending_factor_model.git
cd alpha_mod

# Install the package
pip install -e .

# For visualization capabilities
pip install -e ".[viz]"

# For development
pip install -e ".[dev]"
```

## Quick Start

### Basic Portfolio Optimization

```python
import numpy as np
from alpha_mod import markowitz_problem

# Define expected returns and covariance matrix
expected_returns = np.array([0.10, 0.12, 0.08])
cov_matrix = np.array([
    [0.04, 0.01, 0.02],
    [0.01, 0.09, 0.015],
    [0.02, 0.015, 0.05]
])

# Solve the Markowitz optimization problem
result = markowitz_problem(
    expected_returns=expected_returns,
    covariance_matrix=cov_matrix,
    risk_aversion=2.0,
    long_only=True
)

print(f"Optimal weights: {result['weights']}")
print(f"Expected return: {result['expected_return']:.4f}")
print(f"Portfolio volatility: {result['volatility']:.4f}")
```

### Running a Backtest

```python
import pandas as pd
import numpy as np
from alpha_mod import run_backtest

# Create sample returns data
dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
returns = pd.DataFrame(
    np.random.randn(len(dates), 3) * 0.01,
    index=dates,
    columns=['Asset1', 'Asset2', 'Asset3']
)

# Run backtest
result = run_backtest(
    returns=returns,
    rebalance_frequency=20,
    lookback_window=252,
    risk_aversion=2.0,
    transaction_cost=0.001,
    initial_capital=1000000.0,
    long_only=True
)

print(f"Total Return: {result['total_return']:.2%}")
print(f"Sharpe Ratio: {result['sharpe_ratio']:.3f}")
print(f"Max Drawdown: {result['max_drawdown']:.2%}")
```

### Alpha Modification

Modify alpha signals based on past performance:

```python
def momentum_modifier(expected_returns, past_performance):
    """Boost alphas based on recent momentum"""
    recent_returns = past_performance['returns'].iloc[-20:]
    momentum = recent_returns.mean().values
    return expected_returns + 0.5 * momentum

result = run_backtest(
    returns=returns,
    alpha_modifier=momentum_modifier,
    # ... other parameters
)
```

## Interactive Visualization

Launch the Marimo notebook for interactive analysis:

```bash
marimo edit notebooks/portfolio_analysis.py
```

### Force Marimo to open in Google Chrome (macOS)

macOS may open Marimo links in Safari by default. A helper script is included to prefer Chrome if installed:

```bash
chmod +x scripts/marimo_chrome.sh
./scripts/marimo_chrome.sh edit notebooks/market_backtest.py
```

If Google Chrome isn't found the script falls back to a normal `marimo` invocation. It also uses Poetry automatically when available.

The notebook provides:
- Synthetic data generation
- Single-period optimization with various risk aversion levels
- Full backtesting with visualization
- Strategy comparison
- Examples of alpha modification

## Features

### Markowitz Optimizer

The `markowitz_problem` function supports:
- Customizable risk aversion parameter
- Long-only or long-short portfolios
- Transaction cost modeling
- Weight constraints (min/max per asset)
- Leverage limits
- Optimization via CVXPY (supports multiple solvers)

### Backtester

The `run_backtest` function provides:
- Periodic rebalancing with configurable frequency
- Rolling window estimation
- Transaction cost tracking
- Comprehensive performance metrics:
  - Total return
  - Sharpe ratio
  - Maximum drawdown
  - Volatility
  - Turnover
- Custom alpha modification hooks

## Project Structure

```
alpha_mod/
├── alpha_mod/           # Main package
│   ├── __init__.py
│   ├── portfolio_optimizer.py  # Markowitz optimization
│   └── backtester.py           # Backtesting engine
├── notebooks/           # Marimo notebooks
│   └── portfolio_analysis.py
├── tests/              # Test suite
│   ├── test_portfolio_optimizer.py
│   └── test_backtester.py
├── pyproject.toml      # Project configuration
└── README.md
```

## Testing

Run the test suite:

```bash
pytest tests/
```

## Requirements

- Python >= 3.8
- numpy >= 1.24.0
- pandas >= 2.0.0
- cvxpy >= 1.4.0
- scipy >= 1.10.0

Optional (for visualization):
- marimo >= 0.1.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.