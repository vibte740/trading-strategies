# Trading Strategies Repository

Collection of quantitative trading strategies and pattern matching tools for backtesting and live trading.

## Strategies

### 1. AMD Pattern + POC Volume Strategy (`amd_poc_strategy.py`)
Based on Shadow Intel Trades methodology (TikTok @shadowinteltrades).

**Logic:**
- **Accumulation**: Detect tight consolidation ranges via rolling window
- **Volume Profile**: Calculate POC (Point of Control) - highest volume price in range
- **Manipulation**: Wait for sweep of range extreme (wick rejection)
- **Distribution**: Aggressive expansion opposite direction (>0.7x range size)
- **Entry**: Pullback to POC, don't chase; stop under sweep, target = range projection

**Key Classes:**
- `AMDConfig` - Tunable parameters
- `VolumeProfile` - Fixed-range volume profile + POC detection
- `AMDStrategy` - Full pipeline: consolidation → sweep → distribution → POC pullback
- `backtest_signals()` - Vectorized backtest with equity curve

### 2. Pattern Matcher Engine (`pattern_matcher.py`)
K-NN pattern matching for probabilistic forward return analysis.

**Logic:**
- Convert recent candles into normalized fingerprint (log-returns + volume + range)
- Search historical data for k-nearest neighbors
- Analyze forward returns at multiple horizons (15m, 30m, 60m)
- Aggregate statistics: avg returns, win rates, distributions, best/worst case

**Key Classes:**
- `PatternConfig` - lookback, n_neighbors, horizons, distance_metric
- `PatternFingerprint` - Feature extraction (z-scored)
- `PatternMatcher` - K-NN index building + search
- `PatternAnalysis` - Aggregated forward return statistics

## Installation

```bash
pip install pandas numpy scikit-learn
# Optional for DTW distance:
pip install tslearn
```

## Usage

### AMD Strategy
```python
from amd_poc_strategy import AMDStrategy, AMDConfig, backtest_signals
import pandas as pd

df = pd.read_csv('your_data.csv', index_col='timestamp', parse_dates=True)
# Columns: open, high, low, close, volume

config = AMDConfig(
    lookback_bars=120,
    min_consolidation_bars=20,
    range_threshold_pct=0.02,
    sweep_wick_ratio=0.25,
    pullback_tolerance_atr=0.8
)
strat = AMDStrategy(config)
signals = strat.generate_signals(df)

results = backtest_signals(df, signals, risk_per_trade=0.02)
print(f"Trades: {results['trades']}, WR: {results['winrate']:.1%}")
```

### Pattern Matcher
```python
from pattern_matcher import PatternMatcher, PatternConfig, analyze_current_pattern
import pandas as pd

df = pd.read_csv('btc_1m.csv', index_col='timestamp', parse_dates=True)

config = PatternConfig(
    lookback=100,
    n_neighbors=50,
    horizons=(15, 30, 60),
    use_volume=True,
    distance_metric='euclidean'
)

analysis = analyze_current_pattern(df, config)

for h in config.horizons:
    key = f'{h}m'
    print(f"{h}m: avg={analysis.avg_returns[key]:.3%}, "
          f"wr={analysis.win_rates[key]:.1%}, "
          f"median={analysis.return_distributions[key]['median']:.3%}")
```

### Combined Usage
```python
# 1. Run AMD detector for signal
signals = amd_strat.generate_signals(df)

# 2. When signal triggers, get probabilistic forecast
if signals:
    recent = df.iloc[-100:]  # last 100 candles
    analysis = analyze_current_pattern(recent, pattern_config)
    # Use analysis.avg_returns, win_rates for position sizing
```

## File Structure
```
trading-strategies/
├── amd_poc_strategy.py     # AMD Pattern + POC Volume Strategy
├── pattern_matcher.py      # K-NN Pattern Matching Engine
├── README.md               # This file
└── requirements.txt        # Python dependencies
```

## Notion Documentation
- AMD Strategy: `3e7239d8-18ee-8133-a2c8-d7343cd84511`
- Pattern Matcher: `3e7239d8-18ee-819a-b039-eb66bbfe5f95`

## License
MIT