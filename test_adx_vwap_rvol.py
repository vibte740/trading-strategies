import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generate sample OHLCV data for testing - create multiple days of data
np.random.seed(42)
n = 5000  # More bars for multiple days
dates = pd.date_range(start='2024-01-01', periods=n, freq='5min')

# Generate trending price data
trend = np.linspace(0, 0.02, n)
noise = np.random.normal(0, 0.0015, n)
prices = 100 * np.exp(np.cumsum(trend + noise))

df = pd.DataFrame(index=dates)
df['open'] = prices * (1 + np.random.normal(0, 0.0005, n))
df['high'] = np.maximum(df['open'], prices * (1 + np.abs(np.random.normal(0, 0.003, n))))
df['low'] = np.minimum(df['open'], prices * (1 - np.abs(np.random.normal(0, 0.003, n))))
df['close'] = prices * (1 + np.random.normal(0, 0.0005, n))
df['volume'] = np.random.lognormal(10, 0.5, n).astype(int)

# Add intraday volume pattern (higher at open/close)
df['hour'] = df.index.hour
df['minute'] = df.index.minute
df.loc[df['hour'].isin([9, 10, 15, 16]), 'volume'] = df.loc[df['hour'].isin([9, 10, 15, 16]), 'volume'] * 3

# Test the strategy indicators
from ADX_VWAP_RVOL_Strategy import ADX_VWAP_RVOL_Strategy

config = {
    'timeframe': '5m',
    'minimal_roi': {"0": 0.03},
    'stoploss': -0.05,
    'trailing_stop': False,
    'process_only_new_candles': True,
    'use_adx': True,
    'use_vwap': True,
    'use_rvol': True,
    'adx_len': 14,
    'adx_min': 15.0,
    'rv_look': 20,
    'rv_min': 1.5,
    'atr_len': 14,
    'sl_mult': 1.5,
    'tp_mult': 3.0,
    'allow_short': True,
}

strat = ADX_VWAP_RVOL_Strategy(config)
result = strat.populate_indicators(df.copy(), {})

print("Days in data:", (result.index.max() - result.index.min()).days + 1)
print("ADX range:", result['adx'].min(), "to", result['adx'].max())
print("RVOL range:", result['rvol'].min(), "to", result['rvol'].max())
print("DI+ > DI-:", (result['di_plus'] > result['di_minus']).sum(), "/", len(result))
print("Close > VWAP:", (result['close'] > result['vwap']).sum(), "/", len(result))
print("RVOL > 1.5:", (result['rvol'] > 1.5).sum(), "/", len(result))

print("\nLast 10 rows of key indicators:")
print(result[['close', 'adx', 'di_plus', 'di_minus', 'vwap', 'rvol', 'adx_rising', 'vwap_crossunder', 'vwap_crossover']].tail(10))

# Test entry signals
result = strat.populate_entry_trend(result, {})
print("\nLong signals:", result['enter_long'].sum())
print("Short signals:", result['enter_short'].sum())

# Test exit signals
result = strat.populate_exit_trend(result, {})
print("Long exits:", result['exit_long'].sum())
print("Short exits:", result['exit_short'].sum())

print("\nTest passed!")