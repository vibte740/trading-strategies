"""
AMD POC (Point of Control) Strategy
Consolidation -> Sweep -> Pullback to POC -> Entry
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, List
import warnings
warnings.filterwarnings('ignore')


@dataclass
class AMDConfig:
    lookback_bars: int = 120
    min_consolidation_bars: int = 20
    range_threshold_pct: float = 0.02      # 2% max range
    sweep_wick_ratio: float = 0.25         # wick > 25% of candle body
    pullback_tolerance_atr: float = 0.8    # entry tolerance around POC
    atr_period: int = 14
    min_volume_ratio: float = 1.2          # volume confirmation


@dataclass
class Signal:
    timestamp: pd.Timestamp
    symbol: str
    side: str          # 'long' or 'short'
    entry: float
    stop: float
    target: float
    confidence: float
    metadata: dict


class AMDStrategy:
    def __init__(self, config: AMDConfig):
        self.config = config

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    def _find_poc(self, df: pd.DataFrame, start: int, end: int) -> float:
        """Point of Control = price level with highest volume in consolidation range"""
        consolidation = df.iloc[start:end]
        if len(consolidation) == 0:
            return consolidation['close'].mean()
        # Volume-weighted price (approximation using volume profile)
        prices = np.linspace(consolidation['low'].min(), consolidation['high'].max(), 50)
        vol_at_price = np.zeros_like(prices)
        for i, row in consolidation.iterrows():
            idx = np.argmin(np.abs(prices - (row['high'] + row['low']) / 2))
            vol_at_price[idx] += row['volume']
        poc_idx = np.argmax(vol_at_price)
        return prices[poc_idx]

    def _is_consolidation(self, df: pd.DataFrame, end_idx: int) -> Optional[tuple]:
        """Check if recent bars form a tight consolidation"""
        config = self.config
        start_idx = end_idx - config.lookback_bars
        if start_idx < config.min_consolidation_bars:
            return None

        # Consolidation range: min_consolidation_bars BEFORE end_idx (not including end_idx)
        cons_start = end_idx - config.min_consolidation_bars
        cons_end = end_idx  # exclusive
        recent = df.iloc[cons_start:cons_end]
        rng = (recent['high'].max() - recent['low'].min()) / recent['close'].mean()
        if rng > config.range_threshold_pct:
            return None

        poc = self._find_poc(df, cons_start, cons_end)
        return (start_idx, end_idx, poc)

    def _detect_sweep(self, df: pd.DataFrame, idx: int, poc: float, cons_low: float, cons_high: float, side: str) -> bool:
        """Detect sweep candle wicking beyond consolidation then reversing"""
        config = self.config
        if idx < 1:
            return False
        candle = df.iloc[idx]
        body = abs(candle['close'] - candle['open'])
        if body == 0:
            return False

        upper_wick = candle['high'] - max(candle['close'], candle['open'])
        lower_wick = min(candle['close'], candle['open']) - candle['low']

        if side == 'long':
            # Sweep low: wick below consolidation low by threshold %, close back above POC
            sweep_threshold = (cons_high - cons_low) * config.sweep_wick_ratio
            swept = candle['low'] < (cons_low - sweep_threshold)
            recovered = candle['close'] > poc
            return swept and recovered and (lower_wick / body > config.sweep_wick_ratio)
        else:
            # Sweep high: wick above consolidation high by threshold %, close back below POC
            sweep_threshold = (cons_high - cons_low) * config.sweep_wick_ratio
            swept = candle['high'] > (cons_high + sweep_threshold)
            recovered = candle['close'] < poc
            return swept and recovered and (upper_wick / body > config.sweep_wick_ratio)

    def generate_signals(self, df: pd.DataFrame, symbol: str = "AMD") -> List[Signal]:
        config = self.config
        df = df.copy()
        df['atr'] = self._calculate_atr(df, config.atr_period)
        df['vol_ma'] = df['volume'].rolling(20).mean()

        signals = []
        in_position = False
        position_side = None

        for i in range(config.lookback_bars, len(df) - 1):
            # Exit logic
            if in_position:
                candle = df.iloc[i]
                if position_side == 'long':
                    if candle['low'] <= signals[-1].stop:
                        signals[-1].metadata['exit'] = 'stop'
                        signals[-1].metadata['exit_price'] = signals[-1].stop
                        signals[-1].metadata['exit_time'] = df.index[i]
                        in_position = False
                    elif candle['high'] >= signals[-1].target:
                        signals[-1].metadata['exit'] = 'target'
                        signals[-1].metadata['exit_price'] = signals[-1].target
                        signals[-1].metadata['exit_time'] = df.index[i]
                        in_position = False
                else:
                    if candle['high'] >= signals[-1].stop:
                        signals[-1].metadata['exit'] = 'stop'
                        signals[-1].metadata['exit_price'] = signals[-1].stop
                        signals[-1].metadata['exit_time'] = df.index[i]
                        in_position = False
                    elif candle['low'] <= signals[-1].target:
                        signals[-1].metadata['exit'] = 'target'
                        signals[-1].metadata['exit_price'] = signals[-1].target
                        signals[-1].metadata['exit_time'] = df.index[i]
                        in_position = False
                continue

            # Entry logic - check for consolidation + sweep + pullback to POC
            cons = self._is_consolidation(df, i)
            if not cons:
                continue
            start_idx, end_idx, poc = cons
            
            # Get consolidation range for sweep detection
            cons_start = end_idx - config.min_consolidation_bars
            cons_end = end_idx
            cons_data = df.iloc[cons_start:cons_end]
            cons_low = cons_data['low'].min()
            cons_high = cons_data['high'].max()

            # Check next candle for sweep
            sweep_idx = i
            if sweep_idx >= len(df):
                continue

            # Long setup: sweep low then pullback to POC
            if self._detect_sweep(df, sweep_idx, poc, cons_low, cons_high, 'long'):
                # Wait for pullback to POC tolerance
                atr = df.iloc[sweep_idx]['atr']
                if atr == 0 or np.isnan(atr):
                    continue
                entry_zone_high = poc + config.pullback_tolerance_atr * atr
                entry_zone_low = poc - config.pullback_tolerance_atr * atr

                # Check subsequent bars for entry
                for j in range(sweep_idx + 1, min(sweep_idx + 10, len(df))):
                    c = df.iloc[j]
                    if entry_zone_low <= c['low'] <= entry_zone_high and c['volume'] > c['vol_ma'] * config.min_volume_ratio:
                        entry = max(c['open'], entry_zone_low)
                        stop = poc - 1.5 * atr
                        target = entry + 2 * (entry - stop)  # 2:1 R:R
                        sig = Signal(
                            timestamp=df.index[j],
                            symbol=symbol,
                            side='long',
                            entry=entry,
                            stop=stop,
                            target=target,
                            confidence=0.75,
                            metadata={'poc': poc, 'consolidation_bars': config.min_consolidation_bars, 'sweep_idx': sweep_idx}
                        )
                        signals.append(sig)
                        in_position = True
                        position_side = 'long'
                        break

            # Short setup: sweep high then pullback to POC
            elif self._detect_sweep(df, sweep_idx, poc, cons_low, cons_high, 'short'):
                atr = df.iloc[sweep_idx]['atr']
                if atr == 0 or np.isnan(atr):
                    continue
                entry_zone_high = poc + config.pullback_tolerance_atr * atr
                entry_zone_low = poc - config.pullback_tolerance_atr * atr

                for j in range(sweep_idx + 1, min(sweep_idx + 10, len(df))):
                    c = df.iloc[j]
                    if entry_zone_low <= c['high'] <= entry_zone_high and c['volume'] > c['vol_ma'] * config.min_volume_ratio:
                        entry = min(c['open'], entry_zone_high)
                        stop = poc + 1.5 * atr
                        target = entry - 2 * (stop - entry)
                        sig = Signal(
                            timestamp=df.index[j],
                            symbol=symbol,
                            side='short',
                            entry=entry,
                            stop=stop,
                            target=target,
                            confidence=0.75,
                            metadata={'poc': poc, 'consolidation_bars': config.min_consolidation_bars, 'sweep_idx': sweep_idx}
                        )
                        signals.append(sig)
                        in_position = True
                        position_side = 'short'
                        break

        return signals


def backtest_signals(df: pd.DataFrame, signals: List[Signal], risk_per_trade: float = 0.02,
                     initial_capital: float = 100000) -> dict:
    """Simple backtest engine"""
    if not signals:
        return {'trades': 0, 'wins': 0, 'losses': 0, 'winrate': 0, 'expectancy': 0,
                'total_return': 0, 'max_drawdown': 0, 'sharpe': 0, 'equity_curve': []}

    capital = initial_capital
    equity = [capital]
    trades = []
    wins = 0
    losses = 0

    for sig in signals:
        if 'exit' not in sig.metadata:
            continue  # open trade, skip

        entry = sig.entry
        exit_price = sig.metadata.get('exit_price', entry)
        side = sig.side

        if side == 'long':
            pnl_pct = (exit_price - entry) / entry
        else:
            pnl_pct = (entry - exit_price) / entry

        pnl = capital * risk_per_trade * pnl_pct / 0.02  # normalize to risk
        capital += pnl
        equity.append(capital)

        if pnl > 0:
            wins += 1
        else:
            losses += 1

        trades.append({
            'entry_time': sig.timestamp,
            'exit_time': sig.metadata.get('exit_time'),
            'side': side,
            'entry': entry,
            'exit': exit_price,
            'pnl_pct': pnl_pct,
            'pnl': pnl,
            'exit_reason': sig.metadata.get('exit'),
            'confidence': sig.confidence
        })

    total_trades = wins + losses
    winrate = wins / total_trades if total_trades > 0 else 0

    returns = pd.Series(equity).pct_change().dropna()
    expectancy = returns.mean() if len(returns) > 0 else 0
    sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
    max_dd = (pd.Series(equity).cummax() - pd.Series(equity)).max() / pd.Series(equity).cummax().max()
    total_return = (capital - initial_capital) / initial_capital

    return {
        'trades': total_trades,
        'wins': wins,
        'losses': losses,
        'winrate': winrate,
        'expectancy': expectancy,
        'total_return': total_return,
        'max_drawdown': max_dd,
        'sharpe': sharpe,
        'equity_curve': equity,
        'trade_log': trades
    }