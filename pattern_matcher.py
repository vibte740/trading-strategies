"""
Pattern Matching Engine for Trading
- Convert recent candles into pattern fingerprint
- Search historical data for similar patterns (k-NN)
- Analyze forward returns at multiple horizons
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


@dataclass
class PatternConfig:
    """Pattern matching configuration"""
    lookback: int = 100           # candles to use for pattern
    n_neighbors: int = 50         # top similar patterns to find
    horizons: tuple = (15, 30, 60)  # forward return horizons (minutes/bars)
    normalize: bool = True        # z-score normalize patterns
    use_volume: bool = True       # include volume in pattern
    distance_metric: str = 'euclidean'  # 'euclidean', 'cosine', 'dtw'


@dataclass
class PatternMatch:
    """Single historical pattern match"""
    start_idx: int
    start_time: pd.Timestamp
    distance: float
    forward_returns: dict  # horizon -> return pct
    pattern_data: np.ndarray


@dataclass
class PatternAnalysis:
    """Aggregated analysis of similar patterns"""
    current_pattern: np.ndarray
    matches: list[PatternMatch]
    n_matches: int
    avg_returns: dict
    win_rates: dict
    return_distributions: dict
    best_case: dict
    worst_case: dict


class PatternFingerprint:
    """Convert price/volume windows into normalized feature vectors"""
    
    @staticmethod
    def create(df: pd.DataFrame, start_idx: int, lookback: int,
               use_volume: bool = True, normalize: bool = True) -> Optional[np.ndarray]:
        """Extract and normalize pattern window ending at start_idx"""
        if start_idx < lookback:
            return None
        
        window = df.iloc[start_idx - lookback:start_idx]
        
        # Price features: log returns
        closes = window['close'].values
        log_returns = np.diff(np.log(closes))
        
        # Normalize returns within window
        if normalize and len(log_returns) > 1:
            log_returns = (log_returns - log_returns.mean()) / (log_returns.std() + 1e-8)
        
        features = [log_returns]
        
        if use_volume:
            vols = window['volume'].values[1:]  # align with returns
            # Volume z-score within window
            if normalize and vols.std() > 0:
                vols = (vols - vols.mean()) / vols.std()
            else:
                vols = vols / (vols.mean() + 1e-8)
            features.append(vols)
        
        # Optional: range/volatility features
        highs = window['high'].values[1:]
        lows = window['low'].values[1:]
        ranges = (highs - lows) / closes[1:]
        if normalize and ranges.std() > 0:
            ranges = (ranges - ranges.mean()) / ranges.std()
        features.append(ranges)
        
        return np.concatenate(features)


class PatternMatcher:
    """K-NN pattern search on historical data"""
    
    def __init__(self, df: pd.DataFrame, config: PatternConfig = None):
        self.df = df.copy()
        self.config = config or PatternConfig()
        self.fingerprints: Optional[np.ndarray] = None
        self.valid_indices: Optional[np.ndarray] = None
        self.nn_model: Optional[NearestNeighbors] = None
        self._build_index()
    
    def _build_index(self):
        """Pre-compute fingerprints for all valid historical windows"""
        n = len(self.df)
        fps = []
        valid_idx = []
        
        for i in range(self.config.lookback + max(self.config.horizons), n):
            fp = PatternFingerprint.create(
                self.df, i, self.config.lookback,
                self.config.use_volume, self.config.normalize
            )
            if fp is not None:
                fps.append(fp)
                valid_idx.append(i)
        
        self.fingerprints = np.array(fps)
        self.valid_indices = np.array(valid_idx)
        
        # Build K-NN index
        self.nn_model = NearestNeighbors(
            n_neighbors=min(self.config.n_neighbors, len(self.fingerprints)),
            metric=self.config.distance_metric,
            algorithm='auto'
        )
        self.nn_model.fit(self.fingerprints)
        
        print(f"Built pattern index: {len(self.fingerprints)} patterns, "
              f"dim={self.fingerprints.shape[1]}")
    
    def find_similar(self, current_window: pd.DataFrame) -> PatternAnalysis:
        """Find similar patterns to current window and analyze forward returns"""
        # Create fingerprint for current pattern (most recent window)
        current_fp = PatternFingerprint.create(
            current_window, len(current_window), self.config.lookback,
            self.config.use_volume, self.config.normalize
        )
        
        if current_fp is None:
            raise ValueError("Insufficient data for current pattern")
        
        # Search
        distances, indices = self.nn_model.kneighbors([current_fp])
        
        matches = []
        for dist, idx in zip(distances[0], indices[0]):
            hist_idx = self.valid_indices[idx]
            match = self._analyze_match(hist_idx, dist)
            if match:
                matches.append(match)
        
        return self._aggregate_analysis(current_fp, matches)
    
    def _analyze_match(self, hist_idx: int, distance: float) -> Optional[PatternMatch]:
        """Compute forward returns for a historical match"""
        forward = {}
        for h in self.config.horizons:
            if hist_idx + h < len(self.df):
                entry_price = self.df.iloc[hist_idx]['close']
                exit_price = self.df.iloc[hist_idx + h]['close']
                ret = (exit_price - entry_price) / entry_price
                forward[f'{h}m'] = ret
            else:
                forward[f'{h}m'] = np.nan
        
        return PatternMatch(
            start_idx=hist_idx,
            start_time=self.df.index[hist_idx],
            distance=distance,
            forward_returns=forward,
            pattern_data=self.fingerprints[self.valid_indices == hist_idx][0] 
                if hist_idx in self.valid_indices else None
        )
    
    def _aggregate_analysis(self, current_fp: np.ndarray, 
                           matches: list[PatternMatch]) -> PatternAnalysis:
        """Aggregate statistics across matches"""
        horizons = self.config.horizons
        avg_returns = {}
        win_rates = {}
        return_dists = {}
        best_case = {}
        worst_case = {}
        
        for h in horizons:
            key = f'{h}m'
            returns = [m.forward_returns[key] for m in matches 
                      if not np.isnan(m.forward_returns[key])]
            
            if returns:
                returns = np.array(returns)
                avg_returns[key] = float(returns.mean())
                win_rates[key] = float((returns > 0).mean())
                return_dists[key] = {
                    'mean': float(returns.mean()),
                    'std': float(returns.std()),
                    'median': float(np.median(returns)),
                    'p25': float(np.percentile(returns, 25)),
                    'p75': float(np.percentile(returns, 75)),
                    'min': float(returns.min()),
                    'max': float(returns.max()),
                }
                best_case[key] = {
                    'return': float(returns.max()),
                    'time': str(matches[np.argmax(returns)].start_time)
                }
                worst_case[key] = {
                    'return': float(returns.min()),
                    'time': str(matches[np.argmin(returns)].start_time)
                }
            else:
                avg_returns[key] = 0
                win_rates[key] = 0
                return_dists[key] = {}
                best_case[key] = {}
                worst_case[key] = {}
        
        return PatternAnalysis(
            current_pattern=current_fp,
            matches=matches,
            n_matches=len(matches),
            avg_returns=avg_returns,
            win_rates=win_rates,
            return_distributions=return_dists,
            best_case=best_case,
            worst_case=worst_case
        )


# --- Convenience function ---
def analyze_current_pattern(df: pd.DataFrame, config: PatternConfig = None) -> PatternAnalysis:
    """One-liner: analyze current pattern against all history"""
    matcher = PatternMatcher(df, config)
    current_window = df.iloc[-config.lookback - max(config.horizons):] if config else df.iloc[-160:]
    return matcher.find_similar(current_window)


# --- Example usage ---
if __name__ == '__main__':
    # Generate sample BTC-like data
    dates = pd.date_range('2023-01-01', periods=50000, freq='1min')
    np.random.seed(42)
    
    # Random walk with regime changes
    returns = np.random.randn(50000) * 0.0005
    # Add some trends
    for i in range(0, 50000, 5000):
        regime = np.random.choice([-1, 1]) * 0.0003
        returns[i:i+5000] += regime
    
    close = 30000 * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(np.random.randn(50000)) * 0.001)
    low = close * (1 - np.abs(np.random.randn(50000)) * 0.001)
    volume = np.random.lognormal(10, 0.5, 50000)
    
    df = pd.DataFrame({
        'open': np.roll(close, 1),
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)
    df.iloc[0, df.columns.get_loc('open')] = close[0]
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)
    
    # Run pattern analysis
    config = PatternConfig(lookback=100, n_neighbors=50, horizons=(15, 30, 60))
    analysis = analyze_current_pattern(df, config)
    
    print(f"\n{'='*60}")
    print(f"PATTERN ANALYSIS: {analysis.n_matches} similar patterns found")
    print(f"{'='*60}")
    
    for h in config.horizons:
        key = f'{h}m'
        print(f"\n--- {h}-minute forward ---")
        print(f"  Avg Return: {analysis.avg_returns[key]:.4%}")
        print(f"  Win Rate:   {analysis.win_rates[key]:.1%}")
        if key in analysis.return_distributions:
            d = analysis.return_distributions[key]
            print(f"  Median: {d['median']:.4%}, Std: {d['std']:.4%}")
            print(f"  Range: [{d['min']:.4%}, {d['max']:.4%}]")
            print(f"  Best: {analysis.best_case[key].get('return', 0):.4%} at {analysis.best_case[key].get('time', 'N/A')}")
            print(f"  Worst: {analysis.worst_case[key].get('return', 0):.4%} at {analysis.worst_case[key].get('time', 'N/A')}")
    
    print(f"\n--- Top 5 Closest Matches ---")
    for i, m in enumerate(analysis.matches[:5]):
        print(f"  {i+1}. {m.start_time} | dist={m.distance:.4f} | "
              f"15m={m.forward_returns.get('15m', 0):.3%} "
              f"30m={m.forward_returns.get('30m', 0):.3%} "
              f"60m={m.forward_returns.get('60m', 0):.3%}")