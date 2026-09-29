"""
Python port of the Pine v6 strategy:
"Crypto Scalper - Multi-Filter EMA + VWAP + ATR"

Input: OHLCV DataFrame indexed by UTC datetime with columns open, high, low, close, volume.
Usage:
    python crypto_scalper.py data.csv          # CSV with timestamp,open,high,low,close,volume
    python crypto_scalper.py --ccxt BTC/USDT 5m 5000   # needs `pip install ccxt`
"""
import sys
import numpy as np
import pandas as pd

# ---------------- Parameters (match Pine inputs) ----------------
FAST_EMA = 9
SLOW_EMA = 21
HTF_EMA = 200
ADX_THRESH = 20
ATR_MULT_TP = 2.0
ATR_MULT_SL = 1.2
INITIAL_CAPITAL = 100.0
COMMISSION_PCT = 0.05  # per side, percent


# ---------------- TradingView-compatible indicators ----------------
def rma(s: pd.Series, length: int) -> pd.Series:
    """Wilder's RMA, seeded with SMA like Pine's ta.rma."""
    v = s.to_numpy(dtype=float)
    out = np.full(len(v), np.nan)
    if len(v) < length:
        return pd.Series(out, index=s.index)
    out[length - 1] = np.nanmean(v[:length])
    a = 1.0 / length
    for i in range(length, len(v)):
        out[i] = a * v[i] + (1 - a) * out[i - 1]
    return pd.Series(out, index=s.index)


def ema(s: pd.Series, length: int) -> pd.Series:
    """EMA seeded with SMA like Pine's ta.ema."""
    v = s.to_numpy(dtype=float)
    out = np.full(len(v), np.nan)
    if len(v) < length:
        return pd.Series(out, index=s.index)
    out[length - 1] = v[:length].mean()
    a = 2.0 / (length + 1)
    for i in range(length, len(v)):
        out[i] = a * v[i] + (1 - a) * out[i - 1]
    return pd.Series(out, index=s.index)


def true_range(df: pd.DataFrame) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - pc).abs(),
                    (df["low"] - pc).abs()], axis=1).max(axis=1)
    tr.iloc[0] = df["high"].iloc[0] - df["low"].iloc[0]  # Pine: first bar = high-low
    return tr


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    return rma(true_range(df), length)


def adx(df: pd.DataFrame, di_len: int = 14, adx_len: int = 14) -> pd.Series:
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)
    tr_rma = rma(true_range(df), di_len)
    plus = 100 * rma(plus_dm, di_len) / tr_rma
    minus = 100 * rma(minus_dm, di_len) / tr_rma
    total = plus + minus
    total = total.where(total != 0, 1.0)
    return 100 * rma((plus - minus).abs() / total, adx_len)


def session_vwap(df: pd.DataFrame, src: pd.Series) -> pd.Series:
    """ta.vwap(close): anchor resets each UTC day, weighted by volume."""
    day = df.index.normalize()
    pv = (src * df["volume"]).groupby(day).cumsum()
    vol = df["volume"].groupby(day).cumsum()
    return pv / vol


# ---------------- Signals ----------------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["fast"] = ema(df["close"], FAST_EMA)
    df["slow"] = ema(df["close"], SLOW_EMA)
    df["htf"] = ema(df["close"], HTF_EMA)
    df["vwap"] = session_vwap(df, df["close"])
    df["atr"] = atr(df, 14)
    df["adx"] = adx(df, 14, 14)

    cross_up = (df["fast"] > df["slow"]) & (df["fast"].shift(1) <= df["slow"].shift(1))
    cross_dn = (df["fast"] < df["slow"]) & (df["fast"].shift(1) >= df["slow"].shift(1))
    trend_up = (df["close"] > df["htf"]) & (df["close"] > df["vwap"])
    trend_dn = (df["close"] < df["htf"]) & (df["close"] < df["vwap"])
    has_vol = df["adx"] > ADX_THRESH

    df["long_sig"] = cross_up & trend_up & has_vol
    df["short_sig"] = cross_dn & trend_dn & has_vol
    return df


# ---------------- Backtester (mimics TradingView fills) ----------------
def backtest(df: pd.DataFrame):
    """
    - Signal on bar close -> market entry at NEXT bar's open.
    - SL/TP fixed from signal bar's close +/- ATR multiples.
    - Exit orders are live on the entry bar itself.
    - If SL and TP both fall inside one bar, TradingView's heuristic is used:
      if open is closer to high, assume O->H->L->C, else O->L->H->C.
    - Gaps through SL/TP fill at the open.
    - 100% of equity per trade, commission per side.
    """
    o, h, l, c = (df[x].to_numpy() for x in ("open", "high", "low", "close"))
    atr_v = df["atr"].to_numpy()
    long_s = df["long_sig"].to_numpy()
    short_s = df["short_sig"].to_numpy()
    idx = df.index
    fee = COMMISSION_PCT / 100.0

    equity = INITIAL_CAPITAL
    pos = 0            # +1 long, -1 short, 0 flat
    qty = entry = sl = tp = 0.0
    entry_time = None
    pending = None     # (dir, sl, tp)
    trades = []
    equity_curve = np.full(len(df), np.nan)

    for i in range(len(df)):
        # 1) fill pending entry at open
        if pending is not None and pos == 0:
            d, p_sl, p_tp = pending
            entry = o[i]
            qty = equity / entry
            equity -= qty * entry * fee
            pos, sl, tp, entry_time = d, p_sl, p_tp, idx[i]
            pending = None

        # 2) check exits (including on entry bar)
        if pos != 0:
            exit_px = None
            reason = None
            if pos == 1:
                if o[i] <= sl:
                    exit_px, reason = o[i], "SL(gap)"
                elif o[i] >= tp:
                    exit_px, reason = o[i], "TP(gap)"
                else:
                    high_first = abs(h[i] - o[i]) < abs(l[i] - o[i])
                    hit_tp, hit_sl = h[i] >= tp, l[i] <= sl
                    if high_first:
                        if hit_tp: exit_px, reason = tp, "TP"
                        elif hit_sl: exit_px, reason = sl, "SL"
                    else:
                        if hit_sl: exit_px, reason = sl, "SL"
                        elif hit_tp: exit_px, reason = tp, "TP"
            else:
                if o[i] >= sl:
                    exit_px, reason = o[i], "SL(gap)"
                elif o[i] <= tp:
                    exit_px, reason = o[i], "TP(gap)"
                else:
                    low_first = abs(l[i] - o[i]) < abs(h[i] - o[i])
                    hit_tp, hit_sl = l[i] <= tp, h[i] >= sl
                    if low_first:
                        if hit_tp: exit_px, reason = tp, "TP"
                        elif hit_sl: exit_px, reason = sl, "SL"
                    else:
                        if hit_sl: exit_px, reason = sl, "SL"
                        elif hit_tp: exit_px, reason = tp, "TP"

            if exit_px is not None:
                pnl = pos * (exit_px - entry) * qty
                equity += pnl - qty * exit_px * fee
                trades.append(dict(entry_time=entry_time, exit_time=idx[i],
                                   side="Long" if pos == 1 else "Short",
                                   entry=entry, exit=exit_px, reason=reason,
                                   pnl=pnl - qty * (entry + exit_px) * fee))
                pos = 0

        # 3) new signals at bar close (only when flat and nothing pending)
        if pos == 0 and pending is None and not np.isnan(atr_v[i]):
            if long_s[i]:
                pending = (1, c[i] - atr_v[i] * ATR_MULT_SL, c[i] + atr_v[i] * ATR_MULT_TP)
            elif short_s[i]:
                pending = (-1, c[i] + atr_v[i] * ATR_MULT_SL, c[i] - atr_v[i] * ATR_MULT_TP)

        # mark-to-market equity
        equity_curve[i] = equity + (pos * (c[i] - entry) * qty if pos != 0 else 0.0)

    return pd.DataFrame(trades), pd.Series(equity_curve, index=idx)


def report(trades: pd.DataFrame, curve: pd.Series):
    if trades.empty:
        print("No trades.")
        return
    wins = trades[trades["pnl"] > 0]
    losses = trades[trades["pnl"] <= 0]
    gross_win, gross_loss = wins["pnl"].sum(), -losses["pnl"].sum()
    dd = (curve / curve.cummax() - 1).min() * 100
    print(f"Trades:        {len(trades)}  (long {int((trades.side=='Long').sum())}, short {int((trades.side=='Short').sum())})")
    print(f"Win rate:      {len(wins) / len(trades) * 100:.1f}%")
    print(f"Net profit:    {trades['pnl'].sum():.2f}  ({trades['pnl'].sum() / INITIAL_CAPITAL * 100:.1f}%)")
    print(f"Profit factor: {gross_win / gross_loss if gross_loss else float('inf'):.2f}")
    print(f"Max drawdown:  {dd:.2f}%")
    print(f"Final equity:  {curve.iloc[-1]:.2f}")


# ---------------- Data loading ----------------
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    ts = df.columns[0]
    df[ts] = pd.to_datetime(df[ts], utc=True, unit="ms" if np.issubdtype(df[ts].dtype, np.number) else None)
    df = df.set_index(ts).sort_index()
    df.columns = [x.lower() for x in df.columns]
    return df[["open", "high", "low", "close", "volume"]]


def load_ccxt(symbol="BTC/USDT", timeframe="5m", limit=5000, exchange="binance") -> pd.DataFrame:
    import ccxt
    ex = getattr(ccxt, exchange)()
    tf_ms = ex.parse_timeframe(timeframe) * 1000
    since = ex.milliseconds() - limit * tf_ms
    rows = []
    while len(rows) < limit:
        batch = ex.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        if not batch:
            break
        rows += batch
        since = batch[-1][0] + tf_ms
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.drop_duplicates("ts").set_index("ts")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--ccxt":
        sym = sys.argv[2] if len(sys.argv) > 2 else "BTC/USDT"
        tf = sys.argv[3] if len(sys.argv) > 3 else "5m"
        n = int(sys.argv[4]) if len(sys.argv) > 4 else 5000
        data = load_ccxt(sym, tf, n)
    elif len(sys.argv) >= 2:
        data = load_csv(sys.argv[1])
    else:
        sys.exit(__doc__)

    data = add_indicators(data)
    trades, curve = backtest(data)
    report(trades, curve)
    trades.to_csv("scalper_trades.csv", index=False)
