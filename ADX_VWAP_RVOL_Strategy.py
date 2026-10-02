import numpy as np
import pandas as pd
from datetime import datetime
from freqtrade.strategy import IStrategy, merge_informative_pair
import talib.abstract as ta
from pandas import DataFrame
from typing import Optional, Union
from freqtrade.persistence import Trade


class ADX_VWAP_RVOL_Strategy(IStrategy):
    """
    ADX + VWAP + RVOL(time) Strategy.
    Based on institutional trader indicators: ADX (trend strength), VWAP (avg volume price), RVOL-time (volume vs same time-of-day).
    """
    INTERFACE_VERSION = 3

    # Strategy parameters
    timeframe = '5m'
    can_short = True

    # Minimal ROI
    minimal_roi = {"0": 0.03}

    # Stoploss
    stoploss = -0.05

    # Trailing stop
    trailing_stop = False

    # Process only new candles
    process_only_new_candles = True

    # Parameters
    use_adx = True
    use_vwap = True
    use_rvol = True
    adx_len = 14
    adx_min = 22.0
    rv_look = 20
    rv_min = 1.5
    atr_len = 14
    sl_mult = 1.5
    tp_mult = 3.0
    allow_short = True

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # ATR
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=self.atr_len)

        # ADX / DMI
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=self.adx_len)
        dataframe['di_plus'] = ta.PLUS_DI(dataframe, timeperiod=self.adx_len)
        dataframe['di_minus'] = ta.MINUS_DI(dataframe, timeperiod=self.adx_len)

        # VWAP (session anchored - daily reset)
        dataframe['hlc3'] = (dataframe['high'] + dataframe['low'] + dataframe['close']) / 3
        dataframe['vol_hlc3'] = dataframe['volume'] * dataframe['hlc3']

        # Daily cumsum for VWAP
        dataframe['date'] = dataframe.index.date
        dataframe['cum_vol'] = dataframe.groupby('date')['volume'].cumsum()
        dataframe['cum_vol_hlc3'] = dataframe.groupby('date')['vol_hlc3'].cumsum()
        dataframe['vwap'] = dataframe['cum_vol_hlc3'] / dataframe['cum_vol']

        # RVOL by time-of-day (intraday only)
        dataframe['time_key'] = dataframe.index.hour * 60 + dataframe.index.minute
        # Shift volume by 1 to avoid lookahead, then rolling mean per time bucket
        dataframe['prev_vol'] = dataframe.groupby('time_key')['volume'].shift(1)
        dataframe['avg_vol'] = dataframe.groupby('time_key')['prev_vol'].transform(
            lambda x: x.rolling(window=self.rv_look, min_periods=5).mean()
        )
        dataframe['rvol'] = np.where(
            (dataframe['avg_vol'].notna()) & (dataframe['avg_vol'] > 0),
            dataframe['volume'] / dataframe['avg_vol'],
            np.nan
        )

        # ADX rising
        dataframe['adx_rising'] = dataframe['adx'] > dataframe['adx'].shift(1)

        # VWAP crosses
        dataframe['vwap_crossunder'] = (dataframe['close'].shift(1) > dataframe['vwap'].shift(1)) & (dataframe['close'] < dataframe['vwap'])
        dataframe['vwap_crossover'] = (dataframe['close'].shift(1) < dataframe['vwap'].shift(1)) & (dataframe['close'] > dataframe['vwap'])

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Long conditions
        adx_ok_long = (not self.use_adx) | (
            (dataframe['adx'] > self.adx_min) &
            (dataframe['adx_rising']) &
            (dataframe['di_plus'] > dataframe['di_minus'])
        )
        vwap_ok_long = (not self.use_vwap) | (dataframe['close'] > dataframe['vwap'])
        rvol_ok = (not self.use_rvol) | ((dataframe['rvol'].notna()) & (dataframe['rvol'] > self.rv_min))

        long_sig = adx_ok_long & vwap_ok_long & rvol_ok

        dataframe.loc[long_sig, 'enter_long'] = 1

        # Short conditions
        if self.allow_short:
            adx_ok_short = (not self.use_adx) | (
                (dataframe['adx'] > self.adx_min) &
                (dataframe['adx_rising']) &
                (dataframe['di_minus'] > dataframe['di_plus'])
            )
            vwap_ok_short = (not self.use_vwap) | (dataframe['close'] < dataframe['vwap'])
            short_sig = adx_ok_short & vwap_ok_short & rvol_ok
            dataframe.loc[short_sig, 'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Long exits: VWAP crossunder
        if self.use_vwap:
            dataframe.loc[dataframe['vwap_crossunder'], 'exit_long'] = 1

        # Short exits: VWAP crossover
        if self.use_vwap and self.allow_short:
            dataframe.loc[dataframe['vwap_crossover'], 'exit_short'] = 1

        return dataframe

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        # ATR-based stoploss handled via custom_exit
        return self.stoploss

    def custom_exit(self, pair: str, trade: 'Trade', current_time: datetime,
                    current_rate: float, current_profit: float, **kwargs) -> Optional[Union[str, bool]]:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        entry_price = trade.open_rate
        atr_val = last_candle['atr']

        if np.isnan(atr_val):
            return None

        if trade.is_short:
            stop_loss = entry_price + (self.sl_mult * atr_val)
            take_profit = entry_price - (self.tp_mult * atr_val)
            if current_rate >= stop_loss:
                return 'SL'
            if current_rate <= take_profit:
                return 'TP'
        else:
            stop_loss = entry_price - (self.sl_mult * atr_val)
            take_profit = entry_price + (self.tp_mult * atr_val)
            if current_rate <= stop_loss:
                return 'SL'
            if current_rate >= take_profit:
                return 'TP'

        return None

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                 side: str, **kwargs) -> float:
        return 1.0