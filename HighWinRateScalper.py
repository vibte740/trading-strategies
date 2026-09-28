import numpy as np
import pandas as pd
from freqtrade.strategy import IStrategy, merge_informative_pair
import pandas_ta as ta

class HighWinRateScalper(IStrategy):
    """
    High Win Rate Scalper Strategy.
    - Entry: Price < lower BB and RSI < 30
    - Exit: Price > middle BB
    """
    INTERFACE_VERSION = 3
    
    timeframe = '5m'
    minimal_roi = {"0": 0.005}  # 0.5% profit
    stoploss = -0.02           # 2% stoploss

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe['rsi'] = ta.rsi(dataframe['close'], length=14)
        bb = ta.bbands(dataframe['close'], length=20, std=2.0)
        dataframe['bb_lower'] = bb['BBL_20_2.0']
        dataframe['bb_mid'] = bb['BBM_20_2.0']
        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[
            (dataframe['close'] < dataframe['bb_lower']) & 
            (dataframe['rsi'] < 30),
            'enter_long'
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[
            (dataframe['close'] > dataframe['bb_mid']),
            'exit_long'
        ] = 1
        return dataframe
