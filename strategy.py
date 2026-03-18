"""
Trading strategy — agent modifies this file.
Exports: strategy(df) -> pd.Series of signals (+1 long, -1 short, 0 flat)
"""

import pandas as pd
import numpy as np


def strategy(df: pd.DataFrame) -> pd.Series:
    """Long-only: SMA50 + multi-entry ADX/DI + BB breakout + vol filter.

    Primary: SMA50 trend + ADX>20 + DI spread>12
    Secondary: ADX>36 + DI>6 (strong trend, relaxed directional)
    BB dip-buy + BB upper breakout for momentum
    Regime: Go flat when realized vol is extreme (> 2x median).
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # Trend filter
    sma50 = close.rolling(51).mean()
    trend_up = close > sma50

    # Volume filter
    vol_median = volume.rolling(58).median()
    high_volume = volume > vol_median

    # Bollinger Bands (20, 2)
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_lower = bb_mid - 2 * bb_std
    bb_upper = bb_mid + 2 * bb_std

    # Volatility regime: 20-day realized vol
    daily_ret = close.pct_change()
    vol20 = daily_ret.rolling(20).std()
    vol_median = vol20.rolling(252).median()  # 1-year median vol
    extreme_vol = vol20 > (vol_median * 2.0)

    # ADX(14) with DI
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)

    atr14 = tr.rolling(14).mean()
    plus_di = 100 * (plus_dm.rolling(14).mean() / atr14)
    minus_di = 100 * (minus_dm.rolling(14).mean() / atr14)
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx = dx.rolling(14).mean()

    di_spread = plus_di - minus_di
    di_strong_bullish = di_spread > 11.502

    # Smoothed ADX
    adx_smooth = adx.ewm(span=3, adjust=False).mean()
    strong_trend = adx_smooth > 20.1

    # Secondary: very strong ADX, relaxed DI
    very_strong_trend = adx_smooth > 40
    di_moderate_bullish = di_spread > 5.99

    # Smoothed DI for BB breakout (EMA for faster response)
    di_spread_smooth = di_spread.ewm(span=3, adjust=False).mean()

    signals = pd.Series(0, index=df.index)

    # Primary: DI spread + uptrend + ADX confirmation + volume
    signals[trend_up & strong_trend & di_strong_bullish & high_volume] = 1

    # Secondary: strong ADX with moderate DI
    signals[trend_up & very_strong_trend & di_moderate_bullish] = 1

    # BB oversold bounce in uptrend
    signals[trend_up & (close < bb_lower)] = 1

    # BB upper breakout (momentum entry with smoothed DI + ADX confirmation)
    signals[trend_up & (close > bb_upper) & (di_spread_smooth > 8.64) & strong_trend] = 1

    # Go flat during extreme volatility
    signals[extreme_vol] = 0

    return signals


# ---------------------------------------------------------------------------
# Runner — imports prepare.py, backtests all assets, prints metrics
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from prepare import load_all_assets, run_backtest, print_metrics

    assets = load_all_assets()
    metrics = run_backtest(strategy, assets)
    print_metrics(metrics)
