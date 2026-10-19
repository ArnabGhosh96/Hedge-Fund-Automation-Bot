"""MT5 connection and multi-timeframe market data/indicator collection."""
import logging
from typing import Any

import pandas as pd

import config

log = logging.getLogger(__name__)
try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None


def _require_mt5() -> Any:
    if mt5 is None:
        raise RuntimeError("MetaTrader5 package is unavailable; install it on the MT5 Windows host.")
    return mt5


def initialize_mt5() -> bool:
    api = _require_mt5()
    initialized = api.initialize(path=config.MT5_PATH) if config.MT5_PATH else api.initialize()
    if not initialized:
        raise RuntimeError(f"MT5 initialization failed: {api.last_error()}")
    if config.MT5_LOGIN is not None and not api.login(
        config.MT5_LOGIN, password=config.MT5_PASSWORD, server=config.MT5_SERVER or None
    ):
        error = api.last_error()
        api.shutdown()
        raise RuntimeError(f"MT5 login failed: {error}")
    account = api.account_info()
    if account is None:
        raise RuntimeError(f"MT5 account_info unavailable: {api.last_error()}")
    log.info("[SYSTEM] MT5 connected server=%s balance=%.2f equity=%.2f", account.server, account.balance, account.equity)
    return True


def calculate_indicators(df: pd.DataFrame) -> dict[str, Any]:
    if df is None or df.empty:
        raise ValueError("Cannot calculate indicators from empty rates")
    frame = df.copy()
    required = {"open", "high", "low", "close", "tick_volume"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing OHLCV columns: {sorted(missing)}")
    close = pd.to_numeric(frame["close"], errors="coerce")
    high, low = frame["high"].astype(float), frame["low"].astype(float)
    previous_close = close.shift(1)
    true_range = pd.concat([(high - low), (high - previous_close).abs(), (low - previous_close).abs()], axis=1).max(axis=1)
    frame["ema"] = close.ewm(span=config.EMA_PERIOD, adjust=False, min_periods=config.EMA_PERIOD).mean()
    delta = close.diff()
    gains = delta.clip(lower=0).ewm(alpha=1 / config.RSI_PERIOD, adjust=False, min_periods=config.RSI_PERIOD).mean()
    losses = (-delta.clip(upper=0)).ewm(alpha=1 / config.RSI_PERIOD, adjust=False, min_periods=config.RSI_PERIOD).mean()
    rs = gains / losses.replace(0, float("nan"))
    frame["rsi"] = (100 - 100 / (1 + rs)).where(losses.ne(0), 100.0)
    frame["atr"] = true_range.ewm(alpha=1 / config.ATR_PERIOD, adjust=False, min_periods=config.ATR_PERIOD).mean()
    frame["relative_volume"] = frame["tick_volume"] / frame["tick_volume"].rolling(config.VOL_MA_PERIOD).mean().replace(0, float("nan"))
    latest = frame.iloc[-1]
    trend = "SIDEWAYS"
    if len(frame) >= 10:
        first, last = float(frame["close"].iloc[-10]), float(frame["close"].iloc[-1])
        trend = "UP" if last > first else "DOWN" if last < first else "SIDEWAYS"
    def number(value: Any) -> float | None:
        return float(value) if pd.notna(value) else None
    recent = frame.tail(10)
    return {
        "close": number(latest["close"]), "ema": number(latest["ema"]), "rsi": number(latest["rsi"]),
        "atr": number(latest["atr"]), "relative_volume": number(latest["relative_volume"]),
        "tick_volume": int(latest["tick_volume"]), "trend": trend,
        "structure": {"high_10": float(recent["high"].max()), "low_10": float(recent["low"].min()),
                      "change_10": float(recent["close"].iloc[-1] - recent["close"].iloc[0])},
    }


def _rates(symbol: str, timeframe: int) -> pd.DataFrame:
    api = _require_mt5()
    rates = api.copy_rates_from_pos(symbol, timeframe, 0, 250)
    if rates is None or len(rates) < 20:
        raise RuntimeError(f"Insufficient rates for {symbol}: {api.last_error()}")
    frame = pd.DataFrame(rates)
    if "time" in frame:
        frame["time"] = pd.to_datetime(frame["time"], unit="s", utc=True)
    return frame


def fetch_multi_timeframe_data(symbol: str) -> dict[str, Any]:
    api = _require_mt5()
    if not api.symbol_select(symbol, True):
        raise RuntimeError(f"Could not select {symbol}: {api.last_error()}")
    tick, account = api.symbol_info_tick(symbol), api.account_info()
    if tick is None or account is None:
        raise RuntimeError(f"Missing tick/account data for {symbol}: {api.last_error()}")
    h1, daily = calculate_indicators(_rates(symbol, config.TIMEFRAMES["H1"])), calculate_indicators(_rates(symbol, config.TIMEFRAMES["D1"]))
    return {"symbol": symbol, "bid": float(tick.bid), "ask": float(tick.ask), "spread": float(tick.ask - tick.bid),
            "equity": float(account.equity), "balance": float(account.balance), "h1_data": h1, "daily_data": daily}


def fetch_correlated_asset_prices(current_symbol: str, all_symbols: list[str]) -> dict[str, dict[str, float]]:
    api = _require_mt5()
    prices: dict[str, dict[str, float]] = {}
    for symbol in all_symbols:
        if symbol == current_symbol:
            continue
        try:
            if not api.symbol_select(symbol, True):
                continue
            tick = api.symbol_info_tick(symbol)
            if tick:
                prices[symbol] = {"bid": float(tick.bid), "ask": float(tick.ask)}
        except Exception as exc:
            log.warning("[SYSTEM] Correlation tick unavailable for %s: %s", symbol, exc)
    return prices
