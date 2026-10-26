"""MT5 connection and multi-timeframe market data/indicator collection."""
import logging
import math
from datetime import datetime, timezone
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


def calculate_indicators(df: pd.DataFrame, recent_bars: int = 0, recent_time_fmt: str = "%m-%dT%H:%MZ") -> dict[str, Any]:
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

    # Wilder ADX/DMI: separates trending regimes from chop so RSI/structure signals are not over-read.
    def wilder(series: pd.Series) -> pd.Series:
        return series.ewm(alpha=1 / config.ADX_PERIOD, adjust=False, min_periods=config.ADX_PERIOD).mean()
    up_move, down_move = frame["high"].diff(), -frame["low"].diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    smoothed_tr = wilder(true_range).replace(0, float("nan"))
    di_plus = 100 * wilder(plus_dm) / smoothed_tr
    di_minus = 100 * wilder(minus_dm) / smoothed_tr
    di_sum = (di_plus + di_minus).replace(0, float("nan"))
    frame["adx"] = wilder(100 * (di_plus - di_minus).abs() / di_sum)
    frame["di_plus"], frame["di_minus"] = di_plus, di_minus

    latest = frame.iloc[-1]
    trend = "SIDEWAYS"
    if len(frame) >= 10:
        first, last = float(frame["close"].iloc[-10]), float(frame["close"].iloc[-1])
        trend = "UP" if last > first else "DOWN" if last < first else "SIDEWAYS"
    def number(value: Any) -> float | None:
        return float(value) if pd.notna(value) else None
    recent = frame.tail(10)
    bar_time: int | None = None
    if "time" in frame and pd.notna(latest["time"]):
        # Timestamp of the (possibly still forming) latest candle; the scanner uses it
        # to cache decisions per candle so unchanged bars never trigger an AI call.
        bar_time = int(latest["time"].timestamp())

    def compress_recent(tail: pd.DataFrame) -> list[dict[str, Any]]:
        """Compact OHLCV rows (t,o,h,l,c,v; newest last) so the model can read actual price action."""
        rows: list[dict[str, Any]] = []
        for _, row in tail.iterrows():
            open_, high, low, close = (float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"]))
            volume = int(row["tick_volume"])
            if not all(math.isfinite(v) for v in (open_, high, low, close)):  # keep strict JSON NaN-free
                continue
            rows.append({"t": row["time"].strftime(recent_time_fmt) if "time" in frame and pd.notna(row["time"]) else None,
                         "o": round(open_, 5), "h": round(high, 5), "l": round(low, 5), "c": round(close, 5), "v": volume})
        return rows

    atr_value = number(latest["atr"])
    close_value, ema_value = number(latest["close"]), number(latest["ema"])
    high_10, low_10 = float(recent["high"].max()), float(recent["low"].min())
    change_10 = float(recent["close"].iloc[-1] - recent["close"].iloc[0])
    range_10 = high_10 - low_10
    # ATR-normalized metrics keep every symbol comparable regardless of its price scale.
    payload = {
        "close": close_value, "ema": ema_value, "rsi": number(latest["rsi"]),
        "atr": atr_value, "relative_volume": number(latest["relative_volume"]),
        "tick_volume": int(latest["tick_volume"]), "trend": trend, "bar_time": bar_time,
        "adx": number(latest["adx"]), "di_plus": number(latest["di_plus"]), "di_minus": number(latest["di_minus"]),
        "close_vs_ema_atr": round((close_value - ema_value) / atr_value, 2)
                             if (atr_value and ema_value is not None and close_value is not None) else None,
        "structure": {"high_10": high_10, "low_10": low_10,
                      "change_10_atr": round(change_10 / atr_value, 2) if atr_value else None,
                      "range_10_atr": round(range_10 / atr_value, 2) if atr_value else None,
                      "close_position_10": round((close_value - low_10) / range_10, 2)
                                            if (range_10 > 0 and close_value is not None) else None},
    }
    if recent_bars > 0:
        payload["recent"] = compress_recent(frame.tail(int(recent_bars)))
    return payload


def _rates(symbol: str, timeframe: int) -> pd.DataFrame:
    api = _require_mt5()
    rates = api.copy_rates_from_pos(symbol, timeframe, 0, 250)
    if rates is None or len(rates) < 20:
        raise RuntimeError(f"Insufficient rates for {symbol}: {api.last_error()}")
    frame = pd.DataFrame(rates)
    if "time" in frame:
        frame["time"] = pd.to_datetime(frame["time"], unit="s", utc=True)
    return frame


def identify_key_levels(frame: pd.DataFrame, atr: float) -> list[dict[str, Any]]:
    """Deterministic support/resistance from H1 swing pivots for the decision prompt.

    A pivot high/low stands out on KEY_LEVEL_PIVOT_BARS candles per side; pivots within
    0.35 ATR of each other are clustered and their touch count recorded. The strongest few
    clusters nearest the current price are returned (distance_atr = ATRs from live price).
    Never raises: on any data hiccup the scan simply loses the level hints.
    """
    try:
        width = config.KEY_LEVEL_PIVOT_BARS
        if frame is None or len(frame) < 2 * width + 3 or not atr or atr <= 0:
            return []
        high, low = frame["high"].astype(float).tolist(), frame["low"].astype(float).tolist()
        pivots: list[float] = []
        for i in range(width, len(high) - width):
            span = range(i - width, i + width + 1)
            if all(high[i] >= high[j] for j in span):
                pivots.append(high[i])
            if all(low[i] <= low[j] for j in span):
                pivots.append(low[i])
        reference = float(frame["close"].astype(float).iloc[-1])
        clusters: list[list[float]] = []
        for level in sorted(pivots):
            if clusters and abs(level - clusters[-1][-1]) <= 0.35 * atr:
                clusters[-1].append(level)
            else:
                clusters.append([level])
        levels = [{"price": round(sum(cluster) / len(cluster), 6), "touches": len(cluster),
                   "distance_atr": round(abs(sum(cluster) / len(cluster) - reference) / atr, 2)}
                  for cluster in clusters]
        below = sorted((lv for lv in levels if lv["price"] < reference), key=lambda lv: -lv["price"])[:config.KEY_LEVEL_MAX_PER_SIDE]
        above = sorted((lv for lv in levels if lv["price"] >= reference), key=lambda lv: lv["price"])[:config.KEY_LEVEL_MAX_PER_SIDE]
        return sorted(below + above, key=lambda lv: lv["distance_atr"])
    except (ValueError, KeyError, IndexError, TypeError):
        return []


def session_label(moment: datetime) -> str:
    """Weekday + liquidity-window label for a UTC moment (FX-flavoured, still useful for crypto)."""
    hour = moment.hour
    if hour < 7:
        window = "tokyo-asia"
    elif hour < 13:
        window = "london (asian afternoon)"
    elif hour < 16:
        window = "london-new york overlap (peak liquidity)"
    elif hour < 21:
        window = "new york"
    else:
        window = "after-hours (thin liquidity)"
    return f"{moment.strftime('%A')} {hour:02d}:00 UTC — {window}"


def fetch_multi_timeframe_data(symbol: str) -> dict[str, Any]:
    api = _require_mt5()
    if not api.symbol_select(symbol, True):
        raise RuntimeError(f"Could not select {symbol}: {api.last_error()}")
    tick, account = api.symbol_info_tick(symbol), api.account_info()
    if tick is None or account is None:
        raise RuntimeError(f"Missing tick/account data for {symbol}: {api.last_error()}")
    # Candle windows are .env-configurable; H1 = price action, Daily = trend context.
    h1_frame = _rates(symbol, config.TIMEFRAMES["H1"])
    h1 = calculate_indicators(h1_frame, recent_bars=config.RECENT_H1_BARS)
    daily = calculate_indicators(_rates(symbol, config.TIMEFRAMES["D1"]), recent_bars=config.RECENT_DAILY_BARS, recent_time_fmt="%m-%d")
    # Deterministic context for the model: swing support/resistance + liquidity window.
    if config.KEY_LEVELS_IN_PROMPT:
        h1["key_levels"] = identify_key_levels(h1_frame, float(h1.get("atr") or 0))
    return {"symbol": symbol, "bid": float(tick.bid), "ask": float(tick.ask), "spread": float(tick.ask - tick.bid),
            "equity": float(account.equity), "balance": float(account.balance), "h1_data": h1, "daily_data": daily,
            "session_context": session_label(datetime.now(timezone.utc)) if config.SESSION_CONTEXT_IN_PROMPT else None}


def fetch_h1_atr(symbol: str) -> float | None:
    """Latest H1 ATR (same Wilder/EMA formula as calculate_indicators) for exit management.

    Returns None whenever MT5 data is unavailable so the caller just skips that pass
    instead of failing the whole monitor loop.
    """
    try:
        api = _require_mt5()
        if not api.symbol_select(symbol, True):
            return None
        frame = _rates(symbol, config.TIMEFRAMES["H1"])
        high, low = frame["high"].astype(float), frame["low"].astype(float)
        close = pd.to_numeric(frame["close"], errors="coerce")
        previous_close = close.shift(1)
        true_range = pd.concat([(high - low), (high - previous_close).abs(), (low - previous_close).abs()], axis=1).max(axis=1)
        atr = true_range.ewm(alpha=1 / config.ATR_PERIOD, adjust=False, min_periods=config.ATR_PERIOD).mean().iloc[-1]
        return float(atr) if float(atr) > 0 else None
    except (RuntimeError, ValueError, KeyError, IndexError):
        return None


def _daily_moves_from_rates(rates) -> dict[str, float] | None:
    """1-day and 5-day percent changes from D1 closes (newest last) for readable cross-asset context."""
    closes = [float(r["close"]) for r in rates]
    if len(closes) < 6 or closes[-2] <= 0 or closes[-6] <= 0:
        return None
    return {"chg_1d": round(100 * (closes[-1] - closes[-2]) / closes[-2], 2),
            "chg_5d": round(100 * (closes[-1] - closes[-6]) / closes[-6], 2)}


def fetch_correlated_asset_moves(current_symbol: str, all_symbols: list[str]) -> dict[str, dict[str, float]]:
    """Percent moves of every other asset instead of raw bid/ask levels so intermarket shifts are interpretable."""
    api = _require_mt5()
    moves: dict[str, dict[str, float]] = {}
    for symbol in all_symbols:
        if symbol == current_symbol:
            continue
        try:
            if not api.symbol_select(symbol, True):
                continue
            rates = api.copy_rates_from_pos(symbol, config.TIMEFRAMES["D1"], 0, 8)
            if rates is not None and len(rates) >= 6:
                computed = _daily_moves_from_rates(rates)
                if computed is not None:
                    moves[symbol] = computed
        except Exception as exc:
            log.warning("[SYSTEM] Correlation data unavailable for %s: %s", symbol, exc)
    return moves
