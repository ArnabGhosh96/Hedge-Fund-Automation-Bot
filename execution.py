"""Risk sizing and guarded MT5 order submission."""
import logging
import math
from typing import Any

import config
try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

log = logging.getLogger(__name__)


def _api():
    if mt5 is None:
        raise RuntimeError("MetaTrader5 package is unavailable; install it on the MT5 Windows host.")
    return mt5


def _symbol_info(symbol: str):
    if not mt5.symbol_select(symbol, True):
        raise RuntimeError(f"Unable to select {symbol}: {mt5.last_error()}")
    info = mt5.symbol_info(symbol)
    if info is None:
        raise RuntimeError(f"No symbol metadata for {symbol}: {mt5.last_error()}")
    return info


def calculate_position_size(symbol: str, stop_loss_price: float, risk_percent: float, equity: float) -> float:
    info = _symbol_info(symbol)
    tick = mt5.symbol_info_tick(symbol)
    if tick is None or equity <= 0:
        raise RuntimeError(f"No valid tick/equity for sizing {symbol}")
    risk_percent = min(max(float(risk_percent), 0.01), config.MAX_RISK_PERCENT)
    reference = float(tick.ask if abs(float(stop_loss_price) - tick.ask) < abs(float(stop_loss_price) - tick.bid) else tick.bid)
    distance = abs(reference - float(stop_loss_price))
    tick_size, tick_value = float(info.trade_tick_size or 0), float(info.trade_tick_value_loss or info.trade_tick_value or 0)
    if tick_size <= 0 or tick_value <= 0 or distance <= 0:
        raise ValueError(f"Invalid stop distance or tick economics for {symbol}")
    loss_per_lot = (distance / tick_size) * tick_value
    raw_volume = equity * risk_percent / 100.0 / loss_per_lot
    step, minimum, maximum = float(info.volume_step), float(info.volume_min), float(info.volume_max)
    if step <= 0 or minimum <= 0 or maximum < minimum:
        raise ValueError(f"Invalid volume limits for {symbol}")
    # Round down so the requested risk is never increased by step rounding.
    volume = math.floor(raw_volume / step) * step
    if volume < minimum:
        raise ValueError(f"Calculated volume {volume} is below broker minimum {minimum}; refusing to exceed requested risk")
    volume = min(volume, maximum)
    digits = max(0, min(8, round(-math.log10(step)) if step < 1 else 0))
    return round(volume, digits)


def get_open_positions(symbol: str | None = None) -> list[Any]:
    result = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
    if result is None:
        raise RuntimeError(f"MT5 positions_get failed: {mt5.last_error()}")
    return list(result)


def _fill_modes(info) -> list[int]:
    modes = [getattr(mt5, "ORDER_FILLING_IOC", 1), getattr(mt5, "ORDER_FILLING_FOK", 0), getattr(mt5, "ORDER_FILLING_RETURN", 2)]
    preferred = int(getattr(info, "filling_mode", modes[0]))
    # Try the symbol-preferred value first, then standard alternatives.
    return list(dict.fromkeys([preferred] + modes))


def _send_with_fill_fallback(request: dict[str, Any], info):
    last = None
    for fill in _fill_modes(info):
        request["type_filling"] = fill
        result = mt5.order_send(request)
        if result is None:
            last = mt5.last_error()
            continue
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            return result
        last = f"retcode={result.retcode} comment={getattr(result, 'comment', '')}"
        # Only retry filling-mode rejection; other responses are not helped by changing fill type.
        if result.retcode not in {getattr(mt5, "TRADE_RETCODE_INVALID_FILL", 10030)}:
            break
    raise RuntimeError(f"MT5 order rejected: {last}")


def close_position(ticket_or_pos: Any) -> dict[str, Any]:
    ticket = int(getattr(ticket_or_pos, "ticket", ticket_or_pos))
    positions = mt5.positions_get(ticket=ticket)
    if not positions:
        raise RuntimeError(f"Position {ticket} not found or already closed")
    position = positions[0]
    info = _symbol_info(position.symbol)
    tick = mt5.symbol_info_tick(position.symbol)
    if tick is None:
        raise RuntimeError(f"No tick to close {position.symbol}")
    is_buy = position.type == mt5.POSITION_TYPE_BUY
    order_type = mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY
    price = float(tick.bid if is_buy else tick.ask)
    request = {"action": mt5.TRADE_ACTION_DEAL, "symbol": position.symbol, "position": position.ticket, "volume": float(position.volume),
               "type": order_type, "price": price, "deviation": 20, "magic": 260019, "comment": "Autonomous risk close", "type_time": mt5.ORDER_TIME_GTC}
    result = _send_with_fill_fallback(request, info)
    return {"ticket": int(position.ticket), "deal": int(result.deal), "order": int(result.order), "price": float(result.price), "volume": float(position.volume), "retcode": int(result.retcode)}


def execute_trade(symbol: str, signal: str, stop_loss: float, take_profit: float, risk_percent: float) -> dict[str, Any]:
    signal = signal.upper()
    if signal not in {"BUY", "SELL"}:
        raise ValueError("Signal must be BUY or SELL")
    info, tick = _symbol_info(symbol), mt5.symbol_info_tick(symbol)
    account = mt5.account_info()
    if tick is None or account is None or account.equity <= 0:
        raise RuntimeError(f"Missing tick or account equity for {symbol}")
    price = float(tick.ask if signal == "BUY" else tick.bid)
    if not ((signal == "BUY" and stop_loss < price < take_profit) or (signal == "SELL" and take_profit < price < stop_loss)):
        raise ValueError(f"Invalid {signal} stop/target geometry for {symbol} at {price}")
    volume = calculate_position_size(symbol, stop_loss, risk_percent, float(account.equity))
    request = {"action": mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": volume,
               "type": mt5.ORDER_TYPE_BUY if signal == "BUY" else mt5.ORDER_TYPE_SELL, "price": price,
               "sl": float(stop_loss), "tp": float(take_profit), "deviation": 20, "magic": 260019,
               "comment": "AI risk-managed entry", "type_time": mt5.ORDER_TIME_GTC}
    result = _send_with_fill_fallback(request, info)
    log.info("[TRADE] %s %s vol=%s entry=%s deal=%s", signal, symbol, volume, result.price, result.deal)
    return {"deal": int(result.deal), "order": int(result.order), "position_id": int(getattr(result, "order", 0)),
            "price": float(result.price), "volume": float(volume), "stop_loss": float(stop_loss), "take_profit": float(take_profit)}
