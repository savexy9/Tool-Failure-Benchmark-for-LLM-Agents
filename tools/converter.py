"""
Unit converter tool with fault injection.

Supports real conversions between common unit pairs:
  Temperature: °F ↔ °C, K ↔ °F, K ↔ °R
  Distance:    mi ↔ km, m ↔ mi, cm ↔ in, nmi ↔ km
  Weight:      lb ↔ kg, oz ↔ g, g ↔ lb, st ↔ lb
  Volume:      gal ↔ L
  Area:        ha ↔ acre

Supports four fault modes: silent_wrong, error, malformed, empty.
"""

from __future__ import annotations

import re
import random as _random

# ── Conversion registry ──────────────────────────────────────────────────────
# Each entry: (pattern_regex, conversion_function, from_unit, to_unit)

_CONVERSIONS: list[tuple[str, callable, str, str]] = []


def _register(pattern: str, from_unit: str, to_unit: str):
    """Decorator to register a conversion function."""
    def decorator(fn: callable) -> callable:
        _CONVERSIONS.append((pattern, fn, from_unit, to_unit))
        return fn
    return decorator


# ── Temperature ──────────────────────────────────────────────────────────────

@_register(r"^(-?[\d.]+)\s*[fF](?:ahrenheit)?$", "F", "C")
def f_to_c(v: float) -> float:
    return (v - 32) * 5 / 9

@_register(r"^(-?[\d.]+)\s*[cC](?:elsius|entigrade)?$", "C", "F")
def c_to_f(v: float) -> float:
    return v * 9 / 5 + 32

@_register(r"^(-?[\d.]+)\s*[kK](?:elvin)?$", "K", "F")
def k_to_f(v: float) -> float:
    return (v - 273.15) * 9 / 5 + 32

@_register(r"^(-?[\d.]+)\s*[kK](?:elvin)?$", "K", "R")
def k_to_r(v: float) -> float:
    return v * 1.8

# ── Distance ─────────────────────────────────────────────────────────────────

@_register(r"^(-?[\d.]+)\s*mi(?:les?)?$", "mi", "km")
def mi_to_km(v: float) -> float:
    return v * 1.609344

@_register(r"^(-?[\d.]+)\s*km$", "km", "mi")
def km_to_mi(v: float) -> float:
    return v / 1.609344

@_register(r"^(-?[\d.]+)\s*m(?:eters?)?$", "m", "mi")
def m_to_mi(v: float) -> float:
    return v * 0.000621371

@_register(r"^(-?[\d.]+)\s*cm$", "cm", "in")
def cm_to_in(v: float) -> float:
    return v * 0.393701

@_register(r"^(-?[\d.]+)\s*nmi$", "nmi", "km")
def nmi_to_km(v: float) -> float:
    return v * 1.852

# ── Weight ───────────────────────────────────────────────────────────────────

@_register(r"^(-?[\d.]+)\s*(?:lb|lbs?|pounds?)$", "lb", "kg")
def lb_to_kg(v: float) -> float:
    return v * 0.453592

@_register(r"^(-?[\d.]+)\s*(?:oz|ounces?)$", "oz", "g")
def oz_to_g(v: float) -> float:
    return v * 28.3495

@_register(r"^(-?[\d.]+)\s*(?:g|grams?)$", "g", "lb")
def g_to_lb(v: float) -> float:
    return v * 0.00220462

@_register(r"^(-?[\d.]+)\s*(?:st|stone)$", "st", "lb")
def st_to_lb(v: float) -> float:
    return v * 14

# ── Volume ───────────────────────────────────────────────────────────────────

@_register(r"^(-?[\d.]+)\s*(?:gal|gallons?)$", "gal", "L")
def gal_to_l(v: float) -> float:
    return v * 3.78541

# ── Area ─────────────────────────────────────────────────────────────────────

@_register(r"^(-?[\d.]+)\s*(?:ha|hectares?)$", "ha", "acres")
def ha_to_acres(v: float) -> float:
    return v * 2.47105


# ── Main function ────────────────────────────────────────────────────────────

def tool_converter(value_and_unit: str, fault: str | None = None) -> str:
    """Convert a value from one unit to another, optionally injecting a fault.

    Args:
        value_and_unit: String like "100 F", "5 mi", "300 K".
        fault: Fault type to inject.

    Returns:
        Converted value as a string, or a fault-injected output.

    Raises:
        RuntimeError: When fault == "error".
    """
    if fault == "error":
        raise RuntimeError("Converter crashed")

    if fault == "malformed":
        return "NaN units??"

    if fault == "empty":
        return ""

    if fault == "silent_wrong":
        return "1000"

    # Try to match against registered conversions
    value_and_unit = value_and_unit.strip()
    for pattern, convert_fn, from_u, to_u in _CONVERSIONS:
        m = re.match(pattern, value_and_unit)
        if m:
            v = float(m.group(1))
            result = convert_fn(v)
            # Format: up to 6 significant figures, strip trailing zeros
            formatted = f"{result:.6f}".rstrip("0").rstrip(".")
            return formatted

    return "Error: unsupported conversion"
