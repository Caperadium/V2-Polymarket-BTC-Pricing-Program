"""Reusable Streamlit filter controls shared across dashboard tabs."""
from __future__ import annotations
from typing import Any, Dict, Optional

import streamlit as st


def moneyness_filter_controls(
    container: Any,
    *,
    key_prefix: str,
    default_enabled: bool = False,
    default_mode: str = "abs",
    default_lower: float = 0.0,
    default_upper: float = 0.05,
) -> Dict[str, Optional[float]]:
    """Render the moneyness filter control. Returns:
        {"enabled": bool, "mode": "abs"|"signed",
         "min_moneyness": float|None, "max_moneyness": float|None}
    Bounds are None when disabled. In 'signed' mode, lower may be negative.
    """
    enabled = container.checkbox(
        "Limit Moneyness", value=default_enabled, key=f"{key_prefix}_mny_on"
    )
    mode = container.radio(
        "Moneyness mode",
        options=["abs", "signed"],
        index=0 if default_mode == "abs" else 1,
        horizontal=True,
        disabled=not enabled,
        key=f"{key_prefix}_mny_mode",
        help="abs = |moneyness| (symmetric). signed = raw moneyness; OTM>0, ITM<0.",
    )
    signed = mode == "signed"
    lower = container.number_input(
        "Lower bound" if signed else "Min |Moneyness|",
        min_value=-0.5 if signed else 0.0,
        max_value=0.5,
        value=float(default_lower),
        step=0.01,
        format="%.2f",
        disabled=not enabled,
        key=f"{key_prefix}_mny_lo",
        help=(
            "Keep contracts with moneyness >= this (use 0.0 for OTM-only, negative for ITM)."
            if signed
            else "Keep contracts with |moneyness| >= this (exclude ATM)."
        ),
    )
    upper = container.number_input(
        "Upper bound" if signed else "Max |Moneyness|",
        min_value=-0.5 if signed else 0.0,
        max_value=0.5,
        value=float(default_upper),
        step=0.01,
        format="%.2f",
        disabled=not enabled,
        key=f"{key_prefix}_mny_hi",
        help=(
            "Keep contracts with moneyness <= this (cap extreme OTM)."
            if signed
            else "Keep contracts with |moneyness| <= this. 0.05 = ±5% from spot."
        ),
    )
    if not enabled:
        return {"enabled": False, "mode": "abs", "min_moneyness": None, "max_moneyness": None}
    return {"enabled": True, "mode": mode, "min_moneyness": float(lower), "max_moneyness": float(upper)}
