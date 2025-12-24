#!/usr/bin/env python3
"""
sweep_config.py

Single source of truth for all sweepable parameters used by:
- dashboard.py (sidebar widgets)
- parameter_sweep.py (CLI sweep tool)

This module defines the canonical set of strategy parameters with their
default values, types, and metadata for validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict, fields
from typing import Any, Dict, List, Optional, Type, get_type_hints


@dataclass
class SweepConfig:
    """
    Single source of truth for all sweepable parameters.
    
    These parameters match the dashboard.py sidebar settings and are used
    by both the dashboard and the parameter sweep CLI tool.
    """
    
    # === Strategy Settings ===
    min_edge: float = 0.06
    """Minimum edge (model_prob - market_price) required to take a trade."""
    
    max_bets_per_expiry: int = 3
    """Maximum number of bets allowed per contract expiry date."""
    
    max_capital_per_expiry_frac: float = 0.15
    """Maximum fraction of bankroll to allocate to a single expiry."""
    
    max_capital_total_frac: float = 0.40
    """Maximum fraction of bankroll to allocate across all trades."""
    
    max_net_delta_frac: float = 0.20
    """Maximum net directional exposure (Long - Short) as fraction of bankroll."""
    
    min_price: float = 0.03
    """Minimum contract price to consider for trading."""
    
    max_price: float = 0.95
    """Maximum contract price to consider for trading."""
    
    min_model_prob: float = 0.0
    """Minimum model probability to consider for trading."""
    
    max_model_prob: float = 1.0
    """Maximum model probability to consider for trading."""
    
    use_stability_penalty: bool = True
    """Apply stability penalty to Kelly sizing based on curve fit quality."""
    
    correlation_penalty: float = 0.25
    """Penalty factor for correlated positions in same expiry/direction."""
    
    min_trade_frac: float = 0.01
    """Minimum trade size as fraction of bankroll."""
    
    # === Bankroll Settings ===
    kelly_fraction: float = 0.15
    """Fractional Kelly multiplier (0.15 = 15% Kelly)."""
    
    use_fixed_stake: bool = False
    """Use fixed dollar stake instead of Kelly sizing."""
    
    fixed_stake_amount: float = 10.0
    """Fixed stake amount in USD (when use_fixed_stake=True)."""
    
    bankroll: float = 500.0
    """Starting bankroll in USD."""
    
    # === DTE Filter ===
    use_max_dte: bool = True
    """Enable maximum days-to-expiry filter."""
    
    max_dte: float = 2.0
    """Maximum days to expiry for trade eligibility."""
    
    # === Probability Threshold Mode ===
    use_prob_threshold: bool = False
    """Use probability thresholds instead of edge-based trading."""
    
    prob_threshold_yes: float = 0.7
    """Trade YES when model probability >= this value."""
    
    prob_threshold_no: float = 0.3
    """Trade NO when model probability <= this value."""
    
    # === Moneyness Filter ===
    use_max_moneyness: bool = False
    """Enable moneyness filter."""
    
    min_moneyness: float = 0.0
    """Minimum absolute moneyness to consider."""
    
    max_moneyness: float = 0.05
    """Maximum absolute moneyness to consider."""
    
    def to_strategy_params(self) -> Dict[str, Any]:
        """
        Convert to strategy_params dict expected by BacktestEngine.
        
        Maps SweepConfig fields to the parameter names used by auto_reco.
        """
        params = {
            "kelly_fraction": self.kelly_fraction,
            "min_edge": self.min_edge,
            "max_bets_per_expiry": self.max_bets_per_expiry,
            "max_capital_per_expiry_frac": self.max_capital_per_expiry_frac,
            "max_capital_total_frac": self.max_capital_total_frac,
            "max_net_delta_frac": self.max_net_delta_frac,
            "min_price": self.min_price,
            "max_price": self.max_price,
            "min_model_prob": self.min_model_prob,
            "max_model_prob": self.max_model_prob,
            "use_stability_penalty": self.use_stability_penalty,
            "correlation_penalty": self.correlation_penalty,
            "min_trade_frac": self.min_trade_frac,
            "allow_no": True,  # Always allow NO trades
            "use_fixed_stake": self.use_fixed_stake,
            "fixed_stake_amount": self.fixed_stake_amount,
            "use_prob_threshold": self.use_prob_threshold,
            "prob_threshold_yes": self.prob_threshold_yes,
            "prob_threshold_no": self.prob_threshold_no,
        }
        
        # Conditional filters
        if self.use_max_dte:
            params["max_dte"] = self.max_dte
        else:
            params["max_dte"] = None
            
        if self.use_max_moneyness:
            params["max_moneyness"] = self.max_moneyness
            params["min_moneyness"] = self.min_moneyness
        else:
            params["max_moneyness"] = None
            params["min_moneyness"] = None
            
        return params
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SweepConfig":
        """Create SweepConfig from dictionary, ignoring unknown keys."""
        valid_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)
    
    def update(self, overrides: Dict[str, Any]) -> "SweepConfig":
        """Return new SweepConfig with overrides applied."""
        current = asdict(self)
        current.update(overrides)
        return SweepConfig.from_dict(current)


def get_parameter_names() -> List[str]:
    """Return list of all valid parameter names."""
    return [f.name for f in fields(SweepConfig)]


def get_parameter_defaults() -> Dict[str, Any]:
    """Return dict of parameter names to their default values."""
    return asdict(SweepConfig())


def get_parameter_types() -> Dict[str, Type]:
    """Return dict of parameter names to their types."""
    return get_type_hints(SweepConfig)


def validate_parameter_name(name: str) -> bool:
    """Check if a parameter name is valid."""
    return name in get_parameter_names()


def parse_parameter_value(name: str, value_str: str) -> Any:
    """
    Parse a string value to the appropriate type for the given parameter.
    
    Args:
        name: Parameter name
        value_str: String value to parse
        
    Returns:
        Parsed value in appropriate type
        
    Raises:
        ValueError: If parsing fails or parameter is unknown
    """
    if not validate_parameter_name(name):
        valid = ", ".join(sorted(get_parameter_names()))
        raise ValueError(f"Unknown parameter '{name}'. Valid parameters: {valid}")
    
    param_types = get_parameter_types()
    param_type = param_types.get(name, str)
    
    if param_type == bool:
        # Handle boolean parsing
        if value_str.lower() in ("true", "1", "yes", "on"):
            return True
        elif value_str.lower() in ("false", "0", "no", "off"):
            return False
        else:
            raise ValueError(f"Cannot parse '{value_str}' as boolean for {name}")
    elif param_type == int:
        return int(value_str)
    elif param_type == float:
        return float(value_str)
    else:
        return value_str
