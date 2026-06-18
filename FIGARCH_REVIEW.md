# FIGARCH Implementation Review

## What it is

FIGARCH (Fractionally Integrated GARCH) replaces standard GARCH(1,1) variance recursion with long-memory decay. Located in `core/pricing/btc_pricing_engine.py`.

## How it works

### Standard GARCH(1,1) variance update (line 628)

```
σ²_t = ω + α·ε²_t + β·σ²_{t-1}
```

This has **exponential decay** — each shock's influence decays geometrically at rate β per step. After K steps, shock weight is β^K. For BTC hourly data β≈0.85, half-life ~4.5 hours — shock memory dies fast.

### FIGARCH variance update (line 625)

```
σ²_t = ω/(1-β) + Σ_{k=0}^{K-1} λ_k · ε²_{t-k}
```

This has **hyperbolic decay** — each past squared return ε²_{t-k} gets weight λ_k from binomial expansion of (1-L)^d. For d=0.578, decay follows power law ~k^(d-1) = k^(-0.422). Shock influence persists orders of magnitude longer than exponential.

## Weight computation (`_compute_figarch_weights`, line 293)

Binomial expansion recurrence:

- λ₀ = 1
- λ_k = λ_{k-1} · (k - 1 - d) / k, for k ≥ 1

Truncated at K=1000 lags. d=0.578 (Siu 2025 BTC empirical estimate, SE=0.271).

## Simulation mechanics (line 611-628)

Each hourly step:

1. Compute ε²_t (squared innovation after drift + volatility scaling)
2. **FIGARCH path**: Shift `past_eps_sq` buffer (n_sims × 1000), insert ε²_t at position 0, compute weighted sum across all 1000 lags → `σ²_t = ω/(1-β) + Σ λ_k ε²_{t-k}`
3. **GARCH path**: `σ²_t = ω + α·ε²_t + β·σ²_{t-1}` (2-parameter recursion, no buffer)

FIGARCH path keeps rolling window of last 1000 squared returns per simulation path. Buffer initialized with unconditional variance: `ω/(1-α-β)`.

## Fitting (line 334-399)

Same `arch_model` GARCH(1,1) fit for both modes. `omega`, `alpha`, `beta`, `nu`, `mu` all from standard fit. In FIGARCH mode, the fitted parameters are **reinterpreted** — alpha and beta are ignored in the variance recursion, replaced by precomputed λ_k weights. Only `omega/(1-β)` intercept and structural mu carry over.

## Key differences from standard GARCH

| Aspect | GARCH(1,1) | FIGARCH (this impl) |
|---|---|---|
| **Decay** | Exponential (β^t) | Hyperbolic (~t^(d-1)) |
| **Memory** | Short — ~4.5h half-life | Long — ~1000h / 42 days reach |
| **Parameters** | ω, α, β (3) | ω, β, d (3) — α unused |
| **State** | σ²_t only (scalar per path) | σ²_t + 1000-lag ε² buffer (vector per path) |
| **Persistence** | Stationary if α+β<1 | Always non-stationary (infinite memory in limit) |
| **Cost** | O(1) per step | O(K) per step, K=1000 |
| **β role** | GARCH persistence | Intercept normalizer only: ω/(1-β) |

## Why it matters for BTC options

BTC volatility exhibits **long-range dependence** — large vol events cluster across weeks, not hours. Standard GARCH's exponential decay misses this. FIGARCH's hyperbolic weights mean a volatility spike from 3 days ago still meaningfully influences today's variance estimate, producing fatter-tailed return distributions for short-dated contracts (≤7 days where it's enabled).

## Horizon gating

FIGARCH only active for T ≤ 7 days (full model tier).

| Horizon | Model Configuration |
|---|---|
| T > 90 days | Naive prior only (μ=0, GARCH+Student-t, no jumps) |
| 30 < T ≤ 90 days | Naive prior + simplified (GARCH+t, Kou jumps, no SVCJ/FIGARCH/skewed-t) |
| 7 < T ≤ 30 days | Intermediate (all features except skewed-t) |
| T ≤ 7 days | **Full model** (SVCJ, skewed-t, FIGARCH all enabled) |

Short-dated contracts see the most complex dynamics because jump effects dominate and volatility persistence matters most.

## Simplified specification note

The implementation uses a simplified FIGARCH form — standard FIGARCH(1,d,1) would apply (1-βL)⁻¹ to the ARCH recursion, giving AR(1) feedback on variance. This implementation omits that AR(1) feedback, keeping only the fractional differencing weights. For binary option pricing, the long-memory parameter d dominates; the AR(1) feedback is second-order.
