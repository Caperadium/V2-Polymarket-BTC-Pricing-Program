# FAQ

## General

### What does this system do?

It prices Bitcoin binary options (e.g., "BTC above $90k on Dec 31?") on Polymarket, generates trade recommendations using fractional Kelly sizing, and can execute trades via the Polymarket CLOB API.

### Is this a trading bot?

No — it generates recommendations. The Polymarket Console lets an operator review, approve, and submit trades manually. Full automation is not implemented.

### What markets does it support?

Any Polymarket binary contract with a numeric strike price resolvable to a BTC price level. The slug pattern system supports date-range discovery.

## Pricing

### How accurate are the probabilities?

The pricing engine uses GARCH + jump diffusion Monte Carlo — a standard quantitative finance approach. Calibration via logit-shift and signal diagnostics (Spearman, AUC) measure real-world performance. Results vary by market conditions.

### Why use GARCH instead of Black-Scholes?

BTC volatility is time-varying and fat-tailed. GARCH captures volatility clustering; Student-t errors handle fat tails; jump diffusion adds crash risk. Standard Black-Scholes with constant vol would systematically misprice tail events.

### How many Monte Carlo paths?

Default is 50,000. More paths = smoother probabilities but slower runtime. The backrunner uses fewer (10,000) for speed.

## Strategy

### What edge threshold should I use?

Start at 0.06 (6 cents). Run parameter sweeps to find the optimal value for your data. Higher thresholds mean fewer but higher-quality trades.

### Why fractional Kelly (0.15) vs full Kelly?

Full Kelly maximizes theoretical log-wealth growth but is extremely volatile and sensitive to estimation error. 15% Kelly (¼ to ½ Kelly is standard in practice) trades lower returns for much lower drawdown risk.

### What does the stability penalty do?

It reduces position size for contracts where the logistic curve fit is poor, the monotonicity constraint is violated, or the edge is a statistical outlier. This prevents the strategy from over-allocating to noisy signals.

### When should I use prob_threshold mode vs edge mode?

- **Edge mode** (default): Trade when `model_prob − market_price ≥ min_edge`. Best when you trust the model's absolute probability calibration.
- **Prob threshold mode**: Trade when `model_prob ≥ 0.70` (YES) or `≤ 0.30` (NO). Best when you trust the model's directional calls but not its exact probabilities.

## Backtesting

### Why use decile-conditioned shuffle tests?

Simple outcome shuffling within an expiry can be too easy to beat — it only tests whether you picked good strikes within an expiry. Decile conditioning tests whether you generated edge at all: the null model draws outcomes from contracts with similar edge magnitudes but potentially different strikes/expiries.

### My Z-score is negative — what does that mean?

Your strategy performed worse than random shuffling. Either the edge signal is inverted, transaction costs are eating profits, or the strategy has a structural flaw. Check signal diagnostics for the answer.

## Operations

### How often should I run the pipeline?

Depends on DTE. For contracts ≤2 days to expiry, run every 4 hours. For longer expiries, daily suffices. The staleness system will block entries after 12 hours.

### What happens if the vol gate triggers?

- **HIGH regime**: Kelly sizing halved, edge requirement +2¢
- **EXTREME regime**: All new entries blocked, existing positions can exit
- **SHOCK detected**: Immediate extreme regime

### Can I backtest without live Polymarket data?

Yes — use the prob backrunner with `old_market_prices.csv` and `--skip-data-fetch`. The engine simulates everything locally.
