# Market Making on Binary BTC Ladders

This chapter explains the market-making subsystem in `market_maker/` from first
principles. It assumes no prior knowledge of market making. It does assume you
know what a probability is, what Bitcoin is, and roughly what "the model" (the
Monte Carlo pricing engine in `core/pricing/`) produces -- a probability that
BTC finishes above a strike price. How that engine works internally is covered
in [Pricing Engine](pricing-engine.md); here we only care about how its
*outputs* are consumed.

Everything described here runs **paper-traded**: the bot quotes and tracks
positions as if it were live, but no real orders are ever sent to the venue.
Fills are simulated from the real, live order-flow stream under deliberately
pessimistic assumptions (Section 12).

---

## 1. What a market maker is

### 1.1 The order book

A market on Polymarket such as "Bitcoin above $118,000 on July 10?" trades a
**binary contract**: it pays $1 if the answer is YES and $0 if NO. Its price
therefore lives between 0 and 1 and can be read directly as a probability --
a price of 0.63 means the market collectively assigns roughly 63% to YES.

Trading happens through a **central limit order book**. At any moment the book
holds two stacks of resting orders:

- **Bids**: offers to *buy* at some price or lower. The highest bid is the
  **best bid**.
- **Asks** (or offers): offers to *sell* at some price or higher. The lowest
  ask is the **best ask**.

The gap between them is the **spread**. If the best bid is 0.61 and the best
ask is 0.65, the spread is 4 cents and the **mid price** is 0.63.

Participants come in two flavors:

- A **taker** wants to trade *now*. They cross the spread: buy at the ask or
  sell at the bid. They pay the spread as the price of immediacy.
- A **maker** rests orders in the book and waits. When a taker hits their
  order, the maker earns the spread -- they bought at their bid (below mid) or
  sold at their ask (above mid).

### 1.2 The market maker's business

A **market maker** (MM) is a professional maker: it continuously quotes *both*
a bid and an ask around what it believes the contract is worth (its **fair
value**), hoping to buy low from one taker and sell high to another, pocketing
the spread each round trip. Its ideal life is a steady stream of uninformed
traders arriving at random on both sides, so its position stays near zero and
its revenue is spread times volume.

Two things ruin this ideal, and essentially the entire `market_maker/` module
is machinery for dealing with them:

1. **Inventory risk.** Fills do not alternate neatly. If takers keep hitting
   the bid, the MM accumulates a long position ("inventory"). It is now no
   longer a neutral toll collector -- it is exposed to the contract's price
   moving against it. Binary contracts make this vicious near expiry: a
   position can jump from nearly-worthless to nearly-$1 on a single BTC move.

2. **Adverse selection.** Not all takers are uninformed. Some trade *because
   they know something* (or computed something) the MM has not priced yet --
   e.g. BTC just jumped and the MM's quotes are stale. Informed takers
   systematically pick off the side of the quote that is wrong. Each such
   fill is a loss booked at the moment of trade, even though it looks like
   normal spread revenue at the time.

Every classical MM technique maps onto one of these: *spreads* charge for both
risks, *inventory skew* leans quotes to shed a position, *widening and
pulling* defends against being picked off, and *position caps and sizing*
bound the worst case.

### 1.3 What is special about this venue

Three properties of Polymarket's `bitcoin-above` markets shape the design:

- **Prices are probabilities.** Fair value comes from a probability model, not
  from another market's price. The MM's edge, if any, is that its Monte Carlo
  pricer (GARCH + jumps + regimes) knows the *distribution* of BTC better than
  the crowd does.
- **Markets come in ladders.** One expiry date lists many strikes ($112k,
  $114k, $116k, ...). These are not independent: P(BTC > $114k) can never be
  less than P(BTC > $116k). The whole ladder is one object -- a survival curve
  of the same underlying random variable -- and must be quoted, repaired, and
  hedged as one (Section 9).
- **Hard expiry.** Every contract resolves at 12:00 ET on its date, YES if and
  only if spot is *strictly above* the strike. Everything the MM holds at that
  instant converts to $1 or $0 (Section 13).

---

## 2. System overview

The runner (`paper_runner.py`) drives a loop that "ticks" every few seconds.
One tick of `harness.PaperTradingLoop` does, in order:

1. Advance the clock.
2. Ingest order-book messages for every market in the ladder; update the
   liquidity monitor.
3. Refresh the **pricer snapshot** (the model's probability ladder) if due.
4. Compute the **fair-value anchor** -- a consensus between model and market.
5. Ask the **risk controller** for a per-market directive (quote both sides /
   one side / nothing).
6. Build quotes: **quote engine** (where to center, how wide at minimum) ->
   **sizing** (how many shares) -> **spread builder** (final widening) ->
   **ladder hedger** no-arbitrage repair -> hedge size-skew -> **order
   lifecycle** (place/cancel/amend the minimal set of orders).
7. Feed the same book data to the **fill simulator**; any simulated fills are
   routed atomically into the inventory manager and the durable state store.
8. Compute *next* tick's hedge recommendations from post-fill inventory.

The ordering is deliberately **decide-then-observe**: quotes are decided from
the state known at the start of the tick, and fills against them are observed
afterward. This bakes in a one-tick reaction lag, matching the latency a real
venue imposes -- the risk controller can only respond to a fill on the
following tick, never instantly.

The remainder of this chapter walks through these components in the order risk
flows through them: fair value (Section 3-4), quoting (5-7), sizing (8),
ladder coherence (9), inventory (10), risk directives (11), simulated fills
(12), settlement (13), measurement (14), and operations (15).

---

## 3. From the pricing engine to a snapshot

### 3.1 What the engine hands over

The only component allowed to call the Monte Carlo pricer is
`pricer_adapter.build_snapshot`. Once per expiry per refresh interval it calls
`calculate_probabilities(strikes, hours_to_expiry)` and receives:

- `{strike: p}` -- for each strike K, the simulated probability
  P(S_T > K) that BTC's settlement price exceeds K. Because every strike is
  evaluated against the *same* set of simulated price paths, this ladder is
  monotone non-increasing in K by construction.
- `_meta` -- among other things `n_sims` (number of Monte Carlo paths), `S0`
  (spot used), and `horizon_gate_active`.

The adapter asks the engine for more than the quoted strikes: it densifies the
grid with the midpoints between adjacent quoted strikes. The dense grid
(`p_grid`) later serves as the smooth model curve the no-arb repair projects
toward; the quoted subset (`p_hat`) is what quoting actually uses.

### 3.2 Statistical uncertainty: sigma2

Quoting and sizing both need to know *how uncertain* each probability is. For
a Monte Carlo estimate this has an exact answer: each simulated path either
finishes above K or not -- a Bernoulli trial -- so the variance of the
estimated probability is

```
sigma2_mc(K) = p_hat(K) * (1 - p_hat(K)) / n_sims
```

`n_sims` is always taken from the engine's own `_meta`, never assumed.

There is a subtlety at the **wings** (strikes far from spot, p near 0 or 1):
`p*(1-p)` collapses there, so the MC standard error is tiny -- yet the wings
are exactly where the *model itself* is least trustworthy (tail behavior
depends heavily on estimated jump and vol parameters). MC error measures
sampling noise, not model error. So for wing strikes the adapter substitutes a
**parameter-uncertainty** variance from the Bayesian posterior channel
(`core/pricing/bayesian_estimation.py`): it takes the 90% credible band
[q05, q95] on the probability and converts width to a normal-equivalent
standard deviation,

```
sigma = (q95 - q05) / 3.29        sigma2 = sigma^2
```

(3.29 is the width of a 90% interval in standard deviations for a normal).
The substitution only ever *increases* uncertainty, never lowers it below the
MC floor, and is cached for an hour because posteriors move on
parameter-estimation timescales, not tick timescales. The ladder-level scalar
`sigma2_ladder = max over strikes` is the "common-mode" uncertainty used by
sizing (Section 8).

### 3.3 Trust decays with horizon: confidence tiers

Backtests validated the pricer out to about 7 days to expiry and showed known
weaknesses beyond. Time-to-expiry (tte) therefore maps to a **confidence
tier**:

| tte (days) | Tier |
|---|---|
| <= 7 | FULL |
| 7-14 | DEGRADED |
| 14-30 | MINIMAL |
| > 30 | NAIVE_GATED |

By deliberate design the tier gates exactly one thing: the wing-widening
multiplier in the spread builder (Section 7, term 5) -- 1.0x, 1.5x, 2.0x, 3.0x
respectively. Less trust in the model's tails, more charged to trade them.

### 3.4 Staleness

A snapshot also carries a `stale` flag: true if the engine reported itself
stale or if the snapshot is older than `pricer_max_age_s` (300 s). Stale model
output does not stop the loop -- it feeds the risk controller, which first
widens quotes and eventually pulls them (Section 11).

---

## 4. Fair value: blending model and market

### 4.1 Why not just quote the model?

The naive design centers quotes on the pricer's `p_hat`. That is fragile in
both directions: when the model is wrong the MM gets picked off with size, and
when the market is briefly wrong the MM never leans into it. The market mid is
itself an aggregation of other people's information; ignoring it discards
evidence.

The **fair-value anchor** (`fair_value_anchor.py`) instead computes a
*consensus* between two "experts" -- the pricer's ladder and the market-mid
ladder -- with weights that adapt over time based on which expert has been
predicting better. The scheme (after Beuoy) treats each expert as a bettor
with a **bankroll**; bankroll is credibility.

### 4.2 Buckets: making the ladder a distribution

A strike ladder p(K_1) >= p(K_2) >= ... >= p(K_n) is a survival curve. The
differences between adjacent entries are probabilities of *mutually exclusive*
outcomes -- "BTC finishes between K_j and K_j+1":

```
buckets = [ 1 - p(K_1),  p(K_1) - p(K_2),  ...,  p(K_n-1) - p(K_n),  p(K_n) ]
```

These n+1 numbers are non-negative and sum to 1: a genuine probability
distribution over "which band BTC lands in". All consensus math happens in
bucket space, and the result is integrated back into a ladder (which is
monotone by construction). This is what makes the consensus *ladder-
consistent*: you cannot get an arbitrageable ladder out of valid buckets.

### 4.3 Consensus and the bankroll update

With normalized bankroll weights w_i (starting 50/50), the consensus is simply
the weighted average bucket distribution:

```
consensus_bucket = sum_i  w_i * bucket_i
```

Then, each refresh, every expert is **marked to market**: the new consensus is
treated as the best available proxy for reality, and each expert's bankroll is
multiplied by how well its *previous* forecast anticipated it:

```
b_i  <-  b_i * sum_over_buckets( consensus_new * p_i_prev / consensus_prev )
```

This is a Bayes-style update: an expert whose previous distribution put mass
where the consensus subsequently moved gains bankroll; one that didn't loses
it. Weights are renormalized and floored at 0.02 so no expert can be
permanently silenced -- if the pricer has a bad week, it keeps a toehold from
which it can earn credibility back.

The pricer's weight is exported as **credibility** in [0, 1]. It is used twice
downstream: the spread builder widens quotes when credibility is low
(Section 7, term 4), and it is a first-class monitoring quantity. (Since
package B2, 2026-07-15, this whole update -- consensus, mark-to-market,
credibility -- runs independently for two regions rather than once per
expiry; Section 4.5.)

### 4.4 Degeneracy and freezing

If inputs are broken (non-finite mids, empty ladder) or the consensus escapes
its sanity bound (per strike it must lie *between* the SANITIZED pricer and
market ladders -- each raw ladder round-tripped through the bucket transform,
which repairs non-monotone inputs such as crossed venue mids -- a weighted
average always does, so a violation means a numeric bug), the anchor falls
back to a fixed 50/50 blend and **freezes** the bankrolls (both regions'
bankrolls together, since package B2 -- a fallback is a whole-ladder event)
so garbage ticks cannot corrupt the learned weights. The freeze auto-clears
after 20 consecutive clean recomputes (again, both regions together). A
frozen bankroll is also surfaced in the heartbeat and pages the operator
(Section 15).

Why sanitized, not raw (2026-07-14 fix): the consensus is built in bucket
space from the sanitized ladders, so checking it against the *raw* inputs
made one crossed mid pair anywhere in the ladder (routine on thin far-expiry
wing books) a guaranteed per-strike violation at *every* strike -- the bucket
renormalization is global. In production that froze a 3-DTE ladder's bankroll
permanently: the anchor method was never BEUOY, so the 20-clean-tick
auto-unfreeze could never start. Against the sanitized band the consensus is
provably in-band (it is a convex combination of the sanitized ladders), so
the check is a pure numeric safety net and crossed mids no longer freeze
anything.

One operational subtlety: the consensus only recomputes when *every* market in
the ladder has a valid mid. On ticks where it cannot recompute, the previous
fair value is reused -- and its *age* becomes a risk trigger of its own
(`FAIR_VALUE_STALE`, Section 11), because quoting around a frozen anchor while
BTC moves is precisely how an MM gets run over.

### 4.5 Per-region credibility (belly vs. wing)

Sections 4.1-4.4 describe *one* scalar credibility per expiry. In practice
model skill is not uniform across the ladder: a live audit of FV against
market mid (221k/387k quote snapshots) found the pricer running about +0.8c
rich in the OTM wings and -2.9c cheap deep ITM, monotone in moneyness, while
a belly win could still push the pricer's *overall* bankroll weight as high
as 0.979 (jul-20 ladder) -- one scalar credibility let a belly win buy the
pricer unchecked authority in exactly the region where it was measurably
wrong. Package B2 (2026-07-15) splits the single `BankrollState` into two,
keyed `"belly"`/`"wing"` (`fair_value_anchor.BELLY_REGION`/`WING_REGION`),
each updated by the identical bankroll mechanics of Section 4.3 but
restricted to its own region's buckets.

**Region assignment.** Each strike is classified belly or wing from the
*sanitized market ladder* (the market's own mid, bucket-round-tripped)
through the same `config.in_belly_band` boundary the spread builder and
markout report use -- never from the consensus being built (that would be
self-referential) and never from the pricer (a rich pricer tail must not be
able to reclassify a strike into the region where it holds more credibility).
n strikes produce n+1 buckets; the two *tail* buckets (all mass below the
lowest strike, all mass above the highest) are always `wing`, regardless of
how the extreme strikes themselves classify -- the open tails are
definitionally the wings, and on an all-belly-strike ladder they could
otherwise carry the majority of the mass and pollute belly evidence. Each
interior bucket takes the region of its left-hand strike.

**Two-phase update, each refresh.** The pre-update per-region weights build
a ladder-space consensus (below); that consensus becomes buckets and feeds,
per region, the same Bayes factor as Section 4.3 but summed only over that
region's own buckets. A region with zero assigned buckets, or whose factor
sum is non-positive or non-finite, SKIPS its own update entirely -- its
weights and update count are left unchanged, with no fallback and no freeze
(tail buckets always belonging to `wing` means only `belly` can ever be
empty, on an all-wing-strike ladder). Only non-skipped regions renormalize
and re-floor their weights. The FINAL consensus is then rebuilt with the
POST-update weights -- that is the tick's fair value.

**Ladder-space consensus.** Each strike's consensus price blends the
sanitized pricer and market ladders using *that strike's region's* weights,
then (a) a cummin repair enforcing non-increasing values, THEN (b) a
pointwise clamp into the sanity band of Section 4.4. Order matters here:
clamping before repairing can push values back out of band, while repairing
first and clamping second keeps the result monotone and in-band by
construction -- so the Section 4.4 sanity check still passes by construction
and remains a pure numeric safety net.

**Credibility.** `FairValue.credibility_by_region` carries both regions'
pricer weights; the legacy scalar `FairValue.credibility` becomes a
strike-count-weighted average of the two, so every existing consumer that
reads `credibility` keeps working unchanged. The spread builder's term 4
robust widening (Section 7) and term 7's per-side harness wiring both look
up the region-appropriate credibility for each strike.

**Resume and seeding.** On resume, a state db already carrying per-region
bankroll rows loads each directly. A state db carrying only the pre-B2
legacy (single, unkeyed) row -- the first restart after this deploy --
seeds `belly` from those legacy bankrolls (its credibility was legitimately
earned) and resets `wing` to 50/50 parity (the measured wing bias says that
authority was not legitimately earned; parity is the honest prior). A
brand-new expiry still starts both regions at 50/50.

This is a mitigation of the measured skew, not a fix for it: Beuoy marks
bankrolls to the *consensus*, not to realized settlement, so even at wing
parity the wing consensus still carries roughly half the pricer's own
richness. The root fix is pricer tail recalibration (out of scope here);
package B2 only stops one scalar from giving the pricer unchecked authority
in the region it is worst at, and spread term 7 (Section 7) covers the
residual at the quote layer.

---

## 5. A change of coordinates: log-odds space

Prices of binaries live in [0, 1], which is an awkward space to do quoting
math in: a 1-cent move at p = 0.50 is trivial, while the same 1-cent move at
p = 0.98 is a large event; and any additive formula risks stepping outside
[0, 1]. The standard fix is to work in **log-odds** (logit):

```
x = logit(p) = ln( p / (1 - p) )        p = S(x) = 1 / (1 + e^-x)
```

x ranges over the whole real line, moves in x are comparably meaningful
everywhere, and mapping back through the sigmoid S can never leave [0, 1].
The derivative

```
S'(x) = p * (1 - p)
```

appears repeatedly: it is the "exchange rate" between x-space and price space,
largest at p = 0.5 and vanishing at the extremes. All quoting math below
happens in x; only at the very end are prices converted back (with a clamp to
[0.001, 0.999] so the logit never blows up).

The **belief volatility** `sigma_b` is the volatility *of the fair value
itself* in x-space -- how fast the consensus log-odds wanders per square-root
day. It is estimated by an exponentially weighted moving average (EWMA,
lambda = 0.94) of squared increments of the consensus-x series, floored and
capped, and -- importantly -- estimated on a series *subsampled to at least
300 s intervals*. Raw tick-by-tick diffs annualize book jitter into an
absurdly inflated sigma_b (an early shadow run measured ~100x inflation and
63-cent spreads from this alone); belief vol is a minutes-scale quantity.

---

## 6. The quote engine: where to stand

### 6.1 The Avellaneda-Stoikov idea

The canonical model of optimal market making (Avellaneda and Stoikov, 2008)
answers two questions for a dealer with risk aversion gamma facing takers who
arrive less often the further your quote is from mid (arrival rate decaying
like e^(-k * distance)):

1. **Where is my personal indifference price?** Not at fair value, if I hold
   inventory. Holding q units, every further unit of exposure hurts more, so
   my *reservation price* sits below fair value when I am long (I would pay up
   to shed risk) and above when short. The displacement grows with risk
   aversion, with variance, and with how long I am stuck holding (time to
   expiry).

2. **How wide do I quote around it?** Wide enough that the spread earned
   compensates the risk of holding what a fill gives me, balanced against the
   fact that wider quotes fill less often. The optimum splits into a risk term
   (gamma, variance, time) and a "microstructure" term that depends only on
   the arrival decay k.

This system implements the Dalen adaptation of AS to *binary* markets, quoting
in log-odds. Per contract (`quote_engine.make_quote`, variant `"dalen"`):

```
reservation:   r_x     = x_fair - q * gamma * sigma_b^2 * tte
half-spread:   delta_x = 1/2 * ( gamma * sigma_b^2 * tte  +  (2/k) * ln(1 + gamma/k) )

x_bid = r_x - delta_x        x_ask = r_x + delta_x
p_bid = S(x_bid)             p_ask = S(x_ask)
```

Read the reservation formula slowly, because it is the heart of inventory
management:

- `x_fair` is the consensus fair value in log-odds (Section 4).
- `q` is the (normalized) signed inventory. Long (q > 0) pushes the whole
  quote pair *down*: the bid gets less aggressive (buy less) and the ask gets
  more aggressive (sell more). The book itself now works to flatten the
  position. This skew is the primary, continuous inventory-control mechanism;
  the hard caps of Section 10 are the backstop.
- `gamma * sigma_b^2 * tte` scales the skew: more risk aversion, a faster-
  moving fair value, or a longer holding horizon each mean a unit of inventory
  is more dangerous, so the same q produces a bigger lean.

And the half-spread: the same `gamma * sigma_b^2 * tte` term charges for the
variance a fill exposes you to, while `(2/k) ln(1 + gamma/k)` is the arrival-
rate markup -- with patient flow (small k, takers still arrive far from mid)
you can afford to quote wide.

A second variant, `"glft"` (Guilbaud-type stationary closed form), is
available for comparison; it produces an inventory-independent half-spread
with the skew folded into per-side deltas. Dalen is the default.

Launch parameters (`MMConfig`): gamma = 0.10, k = 1.0, sigma_b floored at 0.05
and capped at 5.0. These are placeholders to be re-estimated from paper-fill
data -- there is no fill history yet from which to calibrate arrival decay or
risk aversion, so the launch posture is deliberately "quote wide, size small".

### 6.2 What the proposal contains

The output `QuoteProposal` carries r_x, delta_x, the skew, sigma_b, and raw
bid/ask in both spaces, plus a hash of every parameter used (so any quote in
the journal can be traced to the exact settings that produced it). This is not
yet an order: it says where the *core* quote stands before venue realities and
model-quality charges are added.

---

## 7. The spread builder: charging for everything else

`spread_builder.build_quote_set` takes the proposal and composes the final
half-spread **additively in probability units** from seven terms, then makes
the result venue-legal. Additive composition is a feature: every fill's spread can
be decomposed after the fact into named charges (`QuoteSet.terms`), so
calibration can later answer "which charge earns and which just blocks flow?".

| # | Term | What it charges for | Status |
|---|---|---|---|
| 1 | Arrival markup | Taker impatience (the (2/k)ln(1+gamma/k) component) | audit-only |
| 2 | Adverse-selection buffer | Being picked off by informed flow | applied |
| 3 | Inventory skew | Displacement due to q | audit-only |
| 4 | Robust widening | Estimator uncertainty + low pricer credibility | applied |
| 5 | Wing widening | Model tail-quality, outside the belly band | applied |
| 6 | Belly widening | Model belly bias, inside the belly band | applied |
| 7 | Markout-fed widening | Measured pick-off on this side, per side | applied |

Terms 1 and 3 are **audit-only** because they are already embedded in the
proposal's x_bid/x_ask from Section 6 -- the builder reports their price-space
magnitude for decomposition but must not add them again. (Term 1 earned its
comment the hard way: an early version double-counted the arrival markup and
produced ~23-cent spreads at the belly.)

The applied terms:

- **Term 2**: `eps_base` (0.0085, a crypto adverse-selection baseline) plus
  whatever `eps_add` the risk controller ordered this tick (Section 11). This
  is the knob risk escalation turns first: widen before pulling.
- **Term 4**: `sqrt(sigma2)` -- one standard deviation of the probability
  estimate itself (Section 3.2) -- plus `(1 - credibility) * 0.02`: when the
  bankroll consensus trusts the pricer less, quotes widen up to 2 cents.
- **Terms 5/6** partition the probability axis using the **belly band**
  [0.2, 0.8] on the consensus p. Exactly one fires per quote.
    - *Wing* (p outside the band): base 1 cent scaled by the confidence-tier
      multiplier (Section 3.3). Tails are where parameter error lives.
    - *Belly* (p inside): a flat 0.5 cents plus 0.75 cents per day of tte
      beyond 2 free days. This term is empirical: backtest suitability
      analysis measured the pricer's belly bias at about +4.8 cents at 1-2
      days growing to +8.6 cents at 5-7 days, so the belly charge grows with
      horizon to cover roughly the un-shared half of that bias.
- **Term 7** (package E, 2026-07-15): side-asymmetric, fed by measured fill
  markout rather than a constant. For each side independently,
  `spread_builder.markout_widen(mk_avg, scale, cap) = clamp(-mk_avg, 0,
  cap) * scale`: a negative average markout (this side is measurably
  getting picked off) widens that side only, up to a 12-cent cap
  (`markout_widen_cap` 0.05 -> 0.12, 2026-07-26: measured 60s side markouts
  ran -9 to -16c while the 5c cap bound everywhere and the bot bled
  -5c/share over 283 fills); a flat or favorable markout adds nothing --
  there is deliberately no symmetric "tighten on good markout" branch. `mk_avg` comes from the markout report's
  per-side breakdown (Section 14.2) at a *60-second* horizon -- shorter than
  sizing's 600-second `markout_horizon_s` on purpose, since 60s is the
  cleanest pick-off signal and 600s starts folding in ordinary BTC drift
  variance -- and is trusted only once that side has `markout_min_n` (20)
  fills. Because term 7 is piecewise-constant across the ladder (one belly
  value, one wing value, per side), widening a belly bid while a
  neighboring wing bid stays put can invert the non-increasing-bid
  invariant right at the region boundary; the mandatory PAV repair (Section
  9.1) then pools the violating neighbors to their average. This is
  accepted, not a bug: a pool average can never exceed its pre-widening
  value, so every widened quote still moves in the intended direction
  post-repair, it just may not carry the full local widening amount right
  at a boundary. The term is self-resetting: the markout report is a
  rolling 28-day window (7d until 2026-07-26 -- the shorter window let a
  measured-toxic verdict expire weekly and full-size quoting resume on the
  same cell, see Section 14.2's persistence note), so if wider quotes stop
  the toxic fills entirely, those fills age out, the side's `n` drops back
  below `markout_min_n`, the term reverts to zero, and quotes re-tighten --
  no separate decay machinery needed, just a month-scale decay instead of a
  week-scale one.

Then mechanics, in strict order: widen each side (terms 2/4/5/6
symmetrically, term 7 asymmetrically per side -- the shift this can put on
the quote *center* survives the next step, since the floor step recomputes
the center from the already-asymmetric prices) -> floor the half-spread at
one tick -> clamp into the venue price band -> quantize to the tick grid,
*flooring* the bid and *ceiling* the ask so rounding can only ever widen ->
if quantization still crossed the quotes, push the ask up one tick. Finally
the risk directive's mode is applied to sizes: BID_ONLY zeroes the ask,
ASK_ONLY the bid, PULLED both.

---

## 8. Sizing: from edge to shares

Prices decided, the remaining question is size, answered by
`robustness_sizing.size_ladder` through a staged pipeline that runs in two
spaces -- fraction space (per-bankroll fractions, comparable across legs),
then share space (actual order sizes, comparable against depth/inventory/ruin
limits that are naturally share- or notional-denominated). The design maxim:
**never full-Kelly, and record every cap that binds** (each decision carries a
`caps_applied` audit list).

### 8.1 Kelly at our own posted quote, haircut by measured markout

For a binary bought at price P with believed win probability p, define net
odds b = (1 - P)/P. The **Kelly fraction** -- the bankroll fraction maximizing
long-run log wealth if your belief is exactly right --

```
f* = ( b*p - (1 - p) ) / b
```

is computed per contract for both legs: a YES leg (belief p_hat, price = our
own posted bid) and a NO leg (belief 1 - p_hat, price = 1 - our own posted
ask). **P is our own posted quote** (the price the spread builder actually
puts on the book, after all six spread terms), not the market mid -- this is
the literature-correct maker frame (Chen-Pennock utility-maker; AS/GLFT
depth-as-edge): as a maker, the bet you are actually making is "I get filled
at P and the true value is p" -- the edge lives at the quote you post, not at
whatever the rest of the book happens to show. `bid_price`/`ask_price` on
`ContractSizingInput` are computed once by `spread_builder.compute_posted_prices`
and fed both into sizing and into the final `QuoteSet`, so the two can never
disagree.

Believing our own posted edge outright would double-count: the spread already
has an adverse-selection buffer baked in, and unmodeled adverse selection
beyond that buffer needs to be haircut out of the edge before Kelly sees it.
That haircut is **measured markout** where evidence exists, and a **prior AS
charge** otherwise:

```
structural_edge = belief - price                 # e.g. p_hat - our_posted_bid
m_prior         = structural_edge - eps_base      # AS prior charge
m = mk_avg   if (measured: mk_n >= markout_min_n)
    m_prior  otherwise
m = max(m, 0)                                     # Glosten-Milgrom: no edge -> no size
f*, b = kelly_buy(price + m, price)               # belief_eff = price + m
```

`kelly_buy` returns exactly 0 when `belief <= price` (an explicit early-out,
2026-07-15): at `belief == price` -- every m-clamped no-edge leg -- the f
formula's float rounding otherwise leaves a +/-1-ulp residue, and a positive
residue survives every downstream cap (all of which only shrink sizes) to
post as an ~1e-45-share dust order, as observed live on the 2026-07-20
ladder.

`mk_avg` is the per-fill realized markout for this market's (region,
tte_bucket, horizon) cell from `pnl_report.markout_report` -- the actual
value captured net of adverse price movement after each fill, the
Conrad-Wahal "realized spread" -- resolved once per tick by the harness via
`pnl_report.markout_stats` and passed in on `ContractSizingInput.mk_avg` /
`mk_var` / `mk_n` / `mk_n_attempted`. Below `markout_min_n` fills in that
cell the measurement is not trusted yet: `markout_stats` first falls back to
the region-only rollup (`by_region`), and only if that is also thin does
sizing fall back to the prior: structural edge minus the adverse-selection
buffer already charged in the spread. A cold-start ladder (no fills yet
anywhere) runs entirely on the prior path everywhere -- this is what
restores the size-to-spread coupling a 2026-07-12 defect had severed
(sizing on the raw pre-widen proposal, with no adverse-selection haircut at
all, left size unconditionally coupled to `p_hat` alone regardless of how
thin the posted spread actually was).

The `mk_n_attempted` returned by `markout_stats` is ALWAYS the exact cell's
attempted count, even when the measurement itself came from the region
rollup (fix 2026-07-15). The presence-floor exploration gate (next section)
is per-cell by design; before the fix the rollup's n_attempted leaked out,
so the moment a region rollup crossed `markout_min_n` with a negative
mk_avg (wing n=23 at -1.7c, belly n=22 at -2.2c on 2026-07-14), every cell
of that region -- measured or not -- had its gate closed simultaneously:
Kelly zeroed by the m-clamp AND no presence floor, so the fleet stopped
posting entirely, no new fills could arrive, and the negative verdict could
never be re-measured (a self-locking shutdown observed live as ~0 resting
orders across 3 ladders). With the cell-scoped count, a trusted-negative
rollup still zeroes the Kelly leg everywhere in the region, but cells that
have not themselves accumulated `markout_min_n` fills keep their
presence-floor probes flowing, so the measurement keeps updating.

Kelly's other catch is the *if your belief is exactly right*. It is
notoriously aggressive under estimation error -- overbetting is punished far
more than underbetting. The **Baker-McHale shrinkage** discounts f* by how
uncertain the *edge* m is (not by how uncertain the terminal-price model
p_hat is -- see the note at the end of this section):

```
k_shrink = f*^2 / ( f*^2  +  ((b+1)/b)^2 * sigma2_edge )        f <- f* * k_shrink

sigma2_edge = mk_var / mk_n         when measured (mk_n >= markout_min_n)
              config.markout_prior_var   otherwise (uninformed prior, ~two
                                          AS-buffers wide)
```

When sigma2_edge = 0 (perfect measurement) k = 1 and full Kelly stands; as
markout uncertainty grows relative to the edge, k -> 0. A cell with a little
measured data and low variance shrinks less than a cold-start leg sitting on
the uninformed prior -- size grows with evidence, exactly the intended
"exploration builds conviction" dynamic.

**Note on the per-strike/per-ladder MC-SE (parameter-posterior) sigma2.** An
earlier design fed the pricer's own per-strike Monte-Carlo standard error
into this same Baker-McHale shrinkage, on the theory that a wing strike's
larger parameter uncertainty should shrink that leg more. In practice this
double-charged the spread bet: MC-SE measures uncertainty in the *terminal
probability estimate*, a different quantity from *markout* uncertainty (how
much realized adverse selection varies around its mean), and at realistic MC
sample sizes the MC-SE contribution was small enough (sub-cent) that it
mostly just diluted the correctly-scaled markout variance without adding
real information. **As of the posted-spread/markout Kelly rewrite, this
channel is dropped from leg shrinkage entirely** -- `sigma2_edge` above is
the only variance Baker-McHale sees. The parameter-posterior sigma2 the
pricer still computes per strike continues to do real work elsewhere: it
widens the *robust* spread term (Section 5, term 4) and feeds `phi` for
audit -- it no longer shrinks leg sizes. The consequence, accepted rather
than fixed: in the cold-start regime (before enough fills exist to measure
markout per cell) every leg shrinks by the same uninformed prior variance
regardless of strike, so a wing leg's extra parameter uncertainty is not
separately punished at the sizing stage. Wing exposure stays bounded by three
other controls instead -- the q_max shrinking mode, the wing spread-widening
term, and the bucket worst-case cap below -- so this is a loss of one
redundant safeguard, not an open risk.

### 8.2 Presence, reduce-side exemption, and the ladder bets one event

The remaining fraction-space stages are portfolio-level and lightweight:

- **Bankroll utilization cap.** Total fraction across the whole ladder <=
  50% of paper bankroll.
- **Fractional Kelly -- the last FRACTION-SPACE ceiling.** Everything
  surviving is multiplied by c <= 0.5. Half-Kelly costs only a quarter of the
  growth rate but halves the drawdowns and is far more robust to misestimated
  edge; this ceiling is non-negotiable in the config. It is *not* the last
  step overall -- see the bucket/ruin cap below, which is a final
  share-space override that runs after it (a ruin control outranks every
  ceiling).

Fractions become shares via `size = f * bankroll / risk_per_share`, where the
risk per share is P for a YES bought at P, and 1 - P for a NO (both P's are
our own posted quote, as above). Several more stages then run in share space,
in this order:

- **Presence floor, gated on measured net edge.** A quote-side minimum,
  `presence_frac * bankroll / risk_per_share`, tapered toward zero as this
  side's inventory approaches q_max (so the floor never fights the inventory
  cap below it). Pure `max()` against the Kelly size when its gate is open --
  it only ever raises a leg, never lowers one a firmer cap has already set.
  The gate: `(m_gate >= 0) or (mk_n_attempted < markout_min_n)`, where
  `m_gate` is the same measured-or-prior net edge used above (unclamped).
  A cell that has never been measured (or barely measured) keeps the floor
  on regardless of the edge's sign -- the **exploration carve-out**: fills
  are the only source of markout/credibility calibration, so an unmeasured
  cell must keep a minimum resting presence to ever accumulate the evidence
  that would let it earn a real edge-driven size. Only once a cell is
  *trusted* (enough fills) and shows a *measured, negative* net edge does the
  floor turn off on that side -- the failure mode this closes is a
  perpetually-resting floor quote bleeding to a genuinely toxic counterparty
  forever, which the wave-1 unconditional floor could not distinguish from
  ordinary cold-start presence.
- **Unmeasured-cell multiplier (2026-07-26).** The gate above has a gap the
  live bleed exposed: an UNMEASURED cell's `m_gate` is the structural prior
  (posted-edge minus eps_base), which is roughly +half-spread -- positive --
  so both the floor and the Kelly path ran at full size until the cell
  accumulated `markout_min_n` fills, i.e. each cell paid ~20 cap-sized
  losses of tuition before a toxic verdict could act. Now every leg whose
  cell has `mk_n_attempted < markout_min_n` is scaled by
  `unmeasured_size_mult` (0.33; 1.0 disables), EXCEPT the reduce-side leg
  of a positioned market (unwinding must never slow). The scaled size is
  floored back up to `depth_cap_floor_shares` (1 share, a proxy for the
  venue minimum order size) when the pre-scale size was at or above it --
  otherwise the venue-minimum rule would silently no-quote the side, no
  fills would accumulate, and the cell could never become measured (a
  learning deadlock). Fill COUNT -- the learning signal -- accrues at
  nearly the same rate; tuition per fill drops ~3x (less on thin legs
  floored back to 1 share).
- **Reduce-side exemption, ungated.** Cash-EV Kelly has a blind spot: when
  inventory skew exceeds the effective spread, the unload side's Kelly
  fraction (and therefore its floor, if gated) can go to exactly zero --
  Kelly only sees the bet's expected cash payoff, not the risk-relief value
  of shedding a position that is pressing against its cap. Observed live: a
  62-share position quoting bid 81.7c / ask 0.0 -- the exact side that should
  be resting most aggressively was silent. Fix: whichever leg is this
  market's *reduce side* (the ask/NO leg if net long YES, the bid/YES leg if
  net long NO) gets floored at `min(|q|, s_presence)` -- the ordinary
  (untapered) presence-floor unit, capped at the position size itself --
  **unconditionally**, ignoring the W4 gate above. This restores *presence*
  on the unload side, not proportional unwind capacity: an 80-share position
  still unwinds over several fills, not one shot. Proportional unload sizing
  is left to the ordinary Kelly path once that side accumulates its own
  measured (positive) markout. A directive that has already suppressed a
  side (PULLED, or the wrong side of BID_ONLY/ASK_ONLY) still wins -- this
  exemption cannot resurrect a side risk control has deliberately zeroed.
- **Inventory headroom cap.** Bid size <= q_max - q (buying more YES cannot
  push the position past its cap); ask size <= q_max + q (selling YES / buying
  NO cannot push a short position past its cap). Wires the inventory manager's
  live position into sizing itself, closing a prior gap where sizing was
  inventory-blind and only the risk controller's one-sided/pull rules reacted
  to a breach *after* it happened. This q_max is the same S'(x)-based cap of
  Section 10 and is untouched by the package D breach-metric change (Section
  11) -- only the risk controller's own breach *trigger* is now risk-based,
  not this headroom cap.
- **Depth cap, floored at a minimum restorable size.** Each side's size is
  also bounded by `max(realized_depth, depth_cap_floor_shares)` -- the
  liquidity monitor's realized displayed depth (Section 11.2), never allowed
  to collapse all the way to zero. A completely dead book (realized depth 0)
  used to permanently zero our size on that side; the depth cap's job is
  impact control (don't rest more than the book ever shows near the touch),
  not presence control, so it now floors at a venue-minimum restorable size
  (default 1 share) instead. Runs after the presence floor, unmeasured-cell
  multiplier, reduce-side exemption, and inventory cap, so it remains a hard
  minimum over all of them.

Finally, the **bucket worst-case cap** is the joint-ladder control: all
strikes in one expiry settle off the *same* terminal BTC spot, so a YES leg
at strike K and a NO leg at strike K' > K can never both lose (spot cannot be
both <= K and > K'). The strikes partition terminal spot into buckets: for
each bucket, every leg that loses if spot lands there contributes its
share-space risk fraction, and the true worst-case loss is the largest
bucket total. If that exceeds the per-expiry cap (10% of bankroll), every
leg's shares are scaled down by the same factor in one pass. This replaces an
earlier stand-in (scaling the *sum* of all fractions to the single largest
one) that ignored hedged cross-strike structure -- an internally-hedged
YES/NO book is no longer punished as if every leg could lose simultaneously,
while a genuinely concentrated same-direction book is still capped to the
true worst case. A leg scaled down here can end up below its own presence
floor; that is intentional -- caps dominate floors, always, and this is the
final override in the whole pipeline.

---

## 9. The ladder as one object

### 9.1 No-arbitrage across strikes

Since P(S > K) is non-increasing in K, quoted prices must be too. If the MM
ever posted mid(114k) = 0.55 and mid(116k) = 0.58 someone could buy the
cheaper higher strike and sell the dearer lower one for a riskless profit --
and beyond arbitrage, an incoherent ladder means the MM is bleeding through
its own internal inconsistency.

Independent per-strike quoting can produce exactly this, because each strike's
widening and skew are computed locally. So *before any ladder reaches the
order layer*, `ladder_hedger.repair` runs a mandatory check: bid and ask
ladders each non-increasing in strike, and the implied density (adjacent mid
differences) non-negative. Violations are repaired by projecting the mids onto
the closest (least-squares) non-increasing sequence -- the classic **pool
adjacent violators** (PAV) isotonic-regression algorithm: walk down the
ladder, and whenever two adjacent values are out of order, pool them into
their average, cascading backwards. Each contract's half-spread is preserved
around its repaired mid. The projection is idempotent (repairing a repaired
ladder changes nothing), and non-finite mids are replaced by the model's dense
CDF (Section 3.1) before projecting -- the model curve is the monotone
reference shape.

### 9.2 Vertical hedging: naked risk into band risk

Suppose the bot ends up long 80 shares of YES-116k against a cap of 100. That
is naked binary exposure: worth $80 more if BTC settles above 116k, $0 below.
Now buy NO on the *adjacent* strike 114k. The combined position only wins or
loses on settlement *between* 114k and 116k -- a **vertical spread**. Maximum
loss shrinks from the full stake to the band exposure. This is the natural
internal hedge in a ladder: no external instrument needed.

`vertical_hedges` scans post-fill inventory each tick; any strike with
|q| > 50% of its cap emits a `HedgeRecommendation`: take the *opposite* side
in the better-liquidity (else nearest) neighbor, sized to the excess, with a
passive price ceiling (fair value plus one tick -- the hedge should earn
spread too, not cross it).

Recommendations are not orders. On the *next* tick they **skew quote sizes**:
the neighbor's hedge-side quote is inflated by the recommended size, subject
to an exact price rule (a BUY_YES recommendation applies only if the quoted
bid is at or under its `max_price`; BUY_NO symmetrically against 1 - ask), and
never resurrects a side the risk directive suppressed. Applied and skipped
recommendations land in a hedge journal. Pending hedge demand also flows into
the inventory manager's *band exposure* view so risk sees the post-hedge
picture. A more aggressive cross-strike hedge sized by an instantaneous
hedge ratio (beta = ratio of the strikes' S'(x) sensitivities) exists behind
`enable_beta_hedge` but is off by default: beta explodes in the wings unless
carefully shrunk, and the shrinkage has not been validated on live data.

---

## 10. Inventory accounting

`inventory_manager.py` tracks, per contract, the signed position `q`
(YES-positive: +10 means long 10 YES shares; short-YES via holding NO is
negative) and `avg_cost`, through **one fill channel** -- maker fills, taker
fills, and settlement fills all flow through the same `apply_fill`, so at any
time replaying the fills table reproduces inventory exactly
(`fold(fills) == inventory`, an invariant tests assert and the restart
protocol relies on).

Conventions worth knowing:

- **Prices are YES-scale on both sides.** A BUY_NO fill at NO-price 0.30 is
  recorded against the YES-scale book. This single convention prevents an
  entire class of sign bugs between the store, the manager, and PnL.
- **avg_cost** is a volume-weighted average while a position *grows* in its
  current direction; it does not change while the position is being reduced
  (realized PnL is the PnL layer's job, not inventory's); it resets when a
  position opens from flat or flips sign.
- **Position caps shrink at the wings.** Each contract's cap is
  `q_max = q_max_scale * max(S'(x_fair), floor)`. Recall S'(x) = p(1-p): the
  cap is largest at p = 0.5 and shrinks toward the extremes. The reasoning:
  near p = 0.99 a position has little left to earn (at most a cent per
  share) but can still lose everything on a late gap through the strike --
  the worst asymmetry an MM can hold into settlement. This q_max still
  bounds the sizing headroom cap (Section 8.2) unchanged. The risk
  controller's breach *trigger*, however, no longer compares against it
  directly -- see Section 11 for the risk-based metric that replaced the
  |q|/q_max ratio (package D, 2026-07-15).
- `mark(now)` runs every tick so age-weighted holding accrues between fills --
  stale inventory is itself a risk signal.

---

## 11. The risk controller and its inputs

### 11.1 Directives

`risk_controller.RiskController.evaluate` is the final authority on **quote
mode** per market each tick: `TWO_SIDED`, `BID_ONLY`, `ASK_ONLY`, or `PULLED`,
plus an additive widening `eps_add` and a `cancel_all` flag. Triggers, each
independent:

| Trigger | Condition | Response |
|---|---|---|
| Vol gate | BTC realized-vol shock or extreme regime (via `core/strategy/vol_gate.py`) | PULLED; "high" regime widens instead (eps_add += edge_add_cents/100) |
| Near resolution | tte < `near_resolution_pull_hours` (6 h since the 2026-07-11 recal) | PULLED -- endgame binaries are gamma bombs |
| Gap-through | spot within 0.5% of strike while vol elevated | PULLED |
| Inventory breach | remaining-loss notional L / cap >= 1, where L = q * p_consensus (long) or \|q\| * (1 - p_consensus) (short), cap = inv_loss_cap_frac (10%) x ladder bankroll | one-sided *away* from the breach (long -> ASK_ONLY, i.e. only offer to sell down) at ANY ratio -- never PULLED (2026-07-14 stranded-inventory fix, mode unchanged; metric replaced 2026-07-15 by package D, below) |
| Feed dead | WebSocket unhealthy | PULLED + mandatory cancel_all |
| Pricer stale | snapshot age > 300 s / > 600 s | widen / PULLED |
| Fair value stale | consensus age > 300 s / > 600 s | widen / PULLED |
| Liquidity degenerate | book effectively empty | PULLED when flat; reduce-only side (away from the position) when holding inventory (2026-07-14 fix) |
| Mid velocity (rule h, 2026-07-26) | any strike's two-sided mid moved > `mid_move_pull_p` (4c) within `mid_move_window_s` (120 s), ladder-wide max | PULLED when flat; reduce-only side (away from the position) when holding inventory |
| Manual override | operator stop, stale BTC data, unresolved resume discrepancy | PULLED |

Multiple triggers combine to the **most restrictive** mode; two *opposite*
one-sided requirements escalate to PULLED (you cannot safely satisfy both);
eps_add contributions sum. Ending PULLED always implies cancel_all -- pulled
means no resting orders, not merely no new ones.

Why the inventory-breach and degenerate-book rules stopped pulling
(2026-07-14): a one-sided mode AWAY from the position never adds risk -- the
add side is suppressed and the reduce leg is bounded by sizing (reduce-side
floor min(|q|, s_presence); inventory headroom bid <= q_max - q / ask <=
q_max + q always leaves the reduce side room). Escalating to PULLED therefore
protected nothing and only removed the unwind path: q_max = q_max_scale *
S'(x) shrinks under a FIXED position as p drifts toward a wing, so a filled
strike that went deep ITM/OTM would cross the 1.5x "extreme" threshold and
strand its inventory until settlement (observed live: 58000 short on all
three ladders, PULLED for hours). Posting the reduce side as sole maker in a
dead (degenerate) book is safe for the same reason: we set the price, the
conservative queue-behind fill sim fills only against real prints, and the
DEPTH cap floor (1 share) means the position drains gradually. Feed-dead,
near-resolution, staleness, vol-shock and manual pulls are deliberately
unchanged -- those are states where no quote can be trusted, and inventory
parked by them is surfaced by the heartbeat's stranded metric instead.

The mid-velocity rule (h) exists because the vol gate cannot see a live
burst: it reads `DATA/btc_intraday_1m.csv`, refreshed only every 30 minutes
by the datafetch timer, while the 2026-07-26 fill autopsy showed ALL of the
bleed landing inside multi-minute BTC bursts -- resting quotes were swept
~10c through the mid before any defense reacted (15 s tick, 300 s reprice,
1-tick requote deadband). The ladder's own mids ARE visible live, per tick,
in the harness: it keeps a small per-market history of two-sided mids
(single-sided fallback mids are excluded -- a book flickering between one-
and two-sided would fake large moves, and the max-over-markets aggregation
would let one noisy strike pull the whole ladder), computes the largest
absolute move within the window across the ladder, and threads it into
`evaluate`. One strike's real move protects every strike -- informed flow
hits the moving strike first and its neighbors next. Detection is one tick
late by construction, so the FIRST fill of a burst still lands; the 60 s
hysteresis latch then holds the pull through the trend continuation, which
is where the repeated cap-sized losses actually occurred. Like rules (c)
and (f), a positioned market quotes reduce-only instead of pulling (same
signed-q basis, so co-fires agree), preserving the unwind path exactly when
unwinding matters most.

Package D (2026-07-15) redefined the breach *metric* itself, leaving the
mode logic above untouched. The old |q|/q_max ratio inherited q_max's
wing-shrinking shape (Section 10), so a large wing position could show an
alarming ratio while carrying almost no real risk: 70k jul-20 at q=14.3
against q_max=4.57 read as a 3.1x breach while its remaining-loss notional
was about $0.72. The controller now compares each position's dollar
mark-to-worst-outcome loss -- `q * p_consensus` if long YES (marks to 0 on a
NO settlement) or `|q| * (1 - p_consensus)` if short (marks to 1 on a YES
settlement) -- against `config.inv_loss_cap_frac * bankroll` (10% of the
loop's sizing bankroll by default; a `cap <= 0` bankroll/frac skips breach
detection entirely rather than dividing by it). `is_long` still comes from
the raw signed q, with no hedge adjustment in this phase: rule (c) here and
rule (f) (the liquidity-degenerate reduce-only rule above) must keep
deriving their one-sided direction from the same signed quantity, or the
most-restrictive combination would escalate two opposite one-sided modes to
PULLED and reopen the 2026-07-14 stranding bug -- a hedge-aware version is
deferred to a follow-up. Sizing's own inventory headroom cap (Section 8.2)
is untouched by this change and still uses the old S'(x)-based q_max, a
separate and deliberately conservative risk decision; only the risk
controller's breach *trigger* is now risk-based.

Note the graduated pattern that recurs: **widen first, pull at 2x**. Widening
keeps earning (at a higher charge) through mild degradation; pulling forfeits
revenue but is the only move once quotes cannot be trusted at any width.

**Hysteresis.** Any transition *into* a restrictive mode latches for 60 s;
returning to normal requires both the latch expired and the trigger cleared.
Without this a signal sitting on its threshold would flap the book with
cancel/replace churn every tick. Escalations re-arm the latch; all transitions
are journaled.

One deliberate non-action: the vol gate's `kelly_mult` is carried on every
directive but **not applied to sizing** -- it is journaled only. Sizing is
already defended in depth (Baker-McHale shrinkage, inventory headroom, depth
cap, the bucket worst-case/ruin cap, fractional-c), and the vol gate acts on
quotes via eps_add and PULL instead. Recording it keeps the option to
activate later with data.

### 11.2 The liquidity monitor

`liquidity_monitor.py` distills the book stream into per-market gauges:

- **Realized depth** near the touch (within 3 ticks), per side -- feeds the
  sizing depth cap and the regime tag.
- **Impact lambda** (a Kyle's-lambda-style magnitude): regress |mid change| on
  unsigned trade size through the origin. Deliberately *unsigned*: public
  Polymarket feed data identifies aggressor direction too unreliably (~59%)
  to build signed flow-toxicity metrics on, so none are built.
- **Arb half-life**: |YES_mid + NO_mid - 1| should mean-revert to zero in a
  healthy market; fitting an AR(1) coefficient to the deviation series and
  converting to a half-life measures how fast arbitrageurs are policing the
  market. Slow decay = thin, inefficient market.
- A **regime tag** (THICK / NORMAL / THIN / DEGENERATE) from combined depth;
  DEGENERATE is a PULL trigger above.

### 11.3 Order lifecycle

`order_lifecycle.py` turns the final QuoteSet + directive into venue actions
with minimal churn: an existing order within the re-quote tolerances (price
within 0.015, size within 10%) is left alone -- every needless cancel/replace
surrenders queue position, which under the fill simulator's queue-behind rule
(next section) is the MM's most valuable asset. The price tolerance is a
deliberate 1-tick deadband (raised from 0.005, 2026-07-16): quantized prices
only ever move in whole 0.01 ticks, so any sub-tick tolerance is dead for
price -- a raw price hovering on an exact tick boundary plus sub-tick
consensus jitter through `_quantize`'s outward floor/ceil produced a 1-tick
square wave (observed live: ask 0.80<->0.81 at 15s cadence with every spread
term frozen), cancelling and reposting the order each tick. With the deadband
a 1-tick flap holds the resting order and its queue position; a >=2-tick move
still reposts, bounding the resting order's lag behind the desired price at
one tick. Trade-off accepted: a resting order may transiently sit 1 tick
tighter than the desired price, and cross-strike no-arb is only guaranteed
between DESIRED ladders (PAV repair) -- two resting orders each lagging 1
tick toward each other could transiently violate it by up to 2 ticks. A side whose size is below
the venue's minimum order size (`VenueDescriptor.min_size`, threaded in by
the harness as `min_order_size`, 2026-07-15) is treated as no-quote -- a live
venue would reject it, so the paper path does not post it either, and a
resting order on that side is cancelled. On restart, all persisted orders
are marked UNKNOWN and reconciled against the venue before quoting resumes.

---

## 12. Paper fills: the load-bearing pessimist

Since no real orders are sent, the entire experiment's validity rests on the
fill simulator (`paper_fill_sim.py`) not being optimistic. Its rules, applied
to the real live book and trade-print stream:

1. **Queue-behind placement.** Joining an existing price level puts our order
   behind *all* displayed size at that level. Posting a new better level gets
   queue_ahead = 0 -- but still earns nothing by itself.
2. **Prints only.** A fill requires an observed aggressor trade at or through
   our price. The print's size first consumes the queue ahead, and only the
   remainder fills us. Queue-ahead can also shrink when level size drops with
   no print (a cancel ahead of us) -- but if a print is present in the same
   update the ambiguity is resolved against us.
3. **No price improvement.** Fills happen at exactly our quoted price.
4. **Latency.** Placements take effect 2 s after the decision; cancels 2 s
   after theirs -- and during that window the dying quote is still live and
   can be hit. We own our stale quotes.
5. **Feed gaps.** While a market's feed is unhealthy, no fills are simulated
   and any live quotes are recorded as *exposure incidents* -- windows where a
   real MM would have been quoting blind.
6. **No self-impact.** Our simulated size never appears in the book (the one
   optimism, counterweighted by rules 1 and 3 and noted in the risk
   register).
7. Every fill is stamped with an **assumption-set version**, so metrics can
   never silently mix fill models. A strictly harsher fallback mode fills
   only on prints strictly *through* our level.

The simulator is deterministic: identical input streams produce identical
fills.

---

## 13. Settlement

At 12:00 ET on the expiry date, each market resolves YES if and only if spot
is **strictly above** the strike (the venue-confirmed rule; equality resolves
NO). `settlement_handler.py` computes the outcome from BTC data and emits a
**synthetic closing fill** at $1 or $0 through the normal fill channel -- no
special-case code path, so `fold(fills) == inventory` survives resolution and
every downstream consumer (store, PnL, restart replay) handles settlement for
free.

Practical wrinkles: the risk controller already pulled quotes 24 h before this
moment (Section 11), and after a restart the settlement pass runs in
*catch-up* mode against the persisted market registry, so positions from a
previous run's ladder -- even a previous event's -- get settled before quoting
resumes (Section 15.3).

One timing subtlety: the first settle attempt fires seconds after the
settlement instant, but the 30-minute data-fetch timer has almost never yet
written the BTC bar covering that instant, so the first attempts are
UNSETTLEABLE by design and succeed on retry. The handler's `BTCDataProvider`
therefore re-checks the CSV's mtime once per settlement resolution
(`refresh()`) and reloads when the file has changed -- a load-once cache here
would freeze the first stale view and defeat the entire 24 h retry window
(observed in production 2026-07-11).

---

## 14. Knowing whether it works

### 14.1 PnL

`pnl_report.py` snapshots PnL every tick, folded from the durable fills table
(not from in-memory state): realized = cash plus q x avg_cost, marked against
current mids, settlement-aware.

### 14.2 Markout: the MM's quality metric

Raw PnL is noisy and slow. The sharper diagnostic is **markout**: for each
fill, compare the fill price to the market mid at fixed horizons *after* the
fill (60 s, 10 min, 1 h; disjoint windows). If we bought at 0.44 and the mid
is 0.47 ten minutes later, the +3 markout says the quote was well-placed; a
systematically *negative* markout is the numeric signature of adverse
selection -- takers were right and we were the wrong side. Spread revenue with
negative markout at all horizons is an MM slowly bleeding to informed flow.

The report cross-tabulates markout by book region (belly / wing -- directly
testing whether the pricer's known belly bias leaks through the consensus
anchor into realized fill quality) and time-to-expiry bucket (0-1 d / 1-2 d /
2-4 d / 4 d+), over a rolling 28-day window (`MARKOUT_LOOKBACK_S`, 7d ->
28d 2026-07-26). Since 2026-07-26 each fill's markout is computed ONCE and
persisted to the state store's `fill_markouts` table
(`(fill_id, horizon_s) -> mk`, INSERT OR IGNORE, pruned past the lookback);
the per-tick `mid_log` itself is still pruned to 7 days
(`MID_LOG_RETENTION_S` -- disk cost unchanged), and fills older than that
resolve from the persisted table. Before this split, lookback and mid
retention were the same 7-day constant, so a measured-toxic cell's verdict
EXPIRED weekly: `mk_n` dropped below `markout_min_n`, sizing reverted to
the optimistic structural prior, and full-size quoting resumed on a cell
already known to be toxic (the observed weekly re-arm bleed cycle).
Each cell reports both successful lookups `n` and `n_attempted`, so "no
fills" is distinguishable from "mids missing" -- with one semantics rule
(2026-07-26): a fill/horizon that resolved (live or persisted) counts in
both; an unresolved fill still young enough that its mids can exist
(within `MID_LOG_RETENTION_S`) counts in `n_attempted` only; an unresolved
fill older than that is permanently unresolvable and counts in NEITHER --
otherwise phantom attempts would mark never-measured cells as "attempted
enough", switching off both the sizing exploration gate and the
unmeasured-cell multiplier on cells with no actual measurements. This table is the primary
input to the post-acceptance spread recalibration: per-cell, spreads should
cover realized adverse selection, and cells that never fill are charging too
much.

Package E (2026-07-15) added a per-side breakdown for quoting to consume:
every cell, and every `by_region` rollup, gains an additive `"sides"` key --
`{"BUY_YES": {n, n_attempted, mk_avg, mk_var}, "BUY_NO": {...}}` -- populated
in exact lockstep with the aggregate figures (BUY_YES fills are our bid
side, BUY_NO our ask side; fills are recorded YES-scale on both sides).
`pnl_report.markout_stats_side` resolves one side's numbers with the same
fallback order as the aggregate `markout_stats` -- exact cell, then region
rollup, then `(None, 0)` if both are thin -- and degrades to `(None, 0)`
rather than raising on a malformed report. This is what feeds spread term 7
(Section 7): the harness looks up each side's `mk_avg` at a 60-second
horizon and turns a measurably negative one into extra width on exactly
that side.

The report is written to `markout_report.json` on a fixed cadence and rendered
read-only in the monitoring dashboard (`app/pages/mm_monitor.py`).

### 14.3 Maker-rebate accounting (display-only)

Polymarket pays makers 20% of the crypto category's taker-fee pool daily in
pUSD, pro-rata by filled-volume fee-equivalent. Each markout cell (and the
`by_region`/`by_expiry` rollups) carries an additive `rebate_avg` -- the mean
per-share estimated rebate (`0.20 * 0.07 * price*(1-price) * size`, MAKER
fills only, `market_maker/config.py` constants) over the same fills that
contributed `mk_avg`, so `mk_avg + rebate_avg` reads as net-of-rebate fill
quality. This is a display-only accounting estimate (an off-equity "Rebates
accrued (est)" metric on the dashboard PnL panel and a line on the Telegram
bot's `/bankroll` command) -- it never feeds `markout_stats`, sizing, or
equity/realized/bankroll. Folding rebates into the quoting layer (sizing net
edge, spread floor) is deliberately not implemented.

---

## 15. Operations

### 15.1 Stages

- **Stage A** (`shadow_runner.py`): read-only REST polling, no fills by
  construction. Used to validate quoting math against live books (it caught
  the double-counted arrival markup and the sigma_b inflation).
- **Stage B** (`paper_runner.py`): the same harness fed from the live CLOB
  WebSocket, so real trade prints reach the fill simulator. This is what runs
  unattended on the VPS.

### 15.1b Multi-expiry orchestration

Stage B can quote up to `--max-expiries` concurrent expiry ladders
(`multi_runner.MultiExpiryOrchestrator`). The single-expiry
`PaperTradingLoop` is unchanged; the orchestrator owns one "ladder slot" per
expiry -- each with its own loop, its own WS connection and its own sim
clock -- over one shared state store, one shared vol gate, one shared BTC
data provider and one shared pricing engine. The engine shares a single
GARCH fit (and one set of calibrated jump params) across per-expiry ladder
caches, and hands out ONE reprice token per tick: the first due expiry (in
round-robin rotation) recomputes, the rest serve their cached ladder, and a
brand-new expiry's slot is skipped entirely until its first-price grant
lands -- so a tick never blocks for more than one engine call, and K fresh
ladders warm up over K ticks. The sizing bankroll is statically split
(`bankroll / max_expiries` per ladder); the Beuoy credibility bankroll was
already per-expiry.

Rollover is in-process: a fully-settled (or settlement-timed-out) ladder is
torn down in place -- final settle attempt, scoped per-market order
cancels, ladder-state flush, adapter stop -- and acquisition immediately
probes for the next event (`resolve_events_multi`: capped, expiry-deduped,
skip-on-thin-ladder, empty-list-on-nothing). The process itself only exits
when there is nothing left to quote at all.

### 15.2 The control loop from outside

A stdlib-only control-file protocol (`run_control.py`) governs the process:
a PID file, a touch-to-stop file, a start lock, and `current_run.json`
pointing at the latest run and its exit reason (plus an `events` list of
every active ladder; the legacy singular fields point at the nearest
expiry). Each run rewrites `heartbeat.json` every tick with tick counters,
feed health (AND over active ladders), BTC-data age, resume discrepancies,
the bankroll-frozen flag (OR over ladders), the stranded-inventory gauge
(`stranded_markets` / `stranded_shares`, current-tick counts of markets
holding inventory whose quotes are PULLED outside the near-resolution
window -- "not currently unwindable", monitor-only, no alert; a transient
latched pull can over-count for ~60 s), and per-expiry breakdowns
(`n_expiries_active`, `ladders_settled_total`, `ladder_settlement_timeouts`,
`expiries`, each expiry carrying a `stranded` count); `engine_status()`
derives RUNNING / STARTING / STALLED / STOPPED / CRASHED from these files. Exit codes are contracts with systemd:
42 means "nothing quotable" in auto mode (`no_quotable_events` -- rollover
itself is in-process now) or the legacy settled/timeout signal in fixed-slug
mode, and triggers a retry restart; 1 means a supervised restart (dead feed,
repeated tick errors); 0 means a clean stop.

Two guards run every tick: the **BTC staleness guard** stats the intraday CSV
(refreshed by a separate systemd timer, never by the runner) and flips
`manual_override=True` -- pull everything -- if it exceeds 2 h; and the alert
timer (`scripts/mm_alert_check.py`, every 5 min) pages a webhook on CRASHED /
STALLED / unhealthy-feed streak / stale data / low disk / resume discrepancies
/ frozen bankroll / in-process ladder settlement timeouts / sustained
zero-active-expiries, de-duplicated, plus one daily heartbeat message so
webhook silence is distinguishable from a dead alert pipeline.

### 15.3 Surviving restarts

With `--state-db`, the SQLite state store (orders, fills, inventory, markets
registry, mid log; WAL mode) persists across restarts. A resumed run executes
a strict sequence: ONE standalone catch-up settlement pass over the merged
market registry (a previous event's still-open positions settle -- via
SETTLEMENT pseudo-fills written through the fills table -- BEFORE any
replay) -> each ladder's loop rebuilds its inventory by replaying ONLY its
own markets' fills (`resume_attach`) -> ONE venue reconcile (all fill sims
are empty at process start, so stale orders cancel and the position check is
the global fold vs the global store inventory). Any position discrepancy
found holds `manual_override=True` (quotes pulled) until the first clean
tick, and is surfaced in the heartbeat and the alert path. Mid-run-acquired
ladders never run the reconcile (it is store-global); a recurring throttled
catch-up pass re-drives orphaned UNSETTLEABLE positions as fresh BTC data
lands. The design principle: **the fills table is the source of truth**;
everything in memory must be reconstructible from it, per ladder.

---

## 16. How the pricing engine's outputs are used: a checklist

Collecting the threads, the GARCH/MC engine touches the market maker at
exactly these points:

1. **`p_hat` per strike** -> one of two experts in the fair-value consensus
   (Section 4); the consensus x_fair centers every quote (Section 6).
2. **`p_grid`** (densified curve) -> the monotone reference for PAV no-arb
   repair (Section 9.1).
3. **`n_sims`** -> the exact Bernoulli standard error `p(1-p)/n_sims`
   (Section 3.2), which feeds the robust spread term (Section 7, term 4) and
   `phi` (audit). It no longer drives Baker-McHale size shrinkage directly
   (Section 8.1) -- that channel now shrinks on measured/prior *markout*
   variance instead; see Section 8.1's note on the parameter-posterior sigma2.
4. **Bayesian posterior bands** (companion module) -> wing-strike parameter
   uncertainty, replacing the misleadingly small MC error in the tails
   (Section 3.2); widens the robust spread term, no longer shrinks leg sizes.
5. **Known model weaknesses from backtests** -> hard-coded structure: the
   belly-widening term's base/slope (Section 7, term 6) and the confidence-
   tier day boundaries gating wing widening (Section 3.3).
6. **Snapshot age** -> staleness triggers: widen, then pull (Section 11).
7. **The BTC vol gate** (same underlying data) -> shock/regime PULL and
   widening triggers (Section 11).

The pattern throughout: the model is *one input*, wrapped in credibility
weighting, uncertainty pricing, and staleness policing -- never trusted
blindly, and its documented weak spots are charged for explicitly in the
spread.

---

## 17. Glossary

| Term | Meaning |
|---|---|
| Adverse selection | Losses from fills against better-informed takers |
| AS model | Avellaneda-Stoikov optimal market-making framework |
| Belly / wing | Probability region 0.2-0.8 / outside it |
| Belief vol (sigma_b) | Volatility of fair value in log-odds, per sqrt-day |
| Fair value | The price the MM believes is correct; here, a model-market consensus |
| Half-spread (delta) | Distance from quote center to each of bid and ask |
| Inventory (q) | Signed position, YES-positive |
| Kelly fraction | Log-wealth-optimal bet fraction given edge and odds |
| Ladder | All strikes sharing one expiry |
| Log-odds / logit (x) | ln(p/(1-p)); quoting coordinate |
| Maker / taker | Rests orders and earns spread / crosses spread for immediacy |
| Markout | Mid-price move at a fixed horizon after a fill; adverse-selection gauge |
| Mid | Midpoint of best bid and best ask |
| PAV | Pool-adjacent-violators isotonic regression; the no-arb repair |
| Reservation price (r_x) | Inventory-adjusted indifference price |
| Tick | Minimum price increment (one cent); also one loop iteration |
| tte | Time to expiry |
| Vertical spread | Opposite positions in adjacent strikes; band-limited risk |
