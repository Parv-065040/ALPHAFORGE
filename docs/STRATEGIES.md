# Strategies (hypotheses to be validated by your own replay/paper results)

All signals use **completed daily bars** plus a fresh quote for the entry reference. Stops/targets are ATR-based.
1. **Momentum breakout (swing, enabled)** – close > SMA50 > SMA100, price above SMA20, price above the prior 20-bar
   high, volume ≥ 1.3× 20-day average, ATR% within [1%, 6%], not extended > 0.5 ATR past the breakout (no chasing).
   Stop 2 ATR, target 3R.
2. **News catalyst (intraday, enabled)** – needs fresh, non-conflicting, material (≥0.5), novel (≥0.5) positive
   sentiment (aggregate ≥ 0.35) **and** price ≥ +1% above last close **and** volume ratio ≥ 1.5. News alone never
   qualifies. Stop 1.5 ATR, target 2R. Flattened before the close. *Limitation:* volume confirmation uses the last
   completed daily bar's ratio, not intraday relative volume.
3. **Relative strength / breakdown (disabled)** – long RS continuation near 50-bar highs; short breakdown
   continuation only if `allow_short_signals` **and** `ALLOW_SHORTING=true`.
**Scoring** (weights in config; renormalised when a component has no data): trend/momentum 0.30, relative strength
0.20 (cross-sectional percentile), volume/breakout 0.20, catalyst 0.20, regime 0.10. Minimum score 0.60. Sentiment is
used once (catalyst component); price/volume once (volume_breakout) to limit double counting.
**Regime** is deterministic from SPY (above SMA50, SMA20>SMA50, SMA50 rising, ATR%). `risk_off` blocks longs.
Missing benchmark data blocks all entries.
**Replay** (`replay`) runs the same strategy code bar by bar: signal on completed day *i*, fill at day *i+1* open
with slippage, stop wins if stop and target are both touched, gaps fill at the open. Catalyst is not replayable
(no historical news store). Compare against the printed buy-and-hold benchmark; synthetic demo results mean nothing.
