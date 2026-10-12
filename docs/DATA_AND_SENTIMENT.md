# Data and Sentiment Intelligence

**Sources** (one vendor, replaceable via `MarketDataProvider` / `NewsProvider` protocols): Alpaca market data (free
IEX feed by default - volume is a *subset* of consolidated volume; use `feed: sip` only with a paid plan) and Alpaca
news (Benzinga). Source health (latency, errors) is logged per call and shown in the dashboard.
**Pipeline:** clean HTML → map symbols to the universe → fold exact/near-duplicate headlines (hash + Jaccard ≥ 0.8)
→ score (FinBERT or labelled rule-based fallback) → materiality (event keywords, single-ticker focus) → novelty
(`1/(1+duplicates)`, including previously stored hashes) → exponential time decay (6 h half-life) → per-symbol
aggregate `tanh(Σ impacts)` with conflicting/stale flags → price/volume confirmation (`confirmed | divergent |
unconfirmed | no_sentiment`).
**Meaning of the score:** `p_positive − p_negative` weighted by confidence/novelty/materiality/decay. It is a text
polarity signal, **not** a calibrated probability of a price move. Evidence (headline, url, source, timestamp, model)
is stored with every row. News is deduplicated across cycles so repeated polls never double-count a story.
**Data quality:** bars with non-positive prices, high<low, negative volume or duplicates are dropped; symbols with
stale bars (> 3 sessions) or stale/missing quotes are excluded with a recorded reason; a >25% jump between quote and
last close is treated as implausible. Corporate-action handling relies on Alpaca split-adjusted bars.
