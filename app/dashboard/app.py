# streamlit: disable magic
"""ALPHAFORGE dashboard (Streamlit + Plotly). Reads only stored/broker-sourced data; never fabricates values.
Run:  python -m app.main dashboard      (or: streamlit run app/dashboard/app.py)"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st  # noqa: E402

from app.config import load_settings  # noqa: E402
from app.dashboard import charts, data  # noqa: E402
from app.storage.db import Storage  # noqa: E402

st.set_page_config(page_title="ALPHAFORGE", layout="wide")


def show(state: data.ViewState) -> bool:
    if state.status == "empty":
        st.info(f"No data yet. {state.reason}")
        return False
    if state.status == "unavailable":
        st.error(f"Unavailable: {state.reason}")
        return False
    return True


def main() -> None:
    s = load_settings()
    try:
        db = Storage(s.app.db_path)
    except Exception as e:
        st.error(f"Unavailable: database could not be opened ({type(e).__name__}: {e})")
        return
    b = data.banner(db, s)
    st.title("ALPHAFORGE - paper-trading control room")
    if b.get("origin") == "demo":
        st.warning("DEMO DATA - synthetic, isolated database. Not market data, not an account.")
    st.caption(f"Paper order execution: {'ENABLED' if b['paper_orders_enabled'] else 'disabled'} | "
               f"Shorting: {'enabled' if b['shorting_enabled'] else 'disabled'} | "
               f"Operator halt: {'ACTIVE' if b['halted'] else 'off'} | DB: {s.app.db_path}")
    if st.sidebar.button("Refresh"):
        st.rerun()
    tabs = st.tabs(["Portfolio", "Strategy monitor", "Agent activity", "Sentiment Intelligence", "Operations", "Config"])

    with tabs[0]:
        v = data.portfolio_view(db, s.app.starting_equity_target)
        if show(v):
            d = v.data
            e, ex, tr = d["summary"]["equity"], d["summary"]["exposure"], d["summary"]["trades"]
            c = st.columns(5)
            c[0].metric("Equity", f"${e['ending_equity']:,.2f}")
            c[1].metric("Total return", f"{e['total_return_pct']:.2f}%" if e.get("total_return_pct") is not None else "n/a")
            c[2].metric("Daily P&L (broker)", f"${e['daily_pnl']:,.2f}" if e.get("daily_pnl") is not None else "n/a")
            c[3].metric("Buying power", f"${e['buying_power']:,.0f}")
            c[4].metric("Max drawdown", f"{e['max_drawdown_pct']:.2f}%")
            st.plotly_chart(charts.equity_chart(d["equity_curve"]), width="stretch")
            c2 = st.columns(4)
            c2[0].metric("Gross exposure", f"{ex.get('gross_pct', 0):.1f}%")
            c2[1].metric("Unrealized P&L (broker)", f"${ex.get('unrealized_pnl', 0):,.2f}")
            c2[2].metric("Realized P&L (derived from fills)", f"${tr['realized_pnl']:,.2f}" if tr.get("available") else "No closed trades")
            c2[3].metric("Win rate (derived)", f"{tr['win_rate']:.0%}" if tr.get("available") else "n/a")
            st.caption(d["summary"]["provenance"])
            st.subheader("Positions")
            if len(d["positions"]):
                st.dataframe(d["positions"], width="stretch")
            else:
                st.info("No open positions.")
            st.subheader("Orders")
            if len(d["orders"]):
                st.dataframe(d["orders"], width="stretch")
            else:
                st.info("No orders yet.")
            if tr.get("available"):
                st.subheader("Strategy attribution (derived)"); st.json(tr["attribution"])

    with tabs[1]:
        v = data.strategy_view(db)
        if show(v):
            d = v.data
            st.write(f"**Regime:** {d['regime']} - {'; '.join(d['regime_reasons'])}")
            if d["blocked"]:
                st.info(f"Scan outcome: {d['blocked']}")
            st.subheader("Signals"); st.dataframe(d["signals"]) if len(d["signals"]) else st.info("No signals (no-trade is valid).")
            st.subheader("Candidates");
            if len(d["candidates"]):
                st.dataframe(d["candidates"])
                fig = charts.score_components(d["candidates"].head(10))
                if fig: st.plotly_chart(fig, width="stretch")
            else:
                st.info("No candidates.")
            st.subheader("No-trade reasons"); st.dataframe(d["no_trade"]) if len(d["no_trade"]) else st.info("None recorded.")
            st.caption("Baseline comparison: run `python -m app.main replay` (replay prints benchmark buy-and-hold). No live baseline is fabricated.")

    with tabs[2]:
        v = data.agent_view(db)
        if show(v):
            st.subheader("Agent runs (latency / errors / fallback use)"); st.dataframe(v.data["runs"])
            st.subheader("Risk-gate decisions (deterministic authority)"); st.dataframe(v.data["decisions"])

    with tabs[3]:
        v = data.sentiment_view(db)
        if show(v):
            d = v.data
            c = st.columns(2)
            c[0].plotly_chart(charts.sentiment_distribution(d["distribution"]), width="stretch")
            c[1].plotly_chart(charts.sentiment_trend(d["feed"]), width="stretch")
            st.write(f"**Scoring models in use:** {d['models']}  | **Symbols with conflicting news:** {d['conflicting_symbols'] or 'none'}")
            st.subheader("High-impact events"); st.dataframe(d["high_impact"][["symbol", "headline", "label", "score", "materiality", "novelty", "decayed_score", "published_at", "url"]]) if len(d["high_impact"]) else st.info("None.")
            st.subheader("News feed")
            st.dataframe(d["feed"][["published_at", "symbol", "source", "headline", "label", "score", "novelty", "materiality", "model", "url"]])
            st.subheader("Source health"); st.dataframe(d["provider_health"]) if len(d["provider_health"]) else st.info("No provider calls recorded.")

    with tabs[4]:
        v = data.ops_view(db)
        if show(v):
            d = v.data
            rec = d["reconcile"]
            st.write(f"**Broker reconciliation:** {'OK' if rec and rec.get('ok') else 'NOT reconciled / unknown'} {rec or ''}")
            st.write(f"**Schema version:** {d['schema_version']} | **Open orders:** {d['open_orders']} | **Halt:** {d['halt']}")
            st.subheader("Alerts"); st.dataframe(d["alerts"]) if len(d["alerts"]) else st.info("No alerts.")
            st.subheader("Provider health"); st.dataframe(d["provider_health"]) if len(d["provider_health"]) else st.info("No provider calls recorded.")

    with tabs[5]:
        st.caption("Credentials are never shown; only presence flags.")
        st.json(data.config_view(s).data)


main()
