"""ALPHAFORGE command line.  python -m app.main <command>  (see README)."""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timedelta

from app.config import Settings, load_settings
from app.logging_setup import setup_logging
from app.preflight import format_report, paper_mode_confirmed, run_preflight
from app.timeutil import session_status, utcnow


def _build(s: Settings, need_broker: bool):
    from app.agents.llm import OllamaClient
    from app.data.providers import AlpacaMarketData, AlpacaNews
    from app.execution.broker import AlpacaPaperBroker
    from app.execution.service import ExecutionService
    from app.flow.pipeline import TradingFlow
    from app.sentiment.pipeline import get_scorer
    from app.storage.db import Storage
    db = Storage(s.app.db_path)
    scorer, warn = get_scorer(s.use_finbert, s.finbert_model)
    if warn:
        print(f"note: {warn}")
    llm = OllamaClient(s.ollama_host, s.ollama_model, s.llm.timeout_seconds, s.llm.temperature) if s.use_llm else None
    broker = AlpacaPaperBroker(s) if need_broker else None
    ex = ExecutionService(s, broker, db) if broker else None
    flow = TradingFlow(s, db, AlpacaMarketData(s), AlpacaNews(s), ex, scorer, llm)
    return db, flow, ex


def _print_cycle(st) -> None:
    sc = st.scan
    if sc:
        print(f"regime={sc.regime.regime} candidates={len(sc.candidates)} signals={len(sc.signals)}")
        for g in sc.signals[:10]:
            print(f"  SIGNAL {g.symbol:6s} {g.strategy:18s} {g.direction.value} score={g.score:.2f} "
                  f"entry={g.entry_price} stop={g.stop_price} target={g.target_price}")
    for p, d in zip(st.proposals, st.decisions):
        print(f"  PROPOSAL {p.symbol} qty={p.qty} -> risk {'APPROVED' if d.approved else 'REJECTED'} "
              f"({'; '.join(d.reasons)[:160]})")
    for pid, r in st.executions:
        print(f"  EXECUTION {pid}: {r.status} {r.detail}")
    if st.blocked_reason:
        print(f"no trade: {st.blocked_reason}")
    for w in st.warnings:
        print(f"warning: {w}")


def cmd_preflight(s: Settings, a) -> int:
    checks = run_preflight(s, online=not a.offline)
    print(format_report(checks))
    return 1 if any(c.status == "FAIL" for c in checks) else 0


def cmd_scan(s: Settings, a) -> int:
    if not s.has_alpaca_keys():
        print("Alpaca keys missing; cannot fetch data. Use `replay --demo` or `demo-seed` for offline demos.")
        return 2
    db, flow, _ = _build(s, need_broker=s.alpaca_paper and s.has_alpaca_keys())
    if a.crewai:
        from app.flow.crewai_flow import build_crewai_flow
        st = build_crewai_flow(flow, dry_run=True).kickoff()
    else:
        st = flow.run_cycle(dry_run=True)       # scan NEVER submits orders
    _print_cycle(st)
    return 0


def cmd_replay(s: Settings, a) -> int:
    from app.replay.engine import demo_frames, load_csv_dir, run_replay
    if a.demo:
        frames, origin = demo_frames(s.universe.symbols), "DEMO_SYNTHETIC (not market data; results are meaningless for performance)"
    else:
        frames, origin = load_csv_dir(a.data_dir), f"csv:{a.data_dir}"
        if not frames:
            print(f"No CSV files in {a.data_dir}. Expected SYMBOL.csv with date,open,high,low,close,volume.")
            return 2
    r = run_replay(s, frames, origin, use_rs=a.rs)
    print(f"origin: {r.origin}")
    print(json.dumps(r.metrics, indent=2))
    print(f"benchmark buy&hold over same window: {r.benchmark_return_pct:.2f}%")
    return 0


def cmd_dashboard(s: Settings, a) -> int:
    import os
    env = dict(os.environ)
    if a.demo:
        from app.demo import DEMO_DB, seed_demo
        seed_demo(s, DEMO_DB).close()
        env["ALPHAFORGE_DB_PATH"] = DEMO_DB
    return subprocess.call([sys.executable, "-m", "streamlit", "run", "app/dashboard/app.py"], env=env)


def cmd_demo_seed(s: Settings, a) -> int:
    from app.demo import DEMO_DB, seed_demo
    seed_demo(s, DEMO_DB).close()
    print(f"Demo DB written to {DEMO_DB} (origin=demo). View with: python -m app.main dashboard --demo")
    return 0


def cmd_report(s: Settings, a) -> int:
    from app.analytics.summary import performance_summary
    from app.storage.db import Storage
    print(json.dumps(performance_summary(Storage(s.app.db_path), s.app.starting_equity_target), indent=2, default=str))
    return 0


def cmd_review(s: Settings, a) -> int:
    from app.agents.review import review_strategies
    from app.storage.db import Storage
    for r in review_strategies(Storage(s.app.db_path)):
        print("-", r)
    return 0


def cmd_halt(s: Settings, a) -> int:
    from app.storage.db import Storage
    Storage(s.app.db_path).set_halt(not a.resume, a.reason)
    print("trading halted" if not a.resume else "halt cleared")
    return 0


def cmd_paper_session(s: Settings, a) -> int:
    checks = run_preflight(s, online=True)
    print(format_report(checks))
    if any(c.status == "FAIL" for c in checks) or not paper_mode_confirmed(checks):
        print("\nRefusing to start: preflight failed or paper mode not confirmed (fail closed).")
        return 1
    if not s.enable_paper_trading:
        print("\nENABLE_PAPER_TRADING=false: running as DRY-RUN (no orders will be submitted).")
    elif not a.confirm_paper:
        print("\nENABLE_PAPER_TRADING=true but --confirm-paper not given. Re-run with --confirm-paper to allow "
              "PAPER orders.")
        return 1
    dry = not (s.enable_paper_trading and a.confirm_paper)
    db, flow, ex = _build(s, need_broker=True)
    rep = ex.reconcile()
    print(f"startup reconcile: ok={rep.ok} issues={rep.issues} warnings={rep.warnings}")
    try:
        while True:
            st = session_status(utcnow())
            if st.is_open:
                _print_cycle(flow.run_cycle(dry_run=dry))
            else:
                print(f"{utcnow().isoformat()} market closed; waiting")
                if a.once:
                    return 0
            if a.once:
                return 0
            time.sleep(s.app.scan_interval_minutes * 60)
    except KeyboardInterrupt:
        print("stopped by operator; broker-side bracket stops remain active")
        return 0



def cmd_crewai_session(s: Settings, a) -> int:
    """Repeated CrewAI scans; this command is strictly dry-run."""
    from app.flow.crewai_flow import build_crewai_flow

    if s.enable_paper_trading:
        print(
            "Refusing to start: set ENABLE_PAPER_TRADING=false "
            "for the CrewAI dry-run session."
        )
        return 1

    checks = run_preflight(s, online=True)
    print(format_report(checks))
    if any(c.status == "FAIL" for c in checks) or not paper_mode_confirmed(checks):
        print("\\nRefusing to start: preflight failed or paper mode not confirmed.")
        return 1

    if not s.has_alpaca_keys():
        print("Alpaca keys missing; cannot run live-market-data research.")
        return 2

    # Deliberately build without a broker/execution service.
    db, flow, _ = _build(s, need_broker=False)
    interval = s.app.scan_interval_minutes
    print(
        f"Starting CrewAI DRY-RUN session; interval={interval} minutes. "
        "No broker execution service is configured. Ctrl+C stops the session."
    )

    try:
        while True:
            status = session_status(utcnow())
            if status.is_open:
                started = utcnow().isoformat()
                print(f"\\n[{started}] Starting CrewAI cycle")
                state = build_crewai_flow(flow, dry_run=True).kickoff()
                _print_cycle(state)
                print(f"[{utcnow().isoformat()}] CrewAI cycle completed")
            else:
                print(f"{utcnow().isoformat()} market closed; waiting")

            if a.once:
                return 0

            time.sleep(interval * 60)
    except KeyboardInterrupt:
        print("\\nCrewAI dry-run session stopped by operator.")
        return 0

def cmd_smoke_order(s: Settings, a) -> int:
    if not (s.enable_paper_trading and a.i_understand):
        print("Requires ENABLE_PAPER_TRADING=true and --i-understand. Submits then cancels a 1-share PAPER limit order.")
        return 1
    db, flow, ex = _build(s, need_broker=True)
    q = flow.market.latest_quotes([a.symbol]).get(a.symbol)
    if not q:
        print("no quote available")
        return 2
    print(ex.paper_smoke_test(a.symbol, q["mid"]))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="alphaforge", description="ALPHAFORGE paper-trading system (paper only)")
    p.add_argument("--config", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)
    x = sub.add_parser("preflight"); x.add_argument("--offline", action="store_true"); x.set_defaults(fn=cmd_preflight)
    x = sub.add_parser("scan", help="one dry-run cycle; never submits orders"); x.add_argument("--crewai", action="store_true"); x.set_defaults(fn=cmd_scan)
    x = sub.add_parser("replay"); x.add_argument("--demo", action="store_true"); x.add_argument("--data-dir", default="data/replay"); x.add_argument("--rs", action="store_true"); x.set_defaults(fn=cmd_replay)
    x = sub.add_parser("dashboard"); x.add_argument("--demo", action="store_true"); x.set_defaults(fn=cmd_dashboard)
    x = sub.add_parser("demo-seed"); x.set_defaults(fn=cmd_demo_seed)
    x = sub.add_parser("report"); x.set_defaults(fn=cmd_report)
    x = sub.add_parser("review"); x.set_defaults(fn=cmd_review)
    x = sub.add_parser("halt"); x.add_argument("--resume", action="store_true"); x.add_argument("--reason", default="operator"); x.set_defaults(fn=cmd_halt)
    x = sub.add_parser("paper-session"); x.add_argument("--confirm-paper", action="store_true"); x.add_argument("--once", action="store_true"); x.set_defaults(fn=cmd_paper_session)
    x = sub.add_parser("crewai-session", help="repeated CrewAI dry-run cycles; never submits orders"); x.add_argument("--once", action="store_true", help="perform one market-session check/cycle and exit"); x.set_defaults(fn=cmd_crewai_session)
    x = sub.add_parser("smoke-order"); x.add_argument("--symbol", default="SPY"); x.add_argument("--i-understand", action="store_true"); x.set_defaults(fn=cmd_smoke_order)
    a = p.parse_args(argv)
    s = load_settings(a.config)
    setup_logging(s.log_level, s.app.log_dir, s.secrets.values())
    return a.fn(s, a)


if __name__ == "__main__":
    sys.exit(main())
