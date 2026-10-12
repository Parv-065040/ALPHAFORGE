"""Optional CrewAI Flow wrapper around the SAME steps as app.flow.pipeline.TradingFlow.

STATUS: written against the documented crewai Flow API (Flow, @start, @listen) but NOT executed or verified
by the author. The default `paper-session`/`scan` commands use the plain-Python flow, which is the tested path.
Install with: pip install crewai   (then run: python -m app.main scan --crewai)
"""
from __future__ import annotations

from app.flow.pipeline import CycleState, TradingFlow


def build_crewai_flow(core: TradingFlow, dry_run: bool = True):
    try:
        from crewai.flow.flow import Flow, listen, start  # type: ignore
    except Exception as e:  # crewai not installed / API changed
        raise RuntimeError(f"CrewAI Flow unavailable ({type(e).__name__}: {e}). Use the default plain flow.") from e

    class AlphaForgeFlow(Flow):  # type: ignore[misc]
        def __init__(self):
            super().__init__()
            self.cycle = CycleState(now=core.now_fn(), dry_run=dry_run)

        @start()
        def collect_data(self):
            core.step_data(self.cycle)
            return "data"

        @listen(collect_data)
        def run_scan(self, _):
            core.step_scan(self.cycle)
            return "scan"

        @listen(run_scan)
        def propose_and_risk(self, _):
            core.step_propose_and_risk(self.cycle)
            return "risk"

        @listen(propose_and_risk)
        def execute(self, _):
            core.step_execute(self.cycle)
            return self.cycle

    return AlphaForgeFlow()
