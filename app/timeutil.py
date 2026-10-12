"""UTC-internal time helpers and US/Eastern market calendar (DST- and holiday-aware)."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd

ET = ZoneInfo("America/New_York")
UTC = timezone.utc

_cal = None


def utcnow() -> datetime:
    return datetime.now(UTC)


def ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        raise ValueError("naive datetime not allowed; timestamps must be timezone-aware")
    return dt.astimezone(UTC)


def parse_ts(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        return ensure_utc(value)
    v = value.strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(v)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def calendar():
    global _cal
    if _cal is None:
        import exchange_calendars as xcals
        _cal = xcals.get_calendar("XNYS")
    return _cal


def _to_utc_ts(x) -> datetime:
    t = pd.Timestamp(x)
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    return t.to_pydatetime()


@dataclass(frozen=True)
class SessionStatus:
    is_session_day: bool
    is_open: bool
    session_date: date
    open_utc: datetime | None
    close_utc: datetime | None
    minutes_since_open: float | None
    minutes_to_close: float | None
    early_close: bool


def session_status(now: datetime) -> SessionStatus:
    now = ensure_utc(now)
    et_date = now.astimezone(ET).date()
    cal = calendar()
    ts = pd.Timestamp(et_date)
    try:
        is_day = bool(cal.is_session(ts))
    except Exception:  # outside calendar bounds -> treat as closed (fail safe)
        is_day = False
    if not is_day:
        return SessionStatus(False, False, et_date, None, None, None, None, False)
    o, c = _to_utc_ts(cal.session_open(ts)), _to_utc_ts(cal.session_close(ts))
    is_open = o <= now < c
    regular_close = datetime(et_date.year, et_date.month, et_date.day, 16, 0, tzinfo=ET).astimezone(UTC)
    return SessionStatus(
        True, is_open, et_date, o, c,
        (now - o).total_seconds() / 60 if now >= o else None,
        (c - now).total_seconds() / 60 if now < c else None,
        c < regular_close,
    )


def last_completed_session(now: datetime) -> date:
    """Most recent session date whose daily bar is complete at `now`."""
    now = ensure_utc(now)
    cal = calendar()
    st = session_status(now)
    if st.is_session_day and st.close_utc and now >= st.close_utc:
        return st.session_date
    ts = pd.Timestamp(st.session_date)
    prev = cal.date_to_session(ts, direction="previous") if not st.is_session_day else cal.previous_session(ts)
    return pd.Timestamp(prev).date()


def sessions_between(start: date, end: date) -> int:
    """Number of trading sessions in (start, end]."""
    cal = calendar()
    if end <= start:
        return 0
    s = cal.sessions_in_range(pd.Timestamp(start), pd.Timestamp(end))
    return max(len(s) - (1 if pd.Timestamp(start) in s else 0), 0)
