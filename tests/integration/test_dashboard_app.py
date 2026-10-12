import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from app.demo import seed_demo
from tests.conftest import ROOT, make_settings


def run_app(monkeypatch, db_path):
    monkeypatch.setenv("ALPHAFORGE_CONFIG", str(ROOT / "config" / "default.yaml"))
    monkeypatch.setenv("ALPHAFORGE_DB_PATH", str(db_path))
    monkeypatch.setenv("ALPACA_API_KEY_ID", "PKTESTKEY1234567890")
    monkeypatch.setenv("ALPACA_API_SECRET_KEY", "testsecretvalue123456")
    return AppTest.from_file(str(ROOT / "app" / "dashboard" / "app.py"), default_timeout=90).run()


def test_dashboard_renders_demo_with_banner_and_no_exceptions(monkeypatch, tmp_path):
    p = tmp_path / "demo.db"
    seed_demo(make_settings(), str(p)).close()
    at = run_app(monkeypatch, p)
    assert not at.exception
    assert any("DEMO DATA" in w.value for w in at.warning)
    assert len(at.tabs) == 6 and "Equity" in [m.label for m in at.metric]


def test_dashboard_empty_db_shows_honest_empty_states_not_fake_numbers(monkeypatch, tmp_path):
    at = run_app(monkeypatch, tmp_path / "empty.db")
    assert not at.exception
    assert any("No data yet" in i.value for i in at.info)
    assert "Equity" not in [m.label for m in at.metric]
    assert not any("DEMO DATA" in w.value for w in at.warning)


def test_dashboard_never_displays_credentials(monkeypatch, tmp_path):
    at = run_app(monkeypatch, tmp_path / "e2.db")
    blob = " ".join(str(j.value) for j in at.json) + " ".join(c.value for c in at.caption)
    assert "PKTESTKEY1234567890" not in blob and "testsecretvalue123456" not in blob
