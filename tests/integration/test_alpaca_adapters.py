"""Alpaca REST adapters against a mocked requests session. Payload shapes follow Alpaca's documented REST API
but have NOT been exercised against the live service in this build."""
import pytest
import requests

from app.config import DATA_BASE_URL, PAPER_BASE_URL
from app.data.http import HttpClient, NotFound, ProviderError, RateLimited, TimeoutOrNetwork
from app.data.providers import AlpacaMarketData, AlpacaNews, bars_from_records
from app.execution.broker import AlpacaPaperBroker, map_status
from app.schemas import OrderStatus
from tests.conftest import MARKET_NOW, make_settings


class Resp:
    def __init__(self, code=200, body=None, text=""):
        self.status_code, self._b, self.text, self.content = code, body, text, b"x"

    def json(self):
        return self._b


class Sess:
    def __init__(self, script):
        self.script, self.calls = list(script), []

    def _next(self):
        r = self.script.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    def get(self, url, **kw):
        self.calls.append(("GET", url, kw)); return self._next()

    def request(self, method, url, **kw):
        self.calls.append((method, url, kw)); return self._next()


def client(script, retries=3, base=DATA_BASE_URL):
    s = Sess(script)
    return HttpClient(base, {"APCA-API-KEY-ID": "k"}, retries=retries, backoff=0, session=s, sleep=lambda _: None,
                      secrets=["testsecretvalue123456"]), s


def test_get_retries_on_429_then_succeeds():
    c, s = client([Resp(429, text="slow"), Resp(200, {"ok": 1})])
    assert c.get("/x") == {"ok": 1} and len(s.calls) == 2


def test_get_retries_timeouts_then_raises_after_bound():
    c, s = client([requests.Timeout("t")] * 3, retries=2)
    with pytest.raises(TimeoutOrNetwork):
        c.get("/x")
    assert len(s.calls) == 3


def test_get_4xx_not_retried_and_404_is_notfound():
    c, s = client([Resp(403, text="forbidden")])
    with pytest.raises(ProviderError):
        c.get("/x")
    assert len(s.calls) == 1
    c, _ = client([Resp(404)])
    with pytest.raises(NotFound):
        c.get("/x")


def test_error_text_is_redacted():
    c, _ = client([Resp(401, text="bad key testsecretvalue123456")], retries=0)
    with pytest.raises(ProviderError) as e:
        c.get("/x")
    assert "testsecretvalue123456" not in str(e.value)


def test_post_is_never_retried():
    c, s = client([requests.Timeout("t"), Resp(200, {})], base=PAPER_BASE_URL)
    with pytest.raises(TimeoutOrNetwork):
        c.request_once("POST", "/v2/orders", json_body={"a": 1})
    assert len(s.calls) == 1


def test_bars_parsing_pagination_and_bad_rows():
    recs = [{"t": "2026-03-04T05:00:00Z", "o": 10, "h": 11, "l": 9, "c": 10.5, "v": 1000},
            {"t": "2026-03-05T05:00:00Z", "o": 10, "h": 8, "l": 9, "c": 10.5, "v": 1000},       # high < low -> dropped
            {"t": "2026-03-06T05:00:00Z", "o": 10, "h": 11, "l": 9, "c": 0, "v": 1000}]         # zero price -> dropped
    assert len(bars_from_records(recs)) == 1
    c, s = client([Resp(200, {"bars": {"AAPL": recs[:1]}, "next_page_token": "t1"}),
                   Resp(200, {"bars": {"AAPL": recs[2:]}, "next_page_token": None})])
    out = AlpacaMarketData(make_settings(), http=c).daily_bars(["AAPL"], MARKET_NOW, MARKET_NOW)
    assert len(s.calls) == 2 and s.calls[0][2]["params"]["feed"] == "iex" and len(out["AAPL"]) == 1


def test_quotes_and_news_parsing():
    q = {"quotes": {"AAPL": {"t": "2026-03-09T15:59:50Z", "bp": 99.9, "ap": 100.1}, "BAD": {"t": "2026-03-09T15:59:50Z", "bp": 0, "ap": 0}}}
    c, _ = client([Resp(200, q)])
    quotes = AlpacaMarketData(make_settings(), http=c).latest_quotes(["AAPL", "BAD"])
    assert quotes["AAPL"]["mid"] == pytest.approx(100.0) and quotes["BAD"]["mid"] == 0.0
    news = {"news": [{"id": 1, "headline": "Apple beats", "summary": "s", "source": "benzinga", "url": "https://x",
                      "created_at": "2026-03-09T14:00:00Z", "symbols": ["AAPL", "BTC/USD"]},
                     {"headline": "missing id"}], "next_page_token": None}
    c, _ = client([Resp(200, news)])
    items = AlpacaNews(make_settings(), http=c).news(["AAPL"], MARKET_NOW)
    assert len(items) == 1 and items[0].symbols == ["AAPL"] and items[0].published_at.tzinfo is not None


def test_status_mapping_covers_alpaca_states():
    assert map_status("new") == map_status("pending_new") == map_status("held") == OrderStatus.ACCEPTED
    assert map_status("partially_filled") == OrderStatus.PARTIALLY_FILLED and map_status("filled") == OrderStatus.FILLED
    assert map_status("done_for_day") == OrderStatus.EXPIRED and map_status("rejected") == OrderStatus.REJECTED
    assert map_status("something_new") == OrderStatus.UNKNOWN


def test_paper_broker_account_orders_and_cancel():
    acct = {"equity": "100000", "cash": "90000", "buying_power": "200000", "last_equity": "99000", "status": "ACTIVE"}
    c, s = client([Resp(200, acct), Resp(200, {"id": "o1", "status": "accepted"}), Resp(404)], base=PAPER_BASE_URL)
    b = AlpacaPaperBroker(make_settings(), http=c)
    a = b.get_account()
    assert a.equity == 100000 and a.last_equity == 99000 and a.status == "ACTIVE"
    assert b.submit_order({"symbol": "AAPL"})["id"] == "o1"
    b.cancel_order("gone")                                             # 404 on cancel is tolerated
    assert s.calls[0][1].startswith(PAPER_BASE_URL) and "//api.alpaca.markets" not in s.calls[0][1]
