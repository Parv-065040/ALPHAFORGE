from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go


def equity_chart(df):
    fig = px.line(df, x="ts", y="equity", title="Equity (broker snapshots)")
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=320)
    return fig


def sentiment_distribution(df):
    fig = px.bar(df, x="label", y="n", color="label", title="Sentiment distribution",
                 color_discrete_map={"positive": "#2e9e5b", "negative": "#d64545", "neutral": "#8a8f98"})
    fig.update_layout(showlegend=False, height=280, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def sentiment_trend(feed):
    d = feed.copy()
    d["hour"] = d["published_at"].astype(str).str[:13]
    g = d.groupby("hour")["score"].mean().reset_index()
    fig = go.Figure(go.Scatter(x=g["hour"], y=g["score"], mode="lines+markers"))
    fig.update_layout(title="Mean raw sentiment per hour", height=280, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def score_components(cands):
    rows = []
    for _, r in cands.iterrows():
        for k, v in (r["components"] or {}).items():
            rows.append({"candidate": f"{r['symbol']}:{r['strategy']}", "component": k, "value": v})
    if not rows:
        return None
    import pandas as pd
    fig = px.bar(pd.DataFrame(rows), x="candidate", y="value", color="component", title="Score components")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
    return fig
