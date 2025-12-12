import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .config import COLORS, CHART_PALETTE

def create_efficient_frontier_chart(df, selected=None, r_min=None):
    if df is None or df.empty:
        return go.Figure()

    fig = go.Figure()

    if r_min is not None:
        df_below = df[df["Return"] < r_min]
        df_above = df[df["Return"] >= r_min]
    else:
        df_below = pd.DataFrame()
        df_above = df

    if not df_below.empty:
        fig.add_trace(
            go.Scatter(
                x=df_below["Risk"] * 100,
                y=df_below["Return"] * 100,
                mode="lines",
                name="Sous minimum",
                line=dict(color=COLORS["pearl"], width=2, dash="dot"),
                hoverinfo="skip",
            )
        )

    if not df_above.empty:
        fig.add_trace(
            go.Scatter(
                x=df_above["Risk"] * 100,
                y=df_above["Return"] * 100,
                mode="lines+markers",
                name="Frontiere Efficiente",
                line=dict(color=COLORS["chart_1"], width=5),
                marker=dict(
                    size=12,
                    color=df_above["Sharpe"],
                    colorscale=[
                        [0, COLORS["chart_1"]],
                        [0.5, COLORS["chart_3"]],
                        [1, COLORS["chart_2"]],
                    ],
                    showscale=True,
                    colorbar=dict(
                        title=dict(text="Sharpe", font=dict(size=14, color=COLORS["ink"])),
                        thickness=20,
                        len=0.6,
                        x=1.08,
                        xpad=20,
                        tickfont=dict(size=12, color=COLORS["ink"]),
                        bgcolor="rgba(255,255,255,0.95)",
                        outlinewidth=1,
                        outlinecolor=COLORS["pearl"],
                        nticks=6,
                    ),
                    line=dict(color=COLORS["white"], width=1.5),
                ),
                hovertemplate="<b>Portefeuille</b><br>Rendement: %{y:.2f}%<br>Risque: %{x:.2f}%<extra></extra>",
            )
        )

        if r_min is not None:
            fig.add_hline(y=r_min * 100, line=dict(color=COLORS["terminal_orange"], width=2, dash="dash"))
            r_min_val = r_min * 100
            fig.add_annotation(
                x=0,
                y=r_min_val,
                xref="paper",
                yref="y",
                text=f"<b>{r_min_val:.1f} %</b>",
                font=dict(size=17, color=COLORS["terminal_orange"], family="DM Mono"),
                showarrow=False,
                xanchor="right",
                xshift=-10,
            )

        if selected:
            fig.add_trace(
                go.Scatter(
                    x=[selected["Risk"] * 100],
                    y=[selected["Return"] * 100],
                    mode="markers",
                    name="Portefeuille Selectionne",
                    marker=dict(
                        size=16,
                        color=COLORS["warning"],
                        symbol="diamond",
                        line=dict(color=COLORS["white"], width=2),
                    ),
                    hovertemplate="<b>Portefeuille Selectionne</b><br>Rendement: %{y:.2f}%<br>Risque: %{x:.2f}%<extra></extra>",
                )
            )

    fig.update_layout(
        font=dict(family="DM Sans", color=COLORS["ink"]),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=COLORS["white"],
        margin=dict(l=100, r=180, t=60, b=70),
        height=800,
        hovermode="closest",
        showlegend=False,
        xaxis=dict(
            title=dict(text="Volatilite (%)", font=dict(size=20, color=COLORS["ink"], family="DM Sans"), standoff=50),
            gridcolor=COLORS["ivory"],
            gridwidth=1,
            zeroline=False,
            tickfont=dict(size=15, color=COLORS["graphite"]),
            showline=True,
            linewidth=1,
            linecolor=COLORS["pearl"],
        ),
        yaxis=dict(
            title=dict(text="Rendement Annuel (%)", font=dict(size=20, color=COLORS["ink"], family="DM Sans"), standoff=70),
            gridcolor=COLORS["ivory"],
            gridwidth=1,
            zeroline=False,
            tickfont=dict(size=15, color=COLORS["graphite"]),
            showline=True,
            linewidth=1,
            linecolor=COLORS["pearl"],
        ),
    )
    return fig

def create_sector_bar(weights, tickers, sector_map):
    df = pd.DataFrame({"Ticker": tickers, "Weight": np.asarray(weights) * 100})
    df["Sector"] = df["Ticker"].map(sector_map).fillna("Autre")
    df_sector = df.groupby("Sector")["Weight"].sum().reset_index()
    df_sector = df_sector[df_sector["Weight"] > 0].sort_values("Weight", ascending=True)
    if df_sector.empty:
        return go.Figure()

    fig = go.Figure(
        data=[
            go.Bar(
                x=df_sector["Weight"],
                y=df_sector["Sector"],
                orientation="h",
                marker=dict(color=COLORS["chart_1"]),
            )
        ]
    )
    fig.update_layout(
        font=dict(family="DM Sans"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=COLORS["white"],
        margin=dict(l=100, r=30, t=20, b=50),
        height=320,
        xaxis=dict(title="Allocation (%)", gridcolor=COLORS["ivory"]),
    )
    return fig

def create_backtest_chart(returns, weights, initial=1000):
    w = np.asarray(weights, dtype=float)
    port_ret = (returns * w).sum(axis=1)
    cumulative = initial * (1 + port_ret).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max * 100

    fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3], shared_xaxes=True, vertical_spacing=0.08)

    fig.add_trace(
        go.Scatter(
            x=cumulative.index,
            y=cumulative.values,
            mode="lines",
            name="Valeur",
            line=dict(color=COLORS["chart_1"], width=2),
            fill="tozeroy",
            fillcolor="rgba(30,58,95,0.1)",
        ),
        row=1,
        col=1,
    )
    fig.add_hline(y=initial, line=dict(color=COLORS["silver"], dash="dash", width=1), row=1, col=1)

    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            mode="lines",
            name="Drawdown",
            line=dict(color=COLORS["danger"], width=1.5),
            fill="tozeroy",
            fillcolor="rgba(229,57,53,0.15)",
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        font=dict(family="DM Sans"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=COLORS["white"],
        margin=dict(l=50, r=30, t=20, b=40),
        height=400,
        showlegend=False,
    )
    fig.update_yaxes(title_text="Valeur (EUR)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)

    stats = {
        "final": float(cumulative.iloc[-1]),
        "total_return": float((cumulative.iloc[-1] / initial - 1) * 100),
        "max_dd": float(drawdown.min()),
        "volatility": float(port_ret.std() * np.sqrt(252) * 100),
        "sharpe": float((port_ret.mean() * 252) / (port_ret.std() * np.sqrt(252))) if port_ret.std() > 0 else 0.0,
    }
    return fig, stats

def create_allocation_donut(weights, tickers, min_pct=0.5):
    df = pd.DataFrame({"Ticker": tickers, "Weight": np.asarray(weights) * 100})
    df = df[df["Weight"] >= float(min_pct)].sort_values("Weight", ascending=False)
    if df.empty:
        return go.Figure()

    fig = go.Figure(
        data=[
            go.Pie(
                labels=df["Ticker"],
                values=df["Weight"],
                hole=0.6,
                marker=dict(colors=CHART_PALETTE[: len(df)], line=dict(color=COLORS["white"], width=2)),
                textinfo="label+percent",
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        font=dict(family="DM Sans"),
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=40, t=20, b=40),
        height=320,
        showlegend=False,
        annotations=[dict(text="Actifs", x=0.5, y=0.5, font_size=14, showarrow=False)],
    )
    return fig
