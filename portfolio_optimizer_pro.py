"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         PORTFOLIO OPTIMIZER PRO                               ║
║                Suite d'Analyse d'Investissement Professionnelle               ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import json
from scipy.optimize import minimize
from typing import Tuple, Dict, Optional
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION DE LA PAGE
# =============================================================================

st.set_page_config(
    page_title="Portfolio Optimizer Pro",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# SYSTÈME DE DESIGN
# =============================================================================

COLORS = {
    'ink': '#1a1a1a',
    'charcoal': '#2d2d2d',
    'slate': '#4a4a4a',
    'graphite': '#6b6b6b',
    'silver': '#9a9a9a',
    'pearl': '#c4c4c4',
    'ivory': '#f5f5f3',
    'paper': '#fafaf8',
    'white': '#ffffff',
    'terminal_orange': '#2d5a3d',
    'success': '#00a67d',
    'warning': '#f0b429',
    'danger': '#e53935',
    'info': '#2196f3',
    'chart_1': '#1e3a5f',
    'chart_2': '#2d5a3d',
    'chart_3': '#00a67d',
    'chart_4': '#7c4dff',
    'chart_5': '#f0b429',
    'chart_6': '#e53935',
    'chart_7': '#26c6da',
    'chart_8': '#ec407a',
}

CHART_PALETTE = [COLORS['chart_1'], COLORS['chart_2'], COLORS['chart_3'],
                 COLORS['chart_4'], COLORS['chart_5'], COLORS['chart_6'],
                 COLORS['chart_7'], COLORS['chart_8']]

# =============================================================================
# CSS PERSONNALISÉ
# =============================================================================

CUSTOM_CSS = f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&family=Fraunces:wght@700;800&family=Inter:wght@400;500;600;700&display=swap');

    :root {{
        --ink: {COLORS['ink']};
        --charcoal: {COLORS['charcoal']};
        --slate: {COLORS['slate']};
        --graphite: {COLORS['graphite']};
        --silver: {COLORS['silver']};
        --pearl: {COLORS['pearl']};
        --ivory: {COLORS['ivory']};
        --paper: {COLORS['paper']};
        --white: {COLORS['white']};
        --accent: {COLORS['terminal_orange']};
        --success: {COLORS['success']};
        --warning: {COLORS['warning']};
        --danger: {COLORS['danger']};
        --font-display: 'Fraunces', Georgia, serif;
        --font-body: 'DM Sans', -apple-system, sans-serif;
        --font-mono: 'DM Mono', 'SF Mono', monospace;
    }}

    .stApp {{
        background: var(--paper);
        font-family: var(--font-body);
        color: var(--ink);
    }}

    .block-container {{
        padding-top: 2rem;
        max-width: 1400px;
    }}

    [data-testid="stSidebar"] {{
        background: var(--white);
        border-right: 1px solid var(--pearl);
    }}

    [data-testid="stSidebar"] label {{
        color: var(--slate) !important;
        font-size: 0.75rem !important;
        font-weight: 500 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }}

    .masthead {{
        border-bottom: 3px solid var(--accent);
        padding-bottom: 1.5rem;
        margin-bottom: 2rem;
        text-align: center;
    }}

    .masthead-title {{
        font-family: 'Inter', -apple-system, sans-serif;
        font-size: clamp(2.5rem, 5vw, 4rem);
        font-weight: 700;
        color: var(--ink);
        line-height: 1;
        margin: 0;
        letter-spacing: -0.03em;
        text-transform: uppercase;
    }}

    .masthead-subtitle {{
        font-family: var(--font-body);
        font-size: 1rem;
        color: var(--graphite);
        margin-top: 1rem;
        max-width: 700px;
        margin-left: auto;
        margin-right: auto;
    }}

    .section-header {{
        display: flex;
        align-items: baseline;
        gap: 1rem;
        margin: 3rem 0 1.5rem 0;
        border-bottom: 1px solid var(--pearl);
        padding-bottom: 1rem;
    }}

    .section-number {{
        font-family: var(--font-display);
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--pearl);
        line-height: 1;
    }}

    .section-title {{
        font-family: var(--font-display);
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--ink);
        margin: 0;
    }}

    .section-desc {{
        font-size: 0.9rem;
        color: var(--graphite);
        margin-top: 0.25rem;
    }}

    .callout {{
        background: var(--ivory);
        border-left: 3px solid var(--ink);
        padding: 1.5rem;
        margin: 1rem 0;
        font-size: 1rem;
        line-height: 1.6;
    }}

    .callout.success {{
        border-left-color: var(--success);
        background: rgba(0, 166, 125, 0.05);
    }}

    .callout.warning {{
        border-left-color: var(--warning);
        background: rgba(240, 180, 41, 0.05);
    }}

    .stTabs [data-baseweb="tab-list"] {{
        gap: 0;
        background: transparent;
        border-bottom: 1px solid var(--pearl);
    }}

    .stTabs [data-baseweb="tab"] {{
        background: transparent;
        border: none;
        border-bottom: 2px solid transparent;
        color: var(--graphite);
        font-weight: 500;
        padding: 1rem 1.5rem;
    }}

    .stTabs [aria-selected="true"] {{
        border-bottom: 2px solid var(--ink) !important;
        color: var(--ink) !important;
    }}

    .stButton > button {{
        background: var(--accent);
        color: var(--white) !important;
        border: none;
        border-radius: 4px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 0.85rem;
    }}

    .stButton > button:hover {{
        background: #234a30;
        box-shadow: 0 2px 8px rgba(45, 90, 61, 0.3);
    }}

    .formula-block {{
        background: var(--white);
        color: var(--ink);
        padding: 1.5rem;
        margin: 1rem 0;
        font-family: var(--font-mono);
        border: 1px solid var(--pearl);
        border-left: 4px solid var(--accent);
    }}

    .formula-label {{
        font-size: 0.7rem;
        color: var(--accent);
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }}

    .sidebar-section {{
        font-family: var(--font-mono);
        font-size: 0.7rem;
        font-weight: 500;
        color: var(--ink);
        text-transform: uppercase;
        letter-spacing: 0.1em;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
        border-bottom: 1px solid var(--pearl);
    }}

    table.dataframe {{
        width: 100%;
        background: white !important;
        border-collapse: collapse;
        font-family: var(--font-body);
        margin: 1rem 0;
    }}

    table.dataframe th {{
        background: var(--ivory) !important;
        color: var(--ink) !important;
        padding: 0.75rem 1rem;
        text-align: left;
        font-weight: 600;
        border-bottom: 2px solid var(--pearl);
    }}

    table.dataframe td {{
        background: white !important;
        color: var(--ink) !important;
        padding: 0.75rem 1rem;
        border-bottom: 1px solid var(--ivory);
    }}

    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    
    /* Barre d'outils Plotly - Centrée, horizontale, plus grande */
    .modebar {{
        left: 50% !important;
        right: auto !important;
        transform: translateX(-50%) !important;
        top: 10px !important;
        background: transparent !important;
    }}
    
    .modebar-group {{
        display: flex !important;
        gap: 5px !important;
        background: transparent !important;
    }}
    
    .modebar-btn {{
        width: 40px !important;
        height: 40px !important;
    }}
    
    .modebar-btn svg {{
        width: 28px !important;
        height: 28px !important;
    }}
    
    .modebar-container {{
        background: transparent !important;
        border-radius: 8px !important;
        padding: 8px 15px !important;
        box-shadow: none !important;
    }}
    
    .js-plotly-plot .plotly .modebar {{
        background: transparent !important;
    }}
    
    .js-plotly-plot .plotly .modebar-group {{
        background: transparent !important;
    }}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =============================================================================
# COMPOSANTS UI
# =============================================================================

def render_masthead():
    """Affiche l'en-tête principal."""
    st.markdown("""
        <div class="masthead">
            <h1 class="masthead-title">Portfolio Optimizer</h1>
            <p class="masthead-subtitle">
                Optimisation de portefeuille multi-objectifs utilisant le cadre moyenne-variance 
                de Markowitz et la simulation Monte Carlo avec contraintes de cardinalité.
            </p>
        </div>
    """, unsafe_allow_html=True)


def render_section(number: str, title: str, description: str = ""):
    """Affiche un en-tête de section."""
    st.markdown(f"""
        <div class="section-header">
            <span class="section-number">{number}</span>
            <div class="section-content">
                <h2 class="section-title">{title}</h2>
                <p class="section-desc">{description}</p>
            </div>
        </div>
    """, unsafe_allow_html=True)


def render_callout(content: str, style: str = ""):
    """Affiche une boîte d'information."""
    st.markdown(f'<div class="callout {style}">{content}</div>', unsafe_allow_html=True)


def render_metrics_strip(metrics: list):
    """Affiche une bande de métriques."""
    cols = st.columns(len(metrics))
    for i, m in enumerate(metrics):
        with cols[i]:
            value = m['value']
            label = m['label']
            if m.get('class') == 'positive':
                color = COLORS['success']
            elif m.get('class') == 'negative':
                color = COLORS['danger']
            else:
                color = COLORS['ink']

            st.markdown(f'''
                <div style="background: #ffffff; border: 1px solid #c4c4c4; padding: 1.25rem; text-align: center;">
                    <div style="font-family: monospace; font-size: 0.7rem; font-weight: 500; color: #6b6b6b; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.5rem;">{label}</div>
                    <div style="font-family: monospace; font-size: 1.75rem; font-weight: 600; color: {color};">{value}</div>
                </div>
            ''', unsafe_allow_html=True)


# =============================================================================
# CHARGEMENT DES DONNÉES
# =============================================================================

@st.cache_data
def load_market_data(data_dir: str = "data", json_path: str = "tick.json"):
    """
    Charge les données de marché depuis les fichiers CSV.
    Gère à la fois les fichiers par ticker individuel et par secteur.
    """
    if not os.path.exists(data_dir):
        st.error(f"Dossier de données '{data_dir}' introuvable. Lancez `python download.py` d'abord.")
        st.stop()

    # Charger les correspondances de secteurs
    sector_map = {}
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            sectors = json.load(f)
        sector_map = {t: s for s, tickers in sectors.items() for t in tickers}

    # Lister les fichiers CSV
    files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    if not files:
        st.error("Aucun fichier CSV trouvé dans le dossier data")
        st.stop()

    dfs = []

    for f in files:
        path = os.path.join(data_dir, f)
        try:
            # Détecter le format du fichier
            with open(path, 'r') as file_check:
                first_lines = [file_check.readline() for _ in range(5)]

            # Trouver la ligne qui contient "Date"
            skip_rows = 0
            ticker_name = os.path.splitext(f)[0].replace('_', ' ')

            for i, line in enumerate(first_lines):
                if line.startswith("Ticker,"):
                    parts = line.strip().split(',')
                    if len(parts) > 1 and parts[1]:
                        ticker_name = parts[1]
                if "Date" in line:
                    skip_rows = i
                    break

            # Lire le CSV
            df = pd.read_csv(path, skiprows=skip_rows, index_col=0, parse_dates=True)

            # Nettoyer les données
            df = df.apply(pd.to_numeric, errors='coerce')
            df = df.dropna(how='all')

            if df.empty:
                continue

            # Si fichier multi-colonnes (par secteur), garder toutes les colonnes
            if df.shape[1] > 1:
                for col in df.columns:
                    if not df[col].isna().all():
                        temp_df = df[[col]].dropna()
                        temp_df.columns = [col]
                        dfs.append(temp_df)
            else:
                # Fichier mono-colonne (par ticker)
                df.columns = [ticker_name]
                df = df.dropna()
                if not df.empty:
                    dfs.append(df)

        except Exception as e:
            st.warning(f"Erreur de lecture du fichier {f}: {e}")
            continue

    if not dfs:
        st.error("Aucune donnée valide n'a pu être chargée.")
        st.stop()

    # Fusionner toutes les données
    prices = pd.concat(dfs, axis=1)

    # Supprimer les colonnes dupliquées
    prices = prices.loc[:, ~prices.columns.duplicated()]

    # Remplir les valeurs manquantes par forward fill puis backward fill
    prices = prices.ffill().bfill()

    # Supprimer les lignes avec des NaN restants
    prices = prices.dropna()

    if prices.empty or prices.shape[1] < 2:
        st.error("Pas assez de données valides après nettoyage (besoin d'au moins 2 actifs).")
        st.stop()

    # Calculer les rendements et statistiques
    returns = prices.pct_change().dropna()
    mu = returns.mean() * 252
    sigma = returns.cov() * 252

    return prices, returns, mu, sigma, sector_map


# =============================================================================
# OPTIMISEUR DE PORTEFEUILLE
# =============================================================================

class PortfolioOptimizer:
    """Moteur d'optimisation de portefeuille multi-objectifs."""

    def __init__(self, mu, sigma, current_weights=None, transaction_cost=0.005):
        self.mu = mu.values
        self.sigma = sigma.values
        self.tickers = list(mu.index)
        self.n = len(mu)
        self.w_current = current_weights if current_weights is not None else np.zeros(self.n)
        self.c = transaction_cost

    def compute_performance(self, w):
        """Calcule rendement et volatilité du portefeuille."""
        ret = np.dot(w, self.mu)
        vol = np.sqrt(np.dot(w.T, np.dot(self.sigma, w)))
        return ret, vol

    def compute_transaction_cost(self, w):
        """Calcule les coûts de transaction."""
        return self.c * np.sum(np.abs(w - self.w_current))

    def compute_efficient_frontier(self, n_points=50):
        """Calcule la frontière efficiente de Markowitz."""
        results = []
        bounds = tuple((0, 1) for _ in range(self.n))
        init = np.array([1.0 / self.n] * self.n)

        target_returns = np.linspace(self.mu.min(), self.mu.max(), n_points)

        for target in target_returns:
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                {'type': 'eq', 'fun': lambda w, t=target: np.dot(w, self.mu) - t}
            ]

            try:
                sol = minimize(
                    lambda w: np.sqrt(np.dot(w.T, np.dot(self.sigma, w))),
                    init, method='SLSQP', bounds=bounds, constraints=constraints,
                    options={'maxiter': 1000}
                )

                if sol.success:
                    ret, vol = self.compute_performance(sol.x)
                    if vol > 0:
                        results.append({
                            'Return': ret,
                            'Risk': vol,
                            'Sharpe': ret / vol,
                            'Weights': sol.x
                        })
            except Exception:
                continue

        return pd.DataFrame(results)

    def run_monte_carlo(self, n_portfolios=5000, max_k=5, min_weight=0.01):
        """Exécute la simulation Monte Carlo avec contraintes de cardinalité."""
        results = []

        for _ in range(n_portfolios):
            w = np.random.random(self.n)

            # Application contrainte de cardinalité
            if max_k < self.n:
                top_k = np.argsort(w)[-max_k:]
                mask = np.zeros(self.n)
                mask[top_k] = 1
                w = w * mask

            # Suppression des petits poids
            w[w < min_weight] = 0
            if w.sum() > 0:
                w = w / w.sum()
            else:
                continue

            ret, vol = self.compute_performance(w)
            cost = self.compute_transaction_cost(w)

            if vol > 0:
                results.append({
                    'Return': ret,
                    'Risk': vol,
                    'Cost': cost,
                    'Sharpe': ret / vol,
                    'Weights': w,
                    'N_Assets': np.count_nonzero(w)
                })

        return pd.DataFrame(results)


# =============================================================================
# FONCTIONS GRAPHIQUES
# =============================================================================

def create_efficient_frontier_chart(df, selected=None, r_min=None):
    """Crée un graphique de frontière efficiente propre, style éditorial."""
    if df is None or df.empty:
        return go.Figure()

    fig = go.Figure()

    # Séparer les données selon le seuil r_min
    if r_min is not None:
        df_below = df[df['Return'] < r_min]
        df_above = df[df['Return'] >= r_min]
    else:
        df_below = pd.DataFrame()
        df_above = df

    # Zone grisée (sous r_min)
    if not df_below.empty:
        fig.add_trace(go.Scatter(
            x=df_below['Risk'] * 100,
            y=df_below['Return'] * 100,
            mode='lines',
            name='Sous minimum',
            line=dict(color=COLORS['pearl'], width=2, dash='dot'),
            hoverinfo='skip'
        ))

    # Frontière valide
    if not df_above.empty:
        fig.add_trace(go.Scatter(
            x=df_above['Risk'] * 100,
            y=df_above['Return'] * 100,
            mode='lines+markers',
            name='Frontière Efficiente',
            line=dict(color=COLORS['chart_1'], width=5),
            marker=dict(
                size=12,
                color=df_above['Sharpe'],
                colorscale=[[0, COLORS['chart_1']], [0.5, COLORS['chart_3']], [1, COLORS['chart_2']]],
                showscale=True,
                colorbar=dict(
                    title=dict(
                        text='Sharpe',
                        font=dict(size=14, color=COLORS['ink'])
                    ),
                    thickness=20,
                    len=0.6,
                    x=1.08,
                    xpad=20,
                    tickfont=dict(size=12, color=COLORS['ink']),
                    bgcolor='rgba(255,255,255,0.95)',
                    outlinewidth=1,
                    outlinecolor=COLORS['pearl'],
                    nticks=6
                ),
                line=dict(color=COLORS['white'], width=1.5)
            ),
            hovertemplate='<b>Portefeuille</b><br>Rendement: %{y:.2f}%<br>Risque: %{x:.2f}%<extra></extra>'
        ))

        # Ligne de rendement minimum
        if r_min is not None:
            fig.add_hline(
                y=r_min * 100,
                line=dict(color=COLORS['terminal_orange'], width=2, dash='dash')
            )

            # Calculs pour placer l'annotation proprement sur l'axe Y (côté gauche)
            y_min = df['Return'].min() * 100
            y_max = df['Return'].max() * 100
            y_range = y_max - y_min
            tick_interval = 5 if y_range > 20 else 2 if y_range > 10 else 1
            r_min_val = r_min * 100
            nearest_tick = round(r_min_val / tick_interval) * tick_interval
            distance_to_tick = abs(r_min_val - nearest_tick)
            yshift = 0
            if distance_to_tick < tick_interval * 0.4:
                if r_min_val >= nearest_tick:
                    yshift = 12
                else:
                    yshift = -12

            fig.add_annotation(
                x=0,
                y=r_min_val,
                xref='paper',
                yref='y',
                text=f"<b>{r_min_val:.1f} %</b>",
                font=dict(size=17, color=COLORS['terminal_orange'], family='DM Mono'),
                showarrow=False,
                xanchor='right',
                xshift=-10,
                yshift=yshift
            )

        # Marqueur portefeuille sélectionné
        if selected:
            fig.add_trace(go.Scatter(
                x=[selected['Risk'] * 100],
                y=[selected['Return'] * 100],
                mode='markers',
                name='Portefeuille Sélectionné',
                marker=dict(
                    size=16,
                    color=COLORS['warning'],
                    symbol='diamond',
                    line=dict(color=COLORS['white'], width=2)
                ),
                hovertemplate='<b>Portefeuille Sélectionné</b><br>Rendement: %{y:.2f}%<br>Risque: %{x:.2f}%<extra></extra>'
            ))

    # Mise en page (Layout)
    fig.update_layout(
        font=dict(family='DM Sans', color=COLORS['ink']),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor=COLORS['white'],
        margin=dict(l=100, r=180, t=60, b=70),
        height=800,
        hovermode='closest',
        showlegend=False,
        xaxis=dict(
            title=dict(
                text='Volatilité (%)',
                font=dict(size=20, color=COLORS['ink'], family='DM Sans'),
                standoff=50
            ),
            gridcolor=COLORS['ivory'],
            gridwidth=1,
            zeroline=False,
            tickfont=dict(size=15, color=COLORS['graphite']),
            showline=True,
            linewidth=1,
            linecolor=COLORS['pearl']
        ),
        yaxis=dict(
            title=dict(
                text='Rendement Annuel (%)',
                font=dict(size=20, color=COLORS['ink'], family='DM Sans'),
                standoff=70
            ),
            gridcolor=COLORS['ivory'],
            gridwidth=1,
            zeroline=False,
            tickfont=dict(size=15, color=COLORS['graphite']),
            showline=True,
            linewidth=1,
            linecolor=COLORS['pearl']
        ),
        modebar=dict(
            orientation='h',
            bgcolor='rgba(255,255,255,0.95)',
            activecolor='#2d5a3d'
        )
    )

    return fig


def create_monte_carlo_3d_chart(df):
    """Crée un nuage de points 3D pour Monte Carlo."""
    if df is None or df.empty:
        return go.Figure()

    fig = go.Figure(data=[go.Scatter3d(
        x=df['Risk'].values * 100,
        y=df['Return'].values * 100,
        z=df['Cost'].values * 100,
        mode='markers',
        marker=dict(
            size=3,
            color=df['Sharpe'].values,
            colorscale='Viridis',
            opacity=0.8,
            colorbar=dict(title='Sharpe', thickness=15)
        ),
        hovertemplate='Risque: %{x:.2f}%<br>Rendement: %{y:.2f}%<br>Coût: %{z:.3f}%<extra></extra>'
    )])

    fig.update_layout(
        height=600,
        margin=dict(l=0, r=50, t=30, b=0),
        scene=dict(
            xaxis_title='Risque (%)',
            yaxis_title='Rendement (%)',
            zaxis_title='Coût (%)',
            bgcolor='white'
        ),
        paper_bgcolor='white'
    )

    return fig


def create_backtest_chart(returns, weights, initial=1000):
    """Crée un graphique de backtest."""
    port_ret = (returns * weights).sum(axis=1)
    cumulative = initial * (1 + port_ret).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max * 100

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        shared_xaxes=True,
        vertical_spacing=0.08
    )

    # Valeur du portefeuille
    fig.add_trace(go.Scatter(
        x=cumulative.index,
        y=cumulative.values,
        mode='lines',
        name='Valeur',
        line=dict(color=COLORS['chart_1'], width=2),
        fill='tozeroy',
        fillcolor='rgba(30,58,95,0.1)',
    ), row=1, col=1)

    # Ligne investissement initial
    fig.add_hline(y=initial, line=dict(color=COLORS['silver'], dash='dash', width=1), row=1, col=1)

    # Drawdown
    fig.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown.values,
        mode='lines',
        name='Drawdown',
        line=dict(color=COLORS['danger'], width=1.5),
        fill='tozeroy',
        fillcolor='rgba(229,57,53,0.15)',
    ), row=2, col=1)

    fig.update_layout(
        font=dict(family='DM Sans'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor=COLORS['white'],
        margin=dict(l=50, r=30, t=20, b=40),
        height=400,
        showlegend=False
    )

    fig.update_yaxes(title_text="Valeur (€)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)

    stats = {
        'final': cumulative.iloc[-1],
        'total_return': (cumulative.iloc[-1] / initial - 1) * 100,
        'max_dd': drawdown.min(),
        'volatility': port_ret.std() * np.sqrt(252) * 100,
        'sharpe': (port_ret.mean() * 252) / (port_ret.std() * np.sqrt(252)) if port_ret.std() > 0 else 0
    }

    return fig, stats


def create_allocation_donut(weights, tickers):
    """Crée un graphique en beignet pour l'allocation."""
    df = pd.DataFrame({'Ticker': tickers, 'Weight': weights * 100})
    df = df[df['Weight'] > 0.5].sort_values('Weight', ascending=False)

    if df.empty:
        return go.Figure()

    fig = go.Figure(data=[go.Pie(
        labels=df['Ticker'],
        values=df['Weight'],
        hole=0.6,
        marker=dict(colors=CHART_PALETTE[:len(df)], line=dict(color=COLORS['white'], width=2)),
        textinfo='label+percent',
        textposition='outside',
    )])

    fig.update_layout(
        font=dict(family='DM Sans'),
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=40, r=40, t=20, b=40),
        height=320,
        showlegend=False,
        annotations=[dict(text='Actifs', x=0.5, y=0.5, font_size=14, showarrow=False)]
    )

    return fig


def create_sector_bar(weights, tickers, sector_map):
    """Crée un graphique à barres pour l'allocation sectorielle."""
    df = pd.DataFrame({'Ticker': tickers, 'Weight': weights * 100})
    df['Sector'] = df['Ticker'].map(sector_map).fillna('Autre')
    df_sector = df.groupby('Sector')['Weight'].sum().reset_index()
    df_sector = df_sector[df_sector['Weight'] > 0].sort_values('Weight', ascending=True)

    if df_sector.empty:
        return go.Figure()

    fig = go.Figure(data=[go.Bar(
        x=df_sector['Weight'],
        y=df_sector['Sector'],
        orientation='h',
        marker=dict(color=COLORS['chart_1'])
    )])

    fig.update_layout(
        font=dict(family='DM Sans'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor=COLORS['white'],
        margin=dict(l=100, r=30, t=20, b=50),
        height=320,
        xaxis=dict(title='Allocation (%)', gridcolor=COLORS['ivory']),
    )

    return fig


# =============================================================================
# APPLICATION PRINCIPALE
# =============================================================================

def main():
    # Initialisation du session state
    if 'selected_portfolio' not in st.session_state:
        st.session_state['selected_portfolio'] = None
    if 'frontier_df' not in st.session_state:
        st.session_state['frontier_df'] = None
    if 'mc_df' not in st.session_state:
        st.session_state['mc_df'] = None

    # Afficher l'en-tête
    render_masthead()

    # =================================
    # BARRE LATÉRALE
    # =================================
    with st.sidebar:
        try:
            prices, returns, mu, sigma, sector_map = load_market_data()
        except Exception as e:
            st.error(f"Erreur de chargement: {e}")
            st.stop()

        st.markdown('<div class="sidebar-section">Univers d\'Investissement</div>', unsafe_allow_html=True)

        all_sectors = sorted(set(sector_map.values())) if sector_map else []

        if all_sectors:
            selected_sectors = st.multiselect("Filtrer par secteur", all_sectors, default=[])
            if selected_sectors:
                available = [t for t in mu.index if sector_map.get(t) in selected_sectors]
            else:
                available = list(mu.index)
        else:
            available = list(mu.index)

        tickers = st.multiselect(
            "Sélectionner les actifs",
            available,
            default=[],
            placeholder="Sélectionnez au moins 2 actifs"
        )

        if len(tickers) >= 2:
            st.success(f"✓ {len(tickers)} actifs sélectionnés")

        if len(tickers) < 2:
            st.info("Sélectionnez au moins 2 actifs pour continuer.")
            st.stop()

        st.markdown('<div class="sidebar-section">Paramètres d\'Optimisation</div>', unsafe_allow_html=True)

        mu_sel = mu[tickers]
        sigma_sel = sigma.loc[tickers, tickers]
        returns_sel = returns[tickers]

        r_min_min = float(mu_sel.min() * 100)
        r_min_max = float(mu_sel.max() * 100)
        r_min_default = float(mu_sel.mean() * 100)
        r_min_default = max(r_min_min, min(r_min_max, r_min_default))

        r_min = st.slider(
            "Rendement minimum r_min (%)",
            r_min_min, r_min_max,
            r_min_default, 0.5
        ) / 100

        st.markdown('<div class="sidebar-section">Contraintes</div>', unsafe_allow_html=True)

        n_sel = len(tickers)
        if n_sel <= 3:
            max_k = n_sel
        else:
            k_default = min(5, n_sel)
            max_k = st.slider("Cardinalité K (max actifs)", 2, n_sel, k_default)

        c_prop = st.number_input("Coût transaction c (%)", 0.0, 2.0, 0.5, 0.1) / 100

        st.markdown('<div class="sidebar-section">Simulation</div>', unsafe_allow_html=True)
        initial = st.number_input("Investissement Initial (€)", 100, 1000000, 1000, 100)

        with st.expander("📐 Formules"):
            st.markdown("""
            **Objectifs :**
            - f₁ = -w'μ (Rendement)
            - f₂ = w'Σw (Risque)
            - f₃ = c·Σ|wᵢ-wₜ,ᵢ| (Coûts)

            **Contraintes :**
            - Σwᵢ = 1
            - wᵢ ≥ 0
            - Card(w) ≤ K
            """)

    # Initialiser l'optimiseur
    optimizer = PortfolioOptimizer(mu_sel, sigma_sel, transaction_cost=c_prop)

    # =================================
    # ONGLETS PRINCIPAUX
    # =================================
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Frontière Efficiente",
        "🎲 Monte Carlo 3D",
        "📊 Analyse Portefeuille",
        "📚 Documentation"
    ])

    # ─────────────────────────────────
    # ONGLET 1 : FRONTIÈRE EFFICIENTE
    # ─────────────────────────────────
    with tab1:
        render_section("01", "Optimisation Moyenne-Variance Markowitz",
                       "Optimisation bi-objectif : minimiser le risque pour chaque niveau de rendement")

        render_callout(
            "<strong>Objectif :</strong> Trouver les portefeuilles qui minimisent le risque f₂(w) = w'Σw "
            "pour chaque niveau de rendement. La courbe représente l'ensemble des solutions <strong>Pareto-optimales</strong>.",
            "success"
        )

        if st.button("🚀 Calculer la Frontière Efficiente", key="btn_frontier"):
            with st.spinner("Calcul en cours..."):
                st.session_state['frontier_df'] = optimizer.compute_efficient_frontier(60)
                st.success("Frontière calculée !")

        if st.session_state['frontier_df'] is not None and not st.session_state['frontier_df'].empty:
            df_f = st.session_state['frontier_df']

            # Trouver le portefeuille optimal
            valid = df_f[df_f['Return'] >= r_min]
            selected = None

            if not valid.empty:
                best = valid.sort_values('Risk').iloc[0]
                selected = {
                    'Return': best['Return'],
                    'Risk': best['Risk'],
                    'Sharpe': best['Sharpe'],
                    'Weights': best['Weights']
                }
                st.session_state['selected_portfolio'] = selected

            # Afficher le graphique avec espace pour la barre d'outils
            st.markdown('<div style="margin-top: 50px;"></div>', unsafe_allow_html=True)
            fig = create_efficient_frontier_chart(df_f, selected, r_min)
            st.plotly_chart(
                fig,
                use_container_width=True,
                config={
                    'displayModeBar': True,
                    'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                    'displaylogo': False
                }
            )

            # Afficher les infos du portefeuille sélectionné - centré sous le graphique
            if selected:
                st.markdown(f'''
                    <div style="display: flex; justify-content: center; gap: 4rem; margin-top: 2rem; padding: 1.5rem 0;">
                        <div style="text-align: center;">
                            <div style="font-size: 0.875rem; color: #6b6b6b; margin-bottom: 0.25rem;">Rendement Annuel</div>
                            <div style="font-size: 2rem; font-weight: 600; color: #1a1a1a;">{selected['Return']*100:.2f}%</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-size: 0.875rem; color: #6b6b6b; margin-bottom: 0.25rem;">Volatilité</div>
                            <div style="font-size: 2rem; font-weight: 600; color: #1a1a1a;">{selected['Risk']*100:.2f}%</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-size: 0.875rem; color: #6b6b6b; margin-bottom: 0.25rem;">Ratio Sharpe</div>
                            <div style="font-size: 2rem; font-weight: 600; color: #1a1a1a;">{selected['Sharpe']:.3f}</div>
                        </div>
                    </div>
                ''', unsafe_allow_html=True)
            else:
                render_callout("Aucun portefeuille ne respecte r_min. Réduisez le seuil.", "warning")

    # ─────────────────────────────────
    # ONGLET 2 : MONTE CARLO
    # ─────────────────────────────────
    with tab2:
        render_section("02", "Optimisation Tri-Objectif Monte Carlo",
                       f"Avec contrainte de cardinalité K={max_k} et coûts de transaction c={c_prop*100:.2f}%")

        render_callout(
            "<strong>Amélioration :</strong> Ajout de la <strong>contrainte de cardinalité</strong> (max K actifs) "
            f"et des <strong>coûts de transaction</strong> f₃. Résolution par Monte Carlo (5,000 simulations).",
            "success"
        )

        if st.button("🎲 Lancer Simulation Monte Carlo", key="btn_mc"):
            with st.spinner("Génération de 5,000 portefeuilles..."):
                st.session_state['mc_df'] = optimizer.run_monte_carlo(5000, max_k)
                st.success("Simulation terminée !")

        if st.session_state['mc_df'] is not None and not st.session_state['mc_df'].empty:
            df_mc = st.session_state['mc_df']

            # Graphique 3D
            fig_3d = create_monte_carlo_3d_chart(df_mc)
            st.plotly_chart(fig_3d, use_container_width=True, theme=None)

            # Meilleurs portefeuilles
            render_section("", "Meilleurs Portefeuilles", "Respectant r_min, triés par risque")

            valid = df_mc[df_mc['Return'] >= r_min].sort_values(['Risk', 'Cost'])

            if not valid.empty:
                top5 = valid.head(5).copy()
                display = top5[['Return', 'Risk', 'Cost', 'Sharpe', 'N_Assets']].copy()
                display.columns = ['Rendement (%)', 'Risque (%)', 'Coût (%)', 'Sharpe', '# Actifs']
                display['Rendement (%)'] = (display['Rendement (%)'] * 100).round(2)
                display['Risque (%)'] = (display['Risque (%)'] * 100).round(2)
                display['Coût (%)'] = (display['Coût (%)'] * 100).round(3)
                display = display.reset_index(drop=True)
                display.index = display.index + 1

                st.dataframe(display, use_container_width=True)

                if st.button("✅ Sélectionner le Meilleur Portefeuille", type="primary"):
                    best = valid.iloc[0]
                    st.session_state['selected_portfolio'] = {
                        'Return': best['Return'],
                        'Risk': best['Risk'],
                        'Sharpe': best['Sharpe'],
                        'Weights': best['Weights'],
                        'Cost': best['Cost']
                    }
                    st.success("Portefeuille sélectionné ! Allez à l'onglet Analyse.")
            else:
                render_callout("Aucun portefeuille valide. Réduisez r_min ou augmentez K.", "warning")

    # ─────────────────────────────────
    # ONGLET 3 : ANALYSE
    # ─────────────────────────────────
    with tab3:
        render_section("03", "Analyse du Portefeuille", "Backtest historique et répartition")

        if st.session_state['selected_portfolio'] is None:
            render_callout("Veuillez d'abord sélectionner un portefeuille (onglet 1 ou 2).", "warning")
        else:
            port = st.session_state['selected_portfolio']
            weights = port['Weights']

            # Vérifier la synchronisation
            if len(weights) != len(tickers):
                st.error(f"⚠️ Désynchronisation : le portefeuille a {len(weights)} actifs, "
                        f"mais {len(tickers)} sont sélectionnés. Relancez le calcul.")
            else:
                # Métriques
                render_metrics_strip([
                    {'label': 'Rendement Espéré', 'value': f"{port['Return']*100:.2f}%", 'class': 'positive'},
                    {'label': 'Volatilité', 'value': f"{port['Risk']*100:.2f}%"},
                    {'label': 'Ratio Sharpe', 'value': f"{port['Sharpe']:.3f}"},
                    {'label': 'Coût Transaction', 'value': f"{port.get('Cost', 0)*100:.3f}%"}
                ])

                # Backtest
                st.markdown("---")
                render_section("", "Performance Historique", f"Simulation de {initial:,} € investis")

                fig_bt, stats = create_backtest_chart(returns_sel, weights, initial)
                st.plotly_chart(fig_bt, use_container_width=True)

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Valeur Finale", f"{stats['final']:,.0f} €")
                with col2:
                    st.metric("Rendement Total", f"{stats['total_return']:.1f}%")
                with col3:
                    st.metric("Max Drawdown", f"{stats['max_dd']:.1f}%")
                with col4:
                    st.metric("Volatilité Réalisée", f"{stats['volatility']:.1f}%")

                # Allocation
                st.markdown("---")
                render_section("", "Composition du Portefeuille", "Répartition par actif et secteur")

                col1, col2 = st.columns(2)
                with col1:
                    fig_donut = create_allocation_donut(weights, tickers)
                    st.plotly_chart(fig_donut, use_container_width=True)
                with col2:
                    if sector_map:
                        fig_sector = create_sector_bar(weights, tickers, sector_map)
                        st.plotly_chart(fig_sector, use_container_width=True)

                # Tableau détaillé
                st.markdown("---")
                render_section("", "Détails de l'Allocation", "")

                df_det = pd.DataFrame({
                    'Actif': tickers,
                    'Poids (%)': weights * 100,
                    'Rendement (%)': mu_sel.values * 100,
                    'Volatilité (%)': np.sqrt(np.diag(sigma_sel.values)) * 100,
                    'Secteur': [sector_map.get(t, '-') for t in tickers]
                })
                df_det = df_det[df_det['Poids (%)'] > 0.5].sort_values('Poids (%)', ascending=False).round(2)

                st.dataframe(df_det, use_container_width=True, hide_index=True)

    # ─────────────────────────────────
    # ONGLET 4 : DOCUMENTATION
    # ─────────────────────────────────
    with tab4:
        render_section("04", "Cadre Mathématique", "Théorie et référence des formules")

        st.markdown("### Les Trois Objectifs")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
                <div class="formula-block">
                    <div class="formula-label">f₁ : Rendement</div>
                    <div>Objectif : <strong>Maximiser</strong></div>
                </div>
            """, unsafe_allow_html=True)
            st.latex(r"f_1(w) = -w^\top \mu")

        with col2:
            st.markdown(f"""
                <div class="formula-block">
                    <div class="formula-label">f₂ : Risque</div>
                    <div>Objectif : <strong>Minimiser</strong></div>
                </div>
            """, unsafe_allow_html=True)
            st.latex(r"f_2(w) = w^\top \Sigma w")

        with col3:
            st.markdown(f"""
                <div class="formula-block">
                    <div class="formula-label">f₃ : Coûts</div>
                    <div>Objectif : <strong>Minimiser</strong></div>
                </div>
            """, unsafe_allow_html=True)
            st.latex(r"f_3(w) = c \sum_{i=1}^{N} |w_i - w_{t,i}|")

        st.markdown("---")
        st.markdown("### Contraintes")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
                <div class="formula-block">
                    <div class="formula-label">Contraintes de Base</div>
                    <div>Plein investissement, pas de vente à découvert</div>
                </div>
            """, unsafe_allow_html=True)
            st.latex(r"\sum_{i=1}^{N} w_i = 1 \quad \text{et} \quad w_i \geq 0")

        with col2:
            st.markdown("""
                <div class="formula-block">
                    <div class="formula-label">Cardinalité</div>
                    <div>Maximum K actifs non-nuls</div>
                </div>
            """, unsafe_allow_html=True)
            st.latex(r"\sum_{i=1}^{N} \mathbb{1}(w_i > \delta) \leq K")

        st.markdown("---")
        st.markdown("### Concepts Clés")

        render_callout(
            "<strong>Pourquoi Monte Carlo ?</strong><br>"
            "La contrainte de cardinalité rend le problème <strong>non-convexe</strong>. "
            "Monte Carlo explore l'espace par échantillonnage aléatoire.",
            "success"
        )

        render_callout(
            "<strong>Frontière Efficiente ?</strong><br>"
            "Ensemble des portefeuilles <strong>Pareto-optimaux</strong> : aucun ne peut améliorer "
            "un objectif sans dégrader l'autre.",
            "success"
        )

        render_callout(
            "<strong>Limites du Modèle :</strong><br>"
            "• Les rendements passés ne prédisent pas le futur<br>"
            "• Les estimations de μ et Σ sont incertaines<br>"
            "• Les coûts de transaction sont simplifiés",
            "warning"
        )


# =============================================================================
# EXÉCUTION
# =============================================================================

if __name__ == "__main__":
    main()