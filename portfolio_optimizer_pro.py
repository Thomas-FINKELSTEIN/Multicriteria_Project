import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import json
from scipy.optimize import minimize
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION DE LA PAGE
# =============================================================================
st.set_page_config(
    page_title="Portfolio Optimizer Pro",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# SYSTEME DE DESIGN
# =============================================================================
COLORS = {
    "ink": "#1a1a1a",
    "charcoal": "#2d2d2d",
    "slate": "#4a4a4a",
    "graphite": "#6b6b6b",
    "silver": "#9a9a9a",
    "pearl": "#c4c4c4",
    "ivory": "#f5f5f3",
    "paper": "#fafaf8",
    "white": "#ffffff",
    "terminal_orange": "#2d5a3d",
    "success": "#00a67d",
    "warning": "#f0b429",
    "danger": "#e53935",
    "info": "#2196f3",
    "chart_1": "#1e3a5f",
    "chart_2": "#2d5a3d",
    "chart_3": "#00a67d",
    "chart_4": "#7c4dff",
    "chart_5": "#f0b429",
    "chart_6": "#e53935",
    "chart_7": "#26c6da",
    "chart_8": "#ec407a",
}

CHART_PALETTE = [
    COLORS["chart_1"],
    COLORS["chart_2"],
    COLORS["chart_3"],
    COLORS["chart_4"],
    COLORS["chart_5"],
    COLORS["chart_6"],
    COLORS["chart_7"],
    COLORS["chart_8"],
]

# =============================================================================
# CSS PERSONNALISE
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

#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}

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
    st.markdown(
        """
        <div class="masthead" style="text-align: center;">
            <h1 class="masthead-title">Portfolio Optimizer</h1>
            <p class="masthead-subtitle" style="text-align: center; margin-left: auto; margin-right: auto;">
                Optimisation de portefeuille multi-objectifs utilisant le cadre moyenne-variance
                de Markowitz et la simulation Monte Carlo avec contraintes de cardinalite.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section(number: str, title: str, description: str = ""):
    st.markdown(
        f"""
        <div class="section-header">
            <span class="section-number">{number}</span>
            <div class="section-content">
                <h2 class="section-title">{title}</h2>
                <p class="section-desc">{description}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_callout(content: str, style: str = ""):
    st.markdown(f'<div class="callout {style}">{content}</div>', unsafe_allow_html=True)


def render_metrics_strip(metrics: list):
    cols = st.columns(len(metrics))
    for i, m in enumerate(metrics):
        with cols[i]:
            value = m["value"]
            label = m["label"]
            if m.get("class") == "positive":
                color = COLORS["success"]
            elif m.get("class") == "negative":
                color = COLORS["danger"]
            else:
                color = COLORS["ink"]

            st.markdown(
                f"""
                <div style="background: #ffffff; border: 1px solid #c4c4c4; padding: 1.25rem; text-align: center;">
                    <div style="font-family: monospace; font-size: 0.7rem; font-weight: 500; color: #6b6b6b; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.5rem;">{label}</div>
                    <div style="font-family: monospace; font-size: 1.75rem; font-weight: 600; color: {color};">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# =============================================================================
# CHARGEMENT DES DONNÉES
# =============================================================================
@st.cache_data
def load_market_data(data_dir: str = "data", json_path: str = "tick.json"):
    if not os.path.exists(data_dir):
        st.error(f"Dossier de donnees '{data_dir}' introuvable. Lancez python download.py d'abord.")
        st.stop()

    sector_map = {}
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            sectors = json.load(f)
        sector_map = {t: s for s, tickers in sectors.items() for t in tickers}

    files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    if not files:
        st.error("Aucun fichier CSV trouve dans le dossier data")
        st.stop()

    dfs = []
    for f in files:
        path = os.path.join(data_dir, f)
        try:
            with open(path, "r", encoding="utf-8") as file_check:
                first_lines = [file_check.readline() for _ in range(8)]

            skip_rows = 0
            ticker_name = os.path.splitext(f)[0].replace("_", " ")

            for i, line in enumerate(first_lines):
                if line.startswith("Ticker,"):
                    parts = line.strip().split(",")
                    if len(parts) > 1 and parts[1]:
                        ticker_name = parts[1]
                if "Date" in line:
                    skip_rows = i
                    break

            df = pd.read_csv(path, skiprows=skip_rows, index_col=0, parse_dates=True)
            df = df.apply(pd.to_numeric, errors="coerce").dropna(how="all")
            if df.empty:
                continue

            if df.shape[1] > 1:
                for col in df.columns:
                    if not df[col].isna().all():
                        temp_df = df[[col]].dropna()
                        temp_df.columns = [col]
                        dfs.append(temp_df)
            else:
                df.columns = [ticker_name]
                df = df.dropna()
                if not df.empty:
                    dfs.append(df)
        except Exception:
            continue

    if not dfs:
        st.error("Aucune donnee valide n'a pu etre chargee.")
        st.stop()

    prices = pd.concat(dfs, axis=1)
    prices = prices.loc[:, ~prices.columns.duplicated()]
    prices = prices.ffill().bfill().dropna()

    if prices.empty or prices.shape[1] < 2:
        st.error("Pas assez de donnees valides apres nettoyage (besoin d'au moins 2 actifs).")
        st.stop()

    returns = prices.pct_change().dropna()
    mu = returns.mean() * 252
    sigma = returns.cov() * 252

    return prices, returns, mu, sigma, sector_map


# =============================================================================
# OPTIMISEUR
# =============================================================================
def compute_pareto_mask(df: pd.DataFrame, ui_context: bool = False) -> np.ndarray:
    if df is None or df.empty:
        return np.array([], dtype=bool)

    if not df.index.is_unique:
        if ui_context:
            st.warning("Attention : indices dupliques detectes, le calcul Pareto peut etre affecte.")
        return np.zeros(len(df), dtype=bool)

    cols = ["Return", "Risk", "Cost"]
    if not all(c in df.columns for c in cols):
        return np.zeros(len(df), dtype=bool)

    df_obj = df[cols].dropna()
    if df_obj.empty:
        return np.zeros(len(df), dtype=bool)

    vals = df_obj.values
    n = len(vals)
    is_pareto = np.ones(n, dtype=bool)

    objectives = np.column_stack([-vals[:, 0], vals[:, 1], vals[:, 2]])

    for i in range(n):
        if not is_pareto[i]:
            continue
        for j in range(n):
            if i == j or not is_pareto[j]:
                continue
            if (np.all(objectives[j] <= objectives[i]) and np.any(objectives[j] < objectives[i])):
                is_pareto[i] = False
                break

    indexer = df.index.get_indexer(df_obj.index)
    if np.any(indexer < 0):
        return np.zeros(len(df), dtype=bool)

    mask_full = np.zeros(len(df), dtype=bool)
    mask_full[indexer] = is_pareto
    return mask_full


class PortfolioOptimizer:
    def __init__(self, mu: pd.Series, sigma: pd.DataFrame, current_weights=None, transaction_cost: float = 0.005):
        self.mu = mu.values
        self.sigma = sigma.values
        self.tickers = list(mu.index)
        self.n = len(mu)
        self.w_current = current_weights if current_weights is not None else np.zeros(self.n)
        self.c = transaction_cost

    def _apply_constraints(self, w, max_k, delta_tol):
        w = np.asarray(w, dtype=float).copy()

        # Top-K
        if max_k < self.n:
            idx_zero = np.argsort(w)[:-max_k]
            w[idx_zero] = 0.0

        # Seuil
        eff_tol = max(delta_tol, 1e-12)
        w[w < eff_tol] = 0.0

        # Renormalisation
        s = w.sum()
        if s > 0:
            w /= s
        else:
            return np.zeros_like(w)

        return w

    def compute_performance(self, w):
        w = np.asarray(w, dtype=float)
        ret = np.dot(w, self.mu)
        vol = np.sqrt(np.dot(w.T, np.dot(self.sigma, w)))
        return ret, vol

    def compute_transaction_cost(self, w):
        w = np.asarray(w, dtype=float)
        return self.c * np.sum(np.abs(w - self.w_current))

    def compute_efficient_frontier(self, n_points=60):
        results = []
        bounds = tuple((0, 1) for _ in range(self.n))
        init = np.array([1.0 / self.n] * self.n)

        target_returns = np.linspace(self.mu.min(), self.mu.max(), n_points)

        for target in target_returns:
            constraints = [
                {"type": "eq", "fun": lambda w: np.sum(w) - 1},
                {"type": "eq", "fun": lambda w, t=target: np.dot(w, self.mu) - t},
            ]
            try:
                sol = minimize(
                    lambda w: np.sqrt(np.dot(w.T, np.dot(self.sigma, w))),
                    init,
                    method="SLSQP",
                    bounds=bounds,
                    constraints=constraints,
                    options={"maxiter": 1000},
                )
                if sol.success:
                    w = sol.x.copy()
                    w[w < 1e-6] = 0.0
                    if w.sum() <= 0:
                        continue
                    w /= w.sum()
                    ret, vol = self.compute_performance(w)
                    if vol > 0:
                        results.append({"Return": ret, "Risk": vol, "Sharpe": ret / vol, "Weights": w})
            except Exception:
                continue

        return pd.DataFrame(results)

    def run_monte_carlo(self, n_portfolios=5000, max_k=5, delta_tol=0.01):
        results = []
        eff_tol = max(delta_tol, 1e-12)

        for _ in range(n_portfolios):
            w_raw = np.random.random(self.n)
            w = self._apply_constraints(w_raw, max_k, delta_tol)
            if w.sum() == 0:
                continue

            ret, vol = self.compute_performance(w)
            cost = self.compute_transaction_cost(w)

            if vol > 0:
                results.append(
                    {
                        "Method": "Monte Carlo",
                        "Return": ret,
                        "Risk": vol,
                        "Cost": cost,
                        "Sharpe": ret / vol,
                        "Weights": w,
                        "N_Assets": int(np.count_nonzero(w > eff_tol)),
                    }
                )

        return pd.DataFrame(results)

    def optimize_scalarization(self, max_k=5, delta_tol=0.01):
        results = []
        bounds = tuple((0, 1) for _ in range(self.n))
        eff_tol = max(delta_tol, 1e-12)

        lambda_combinations = []
        steps = 5
        for l1 in np.linspace(0.1, 0.8, steps):
            for l2 in np.linspace(0.1, 0.8, steps):
                l3 = 1 - l1 - l2
                if l3 >= 0.05:
                    lambda_combinations.append((l1, l2, l3))

        for (l1, l2, l3) in lambda_combinations:

            def objective(w, l1=l1, l2=l2, l3=l3):
                ret = np.dot(w, self.mu)
                risk = np.sqrt(np.dot(w.T, np.dot(self.sigma, w)))
                cost = self.c * np.sum(np.abs(w - self.w_current))
                return -l1 * ret + l2 * risk + l3 * cost

            constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
            init = np.array([1.0 / self.n] * self.n)

            try:
                sol = minimize(
                    objective,
                    init,
                    method="SLSQP",
                    bounds=bounds,
                    constraints=constraints,
                    options={"maxiter": 1000},
                )
                if sol.success:
                    w_final = self._apply_constraints(sol.x.copy(), max_k, delta_tol)
                    if w_final.sum() == 0:
                        continue
                    ret, vol = self.compute_performance(w_final)
                    cost = self.compute_transaction_cost(w_final)
                    if vol > 0:
                        results.append(
                            {
                                "Return": ret,
                                "Risk": vol,
                                "Cost": cost,
                                "Sharpe": ret / vol,
                                "Weights": w_final,
                                "Lambda_Ret": l1,
                                "Lambda_Risk": l2,
                                "Lambda_Cost": l3,
                                "N_Assets": int(np.count_nonzero(w_final > eff_tol)),
                                "Method": "Scalarisation",
                            }
                        )
            except Exception:
                continue

        return pd.DataFrame(results)

    def optimize_nsga2_simple(self, n_generations=50, pop_size=100, max_k=5, delta_tol=0.01):
        eff_tol = max(delta_tol, 1e-12)

        def dominates(a, b):
            better_in_one = False
            for i in range(3):
                if a[i] > b[i]:
                    return False
                if a[i] < b[i]:
                    better_in_one = True
            return better_in_one

        def evaluate(w):
            ret, vol = self.compute_performance(w)
            cost = self.compute_transaction_cost(w)
            return (-ret, vol, cost)

        def create_individual():
            w = np.random.random(self.n)
            return self._apply_constraints(w, max_k, delta_tol)

        def crossover(p1, p2):
            alpha = np.random.random()
            child = alpha * p1 + (1 - alpha) * p2
            return self._apply_constraints(child, max_k, delta_tol)

        def mutate(w, rate=0.1):
            if np.random.random() < rate:
                w = w + np.random.normal(0, 0.05, self.n)
                w = np.clip(w, 0, 1)
                w = self._apply_constraints(w, max_k, delta_tol)
            return w

        population = [create_individual() for _ in range(pop_size)]
        population = [ind for ind in population if ind.sum() > 0]

        for _ in range(n_generations):
            evaluated = [(i, ind, evaluate(ind)) for i, ind in enumerate(population)]

            fronts = []
            remaining_ids = set(range(len(evaluated)))
            while remaining_ids:
                front_ids = []
                for i in list(remaining_ids):
                    obj_i = evaluated[i][2]
                    dominated_flag = False
                    for j in list(remaining_ids):
                        if i == j:
                            continue
                        obj_j = evaluated[j][2]
                        if dominates(obj_j, obj_i):
                            dominated_flag = True
                            break
                    if not dominated_flag:
                        front_ids.append(i)

                fronts.append(front_ids)
                remaining_ids -= set(front_ids)

            new_pop = []
            for front_ids in fronts:
                if len(new_pop) + len(front_ids) <= pop_size:
                    new_pop.extend([evaluated[i][1] for i in front_ids])
                else:
                    needed = pop_size - len(new_pop)
                    new_pop.extend([evaluated[i][1] for i in front_ids[:needed]])
                    break

            children = []
            while len(children) < pop_size // 2 and len(new_pop) >= 2:
                idx1, idx2 = np.random.choice(len(new_pop), 2, replace=False)
                child = crossover(new_pop[idx1].copy(), new_pop[idx2].copy())
                child = mutate(child)
                if child.sum() > 0:
                    children.append(child)

            population = (new_pop + children)[:pop_size]

        results = []
        for w in population:
            if w.sum() > 0:
                ret, vol = self.compute_performance(w)
                cost = self.compute_transaction_cost(w)
                if vol > 0:
                    results.append(
                        {
                            "Return": ret,
                            "Risk": vol,
                            "Cost": cost,
                            "Sharpe": ret / vol,
                            "Weights": w,
                            "N_Assets": int(np.count_nonzero(w > eff_tol)),
                            "Method": "NSGA-II",
                        }
                    )

        return pd.DataFrame(results)


# =============================================================================
# ROBUSTESSE (BOOTSTRAP)
# =============================================================================
class RobustnessAnalyzer:
    def __init__(self, returns: pd.DataFrame):
        self.returns = returns

    def bootstrap_sample(self):
        n_days = len(self.returns)
        idx = np.random.choice(n_days, size=n_days, replace=True)
        resampled = self.returns.iloc[idx]
        mu_boot = resampled.mean() * 252
        sigma_boot = resampled.cov() * 252
        return mu_boot, sigma_boot

    def run_bootstrap_analysis(self, base_weights: np.ndarray, n_bootstrap=100, confidence=0.95):
        rets, risks, sharpes = [], [], []

        for _ in range(n_bootstrap):
            mu_boot, sigma_boot = self.bootstrap_sample()
            w = np.asarray(base_weights, dtype=float)

            ret = np.dot(w, mu_boot.values)
            vol = np.sqrt(np.dot(w.T, np.dot(sigma_boot.values, w)))
            sharpe = ret / vol if vol > 0 else 0

            rets.append(ret)
            risks.append(vol)
            sharpes.append(sharpe)

        rets = np.asarray(rets)
        risks = np.asarray(risks)
        sharpes = np.asarray(sharpes)

        alpha = (1 - confidence) / 2

        return {
            "return_mean": rets.mean(),
            "return_std": rets.std(),
            "return_ci_low": np.percentile(rets, alpha * 100),
            "return_ci_high": np.percentile(rets, (1 - alpha) * 100),
            "return_worst": np.percentile(rets, 5),
            "risk_mean": risks.mean(),
            "risk_std": risks.std(),
            "risk_ci_low": np.percentile(risks, alpha * 100),
            "risk_ci_high": np.percentile(risks, (1 - alpha) * 100),
            "risk_worst": np.percentile(risks, 95),
            "sharpe_mean": sharpes.mean(),
            "sharpe_std": sharpes.std(),
            "sharpe_ci_low": np.percentile(sharpes, alpha * 100),
            "sharpe_ci_high": np.percentile(sharpes, (1 - alpha) * 100),
            "sharpe_worst": np.percentile(sharpes, 5),
            "returns_distribution": rets,
            "risks_distribution": risks,
            "sharpes_distribution": sharpes,
        }

    def compare_portfolios_robustness(self, portfolios_dict: dict, n_bootstrap=100):
        rows = []
        for name, weights in portfolios_dict.items():
            res = self.run_bootstrap_analysis(weights, n_bootstrap=n_bootstrap)
            rows.append(
                {
                    "Portfolio": name,
                    "Rendement Moyen (%)": res["return_mean"] * 100,
                    "Rendement Worst-Case (%)": res["return_worst"] * 100,
                    "Risque Moyen (%)": res["risk_mean"] * 100,
                    "Risque Worst-Case (%)": res["risk_worst"] * 100,
                    "Sharpe Moyen": res["sharpe_mean"],
                    "Sharpe Worst-Case": res["sharpe_worst"],
                    "Stabilite Rendement": 1 / (res["return_std"] + 0.001),
                }
            )
        return pd.DataFrame(rows)


# =============================================================================
# CHARTS
# =============================================================================
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


# =============================================================================
# APPLICATION
# =============================================================================
def main():
    # --- session_state init
    defaults = {
        "selected_portfolio": None,
        "frontier_df": None,
        "mc_df": None,
        "comparison_df": None,
        "robustness_df": None,
        "robustness_distributions": None,
        "portfolios_to_test": None,
        "universe_key": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    render_masthead()

    # =================================
    # SIDEBAR
    # =================================
    with st.sidebar:
        prices, returns, mu, sigma, sector_map = load_market_data()

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
            "Selectionner les actifs",
            available,
            default=[],
            placeholder="Selectionnez au moins 2 actifs",
        )

        if len(tickers) >= 2:
            st.success(f"{len(tickers)} actifs selectionnes")

        if len(tickers) < 2:
            st.info("Selectionnez au moins 2 actifs pour continuer.")
            st.stop()

        # ✅ FIX 1: reset si l'univers change (evite weights vs tickers desynchronises)
        universe_key = tuple(tickers)
        if st.session_state["universe_key"] is None:
            st.session_state["universe_key"] = universe_key
        elif st.session_state["universe_key"] != universe_key:
            st.session_state["universe_key"] = universe_key
            for k in [
                "selected_portfolio",
                "frontier_df",
                "mc_df",
                "comparison_df",
                "robustness_df",
                "robustness_distributions",
                "portfolios_to_test",
            ]:
                st.session_state[k] = None

        st.markdown('<div class="sidebar-section">Parametres d\'Optimisation</div>', unsafe_allow_html=True)

        mu_sel = mu[tickers]
        sigma_sel = sigma.loc[tickers, tickers]
        returns_sel = returns[tickers]

        r_min_min = float(mu_sel.min() * 100)
        r_min_max = float(mu_sel.max() * 100)
        r_min_default = float(mu_sel.mean() * 100)
        r_min_default = max(r_min_min, min(r_min_max, r_min_default))

        r_min = st.slider("Rendement minimum r_min (%)", r_min_min, r_min_max, r_min_default, 0.5) / 100

        st.markdown('<div class="sidebar-section">Contraintes (Niveaux 2 & 3)</div>', unsafe_allow_html=True)

        n_sel = len(tickers)

        # ✅ FIX 2: eviter slider min==max quand n_sel==2 (RangeError JS)
        if n_sel <= 2:
            max_k = n_sel
            st.markdown("**Cardinalite K (max actifs)**")
            st.info(f"K fixe : {max_k} (car seulement {n_sel} actifs selectionnes)")
        else:
            k_default = min(5, n_sel)
            max_k = st.slider("Cardinalite K (max actifs)", 2, n_sel, k_default)
            st.caption(f"K effectif : {max_k} / {n_sel} actifs")

        delta_tol = st.slider("Seuil poids minimum delta_tol (%)", 0.5, 5.0, 1.0, 0.5) / 100
        st.caption("Avec delta, le portefeuille final peut avoir < K actifs.")

        c_prop = st.number_input("Cout transaction c (%)", 0.0, 2.0, 0.5, 0.1) / 100

        st.markdown('<div class="sidebar-section">Simulation</div>', unsafe_allow_html=True)
        initial = st.number_input("Investissement Initial (EUR)", 100, 1_000_000, 1000, 100)

        seed = st.number_input("Seed (optionnel, 0 = aleatoire)", min_value=0, value=0, step=1)
        if seed != 0:
            np.random.seed(seed)

    optimizer = PortfolioOptimizer(mu_sel, sigma_sel, transaction_cost=c_prop)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Frontiere Efficiente", "Monte Carlo", "Analyse Portefeuille", "Robustesse & Comparaison", "Documentation"]
    )

    # ────────────────────────────────
    # TAB 1
    # ────────────────────────────────
    with tab1:
        render_section(
            "01",
            "Optimisation Moyenne-Variance Markowitz",
            "Optimisation bi-objectif : minimiser le risque pour chaque niveau de rendement",
        )

        render_callout(
            "<strong>Objectif :</strong> Trouver les portefeuilles qui minimisent le risque f2(w) = w'Sw "
            "pour chaque niveau de rendement. La courbe represente l'ensemble des solutions <strong>Pareto-optimales</strong>. "
            "<strong>Note :</strong> Ce niveau est pur (sans contraintes K ou delta).",
            "success",
        )

        if st.button("Calculer la Frontiere Efficiente", key="btn_frontier"):
            with st.spinner("Calcul en cours..."):
                st.session_state["frontier_df"] = optimizer.compute_efficient_frontier(60)
                st.success("Frontiere calculee !")

        if st.session_state["frontier_df"] is not None and not st.session_state["frontier_df"].empty:
            df_f = st.session_state["frontier_df"]
            valid = df_f[df_f["Return"] >= r_min]
            selected = None

            if not valid.empty:
                best = valid.sort_values("Risk").iloc[0]
                selected = {
                    "Return": best["Return"],
                    "Risk": best["Risk"],
                    "Sharpe": best["Sharpe"],
                    "Weights": best["Weights"],
                }
                st.session_state["selected_portfolio"] = selected

            st.markdown('<div style="margin-top: 50px;"></div>', unsafe_allow_html=True)
            fig = create_efficient_frontier_chart(df_f, selected, r_min)
            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displayModeBar": True, "modeBarButtonsToRemove": ["lasso2d", "select2d"], "displaylogo": False},
            )

            if selected:
                st.markdown(
                    f"""
                    <div style="display: flex; justify-content: center; gap: 4rem; margin-top: 2rem; padding: 1.5rem 0;">
                        <div style="text-align: center;">
                            <div style="font-size: 0.875rem; color: #6b6b6b; margin-bottom: 0.25rem;">Rendement Annuel</div>
                            <div style="font-size: 2rem; font-weight: 600; color: #1a1a1a;">{selected['Return']*100:.2f}%</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-size: 0.875rem; color: #6b6b6b; margin-bottom: 0.25rem;">Volatilite</div>
                            <div style="font-size: 2rem; font-weight: 600; color: #1a1a1a;">{selected['Risk']*100:.2f}%</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-size: 0.875rem; color: #6b6b6b; margin-bottom: 0.25rem;">Ratio Sharpe</div>
                            <div style="font-size: 2rem; font-weight: 600; color: #1a1a1a;">{selected['Sharpe']:.3f}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                render_callout("Aucun portefeuille ne respecte r_min. Reduisez le seuil.", "warning")

    # ────────────────────────────────
    # TAB 2
    # ────────────────────────────────
    with tab2:
        render_section("02", "Optimisation Tri-Objectif Monte Carlo", f"Avec contrainte de cardinalite K={max_k}")

        mode_reequilibrage = st.toggle("Vous possedez deja un portefeuille ?", value=False)

        if mode_reequilibrage:
            render_callout(
                "<strong>Mode Reequilibrage :</strong> Definissez votre portefeuille actuel ci-dessous. "
                "Les couts de transaction varieront selon l'ecart entre votre position actuelle et les nouveaux portefeuilles.",
                "success",
            )

            st.markdown("### Votre portefeuille actuel")
            st.caption("Definissez le pourcentage de chaque actif (total = 100%)")

            w_current = []
            cols = st.columns(min(4, len(tickers)))
            for i, ticker in enumerate(tickers):
                with cols[i % 4]:
                    val = st.number_input(
                        f"{ticker}",
                        min_value=0.0,
                        max_value=100.0,
                        value=0.0,
                        step=5.0,
                        key=f"w_current_{ticker}",
                    )
                    w_current.append(val / 100)

            w_current = np.array(w_current, dtype=float)
            total_current = w_current.sum() * 100

            if abs(total_current - 100) < 0.1:
                st.success(f"Total : {total_current:.1f}%")
            elif total_current > 1e-6:
                st.warning(f"Total : {total_current:.1f}% (devrait etre 100%)")
            else:
                st.info("Considere comme portefeuille vide (nouveau client)")

            if st.button("Lancer Simulation Monte Carlo 3D", key="btn_mc_3d"):
                with st.spinner("Generation de 5,000 portefeuilles..."):
                    s = w_current.sum()
                    w_curr_norm = (w_current / s) if s > 1e-6 else np.zeros_like(w_current)
                    optimizer_rebal = PortfolioOptimizer(mu_sel, sigma_sel, w_curr_norm, c_prop)
                    st.session_state["mc_df"] = optimizer_rebal.run_monte_carlo(5000, max_k, delta_tol=delta_tol)
                    st.success("Simulation terminee !")

            if st.session_state["mc_df"] is not None and not st.session_state["mc_df"].empty:
                df_mc = st.session_state["mc_df"]
                cost_std = df_mc["Cost"].std()

                if cost_std > 0.0001:
                    plot_df = pd.DataFrame(
                        {
                            "Rendement": df_mc["Return"].values * 100,
                            "Risque": df_mc["Risk"].values * 100,
                            "Cout": df_mc["Cost"].values * 100,
                            "Sharpe": df_mc["Sharpe"].values,
                        }
                    ).dropna()

                    plot_df["Qualite"] = pd.cut(
                        plot_df["Sharpe"],
                        bins=[-np.inf, 0.5, 0.8, 1.1, np.inf],
                        labels=["Faible", "Moyen", "Bon", "Excellent"],
                    )

                    fig_3d = px.scatter_3d(
                        plot_df, x="Rendement", y="Risque", z="Cout", color="Qualite", hover_data=["Sharpe"], opacity=0.8
                    )
                    fig_3d.update_traces(marker=dict(size=5))
                    fig_3d.update_layout(
                        height=650,
                        margin=dict(l=0, r=0, b=0, t=30),
                        scene=dict(
                            xaxis_title="Rendement (%)",
                            yaxis_title="Risque (%)",
                            zaxis_title="Cout de Transaction (%)",
                        ),
                        legend=dict(
                            title=dict(text="Qualite", font=dict(size=18), side="top center"),
                            font=dict(size=16),
                            itemsizing="constant",
                            itemwidth=50,
                            yanchor="top",
                            y=0.95,
                            xanchor="right",
                            x=0.99,
                            bgcolor="rgba(255,255,255,0.9)",
                            bordercolor="#ccc",
                            borderwidth=1,
                        ),
                    )

                    st.markdown('<div style="margin-top: 50px;"></div>', unsafe_allow_html=True)
                    st.plotly_chart(
                        fig_3d,
                        use_container_width=True,
                        config={"displayModeBar": True, "modeBarButtonsToRemove": ["lasso2d", "select2d"], "displaylogo": False},
                    )
                else:
                    st.warning("Les couts ne varient pas. Definissez un portefeuille actuel non-nul.")

                render_section("", "Meilleurs Portefeuilles", "Respectant r_min, tries par risque puis cout")

                valid = df_mc[df_mc["Return"] >= r_min].sort_values(["Risk", "Cost"])

                if not valid.empty:
                    top5 = valid.head(5).copy()
                    display = top5[["Return", "Risk", "Cost", "Sharpe", "N_Assets"]].copy()
                    display.columns = ["Rendement (%)", "Risque (%)", "Cout (%)", "Sharpe", "# Actifs"]
                    display["Rendement (%)"] = (display["Rendement (%)"] * 100).round(2)
                    display["Risque (%)"] = (display["Risque (%)"] * 100).round(2)
                    display["Cout (%)"] = (display["Cout (%)"] * 100).round(3)
                    display = display.reset_index(drop=True)
                    display.index = display.index + 1

                    st.dataframe(display, use_container_width=True)

                    if st.button("Selectionner le Meilleur Portefeuille", type="primary", key="btn_select_3d"):
                        best = valid.iloc[0]
                        st.session_state["selected_portfolio"] = {
                            "Return": best["Return"],
                            "Risk": best["Risk"],
                            "Sharpe": best["Sharpe"],
                            "Weights": best["Weights"],
                            "Cost": best["Cost"],
                        }
                        st.success("Portefeuille selectionne ! Allez a l'onglet Analyse.")
                else:
                    render_callout("Aucun portefeuille valide. Reduisez r_min ou augmentez K.", "warning")

        else:
            render_callout(
                "<strong>Mode Nouvel Investisseur :</strong> Simulation pour un investisseur partant de zero. "
                f"Contrainte de cardinalite K={max_k} et cout de transaction fixe c={c_prop * 100:.2f}%.",
                "success",
            )

            if st.button("Lancer Simulation Monte Carlo", key="btn_mc_2d"):
                with st.spinner("Generation de 5,000 portefeuilles..."):
                    st.session_state["mc_df"] = optimizer.run_monte_carlo(5000, max_k, delta_tol=delta_tol)
                    st.success("Simulation terminee !")

            if st.session_state["mc_df"] is not None and not st.session_state["mc_df"].empty:
                df_mc = st.session_state["mc_df"]

                plot_df = pd.DataFrame(
                    {
                        "Rendement": df_mc["Return"].values * 100,
                        "Risque": df_mc["Risk"].values * 100,
                        "Sharpe": df_mc["Sharpe"].values,
                        "N_Assets": df_mc["N_Assets"].values,
                    }
                ).dropna()

                plot_df["Qualite"] = pd.cut(
                    plot_df["Sharpe"], bins=[-np.inf, 0.5, 0.8, 1.1, np.inf], labels=["Faible", "Moyen", "Bon", "Excellent"]
                )

                fig_2d = px.scatter(
                    plot_df,
                    x="Risque",
                    y="Rendement",
                    color="Qualite",
                    hover_data=["Sharpe", "N_Assets"],
                    opacity=0.7,
                )
                fig_2d.update_traces(marker=dict(size=10, line=dict(width=0)))
                fig_2d.add_hline(y=r_min * 100, line=dict(color=COLORS["terminal_orange"], width=2, dash="dash"))
                fig_2d.add_annotation(
                    x=0,
                    y=r_min * 100,
                    xref="paper",
                    yref="y",
                    text=f"<b>{r_min * 100:.1f} %</b>",
                    font=dict(size=17, color=COLORS["terminal_orange"], family="DM Mono"),
                    showarrow=False,
                    xanchor="right",
                    xshift=-10,
                )
                fig_2d.update_layout(
                    height=700,
                    margin=dict(l=100, r=50, b=70, t=60),
                    plot_bgcolor=COLORS["white"],
                    paper_bgcolor="rgba(0,0,0,0)",
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
                    legend=dict(
                        title=dict(text="Qualite", font=dict(size=18), side="top center"),
                        font=dict(size=16),
                        itemsizing="constant",
                        itemwidth=50,
                        yanchor="top",
                        y=0.95,
                        xanchor="right",
                        x=0.99,
                        bgcolor="rgba(255,255,255,0.9)",
                        bordercolor="#ccc",
                        borderwidth=1,
                    ),
                )

                st.markdown('<div style="margin-top: 50px;"></div>', unsafe_allow_html=True)
                st.plotly_chart(
                    fig_2d,
                    use_container_width=True,
                    config={"displayModeBar": True, "modeBarButtonsToRemove": ["lasso2d", "select2d"], "displaylogo": False},
                )

                st.info(f"Cout de transaction fixe : {c_prop * 100:.2f}% (nouvel investisseur)")

                render_section("", "Meilleurs Portefeuilles", "Respectant r_min, tries par risque")

                valid = df_mc[df_mc["Return"] >= r_min].sort_values("Risk")

                if not valid.empty:
                    top5 = valid.head(5).copy()
                    display = top5[["Return", "Risk", "Sharpe", "N_Assets"]].copy()
                    display.columns = ["Rendement (%)", "Risque (%)", "Sharpe", "# Actifs"]
                    display["Rendement (%)"] = (display["Rendement (%)"] * 100).round(2)
                    display["Risque (%)"] = (display["Risque (%)"] * 100).round(2)
                    display = display.reset_index(drop=True)
                    display.index = display.index + 1

                    st.dataframe(display, use_container_width=True)

                    if st.button("Selectionner le Meilleur Portefeuille", type="primary", key="btn_select_2d"):
                        best = valid.iloc[0]
                        st.session_state["selected_portfolio"] = {
                            "Return": best["Return"],
                            "Risk": best["Risk"],
                            "Sharpe": best["Sharpe"],
                            "Weights": best["Weights"],
                            "Cost": best["Cost"],
                        }
                        st.success("Portefeuille selectionne ! Allez a l'onglet Analyse.")
                else:
                    render_callout("Aucun portefeuille valide. Reduisez r_min ou augmentez K.", "warning")

    # ────────────────────────────────
    # TAB 3
    # ────────────────────────────────
    with tab3:
        render_section("03", "Analyse du Portefeuille", "Backtest historique et repartition")

        if st.session_state["selected_portfolio"] is None:
            render_callout("Veuillez d'abord selectionner un portefeuille (onglet 1 ou 2).", "warning")
        else:
            port = st.session_state["selected_portfolio"]
            weights = np.asarray(port["Weights"], dtype=float)

            # ✅ FIX 3: stop net si dimension invalide
            if len(weights) != len(tickers):
                st.error(
                    "Portefeuille invalide: dimension des poids != nombre d'actifs selectionnes. "
                    "Change d'univers => relance les calculs."
                )
                st.stop()

            eff_tol = max(delta_tol, 1e-12)
            n_assets_eff = int(np.count_nonzero(weights > eff_tol))

            render_metrics_strip(
                [
                    {"label": "Rendement Espere", "value": f"{port['Return']*100:.2f}%", "class": "positive"},
                    {"label": "Volatilite", "value": f"{port['Risk']*100:.2f}%"},
                    {"label": "Ratio Sharpe", "value": f"{port['Sharpe']:.3f}"},
                    {"label": "# Actifs (w > delta)", "value": f"{n_assets_eff}"},
                ]
            )

            st.markdown("---")
            render_section("", "Performance Historique", f"Simulation de {initial:,} EUR investis")

            fig_bt, stats = create_backtest_chart(returns_sel, weights, initial)
            st.plotly_chart(fig_bt, use_container_width=True)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Valeur Finale", f"{stats['final']:,.0f} EUR")
            with col2:
                st.metric("Rendement Total", f"{stats['total_return']:.1f}%")
            with col3:
                st.metric("Max Drawdown", f"{stats['max_dd']:.1f}%")
            with col4:
                st.metric("Volatilite Realisee", f"{stats['volatility']:.1f}%")

            st.markdown("---")
            render_section("", "Composition du Portefeuille", "Repartition par actif et secteur")

            col1, col2 = st.columns(2)
            with col1:
                fig_donut = create_allocation_donut(weights, tickers, min_pct=delta_tol * 100)
                st.plotly_chart(fig_donut, use_container_width=True)
            with col2:
                if sector_map:
                    fig_sector = create_sector_bar(weights, tickers, sector_map)
                    st.plotly_chart(fig_sector, use_container_width=True)

            st.markdown("---")
            render_section("", "Details de l'Allocation", f"Filtres a {delta_tol*100:.1f}%")

            df_det = pd.DataFrame(
                {
                    "Actif": tickers,
                    "Poids (%)": weights * 100,
                    "Rendement (%)": mu_sel.values * 100,
                    "Volatilite (%)": np.sqrt(np.diag(sigma_sel.values)) * 100,
                    "Secteur": [sector_map.get(t, "-") for t in tickers],
                }
            )
            df_det = df_det[df_det["Poids (%)"] >= delta_tol * 100].sort_values("Poids (%)", ascending=False).round(2)
            st.dataframe(df_det, use_container_width=True, hide_index=True)

    # ────────────────────────────────
    # TAB 4
    # ────────────────────────────────
    with tab4:
        render_section(
            "04",
            "Analyse de Robustesse & Comparaison des Methodes",
            "Niveau 2 & 3 : Methodes alternatives et stabilite des portefeuilles",
        )

        st.markdown("### Comparaison des Methodes d'Optimisation")

        render_callout(
            "<strong>3 methodes tri-objectif comparees :</strong><br>"
            "• <strong>Monte Carlo</strong> : Echantillonnage aleatoire (3000 portefeuilles)<br>"
            "• <strong>Scalarisation</strong> : Ponderation des objectifs F = lambda1(-R) + lambda2(sigma) + lambda3(C)<br>"
            "• <strong>NSGA-II</strong> : Algorithme evolutionnaire multi-objectif",
            "success",
        )

        col1, col2 = st.columns(2)
        with col1:
            n_gen = st.slider("Generations NSGA-II", 20, 100, 50)
        with col2:
            n_bootstrap = st.slider("Echantillons Bootstrap", 50, 500, 100)

        if st.button("Lancer la Comparaison Complete", type="primary", key="btn_comparison"):
            with st.spinner("Optimisation Monte Carlo..."):
                df_mc = optimizer.run_monte_carlo(3000, max_k, delta_tol=delta_tol)
            with st.spinner("Optimisation par Scalarisation..."):
                df_scalar = optimizer.optimize_scalarization(max_k=max_k, delta_tol=delta_tol)
            with st.spinner("Optimisation NSGA-II..."):
                df_nsga = optimizer.optimize_nsga2_simple(n_generations=n_gen, max_k=max_k, delta_tol=delta_tol)

            df_all = pd.concat([df_mc, df_scalar, df_nsga], ignore_index=True)
            df_all["Pareto"] = compute_pareto_mask(df_all, ui_context=True)
            st.session_state["comparison_df"] = df_all
            st.success("Comparaison terminee !")

        if st.session_state["comparison_df"] is not None and not st.session_state["comparison_df"].empty:
            df_all = st.session_state["comparison_df"]

            st.markdown("### Comparaison Visuelle des Methodes")
            show_pareto = st.checkbox("Afficher le Front de Pareto", value=False)

            plot_df = pd.DataFrame(
                {
                    "Rendement": df_all["Return"].values * 100,
                    "Risque": df_all["Risk"].values * 100,
                    "Sharpe": df_all["Sharpe"].values,
                    "Methode": df_all["Method"].values,
                }
            ).dropna()

            fig_comp = px.scatter(plot_df, x="Risque", y="Rendement", color="Methode", hover_data=["Sharpe"], opacity=0.7)
            fig_comp.update_traces(marker=dict(size=8, line=dict(width=0)))

            if show_pareto:
                pareto_df = df_all[df_all["Pareto"]]
                if not pareto_df.empty:
                    fig_comp.add_trace(
                        go.Scatter(
                            x=pareto_df["Risk"] * 100,
                            y=pareto_df["Return"] * 100,
                            mode="markers",
                            name="Front Pareto",
                            marker=dict(size=12, color="rgba(0,0,0,0)", line=dict(color=COLORS["warning"], width=2), symbol="circle-open"),
                            hovertemplate="<b>Pareto</b><br>Rendement: %{y:.2f}%<br>Risque: %{x:.2f}%<extra></extra>",
                        )
                    )

            fig_comp.add_hline(y=r_min * 100, line=dict(color=COLORS["terminal_orange"], width=2, dash="dash"))
            fig_comp.update_layout(
                height=600,
                margin=dict(l=100, r=50, b=70, t=60),
                plot_bgcolor=COLORS["white"],
                xaxis=dict(title=dict(text="Volatilite (%)", font=dict(size=18), standoff=30), gridcolor=COLORS["ivory"], tickfont=dict(size=14)),
                yaxis=dict(title=dict(text="Rendement (%)", font=dict(size=18), standoff=30), gridcolor=COLORS["ivory"], tickfont=dict(size=14)),
                legend=dict(title=dict(text="Methode", font=dict(size=16)), font=dict(size=14), bgcolor="rgba(255,255,255,0.9)", bordercolor="#ccc", borderwidth=1),
            )
            st.plotly_chart(fig_comp, use_container_width=True)
            st.caption("Note : La comparaison applique les couts pour un 'Nouvel Investisseur' (portefeuille initial vide).")

            st.markdown("### Meilleur Portefeuille par Methode")
            best_per_method = []
            for method in df_all["Method"].unique():
                df_method = df_all[df_all["Method"] == method]
                valid = df_method[df_method["Return"] >= r_min]
                if not valid.empty:
                    best = valid.sort_values("Risk").iloc[0]
                    best_per_method.append(
                        {
                            "Methode": method,
                            "Rendement (%)": round(best["Return"] * 100, 2),
                            "Risque (%)": round(best["Risk"] * 100, 2),
                            "Cout (%)": round(best["Cost"] * 100, 3),
                            "Sharpe": round(best["Sharpe"], 3),
                            "# Actifs": int(best["N_Assets"]) if "N_Assets" in best else "-",
                        }
                    )
            if best_per_method:
                st.dataframe(pd.DataFrame(best_per_method), use_container_width=True, hide_index=True)

            st.markdown("---")
            st.markdown("### Analyse de Robustesse (Bootstrap)")
            render_callout(
                "<strong>Bootstrap :</strong> Reechantillonnage des donnees historiques pour tester "
                "la stabilite du portefeuille face a l'incertitude des estimations (mu, Sigma).",
                "success",
            )

            if st.button("Analyser la Robustesse des Meilleurs Portefeuilles", key="btn_robustness"):
                with st.spinner(f"Analyse bootstrap ({n_bootstrap} echantillons)..."):
                    portfolios_to_test = {}
                    for method in df_all["Method"].unique():
                        df_method = df_all[df_all["Method"] == method]
                        valid = df_method[df_method["Return"] >= r_min]
                        if not valid.empty:
                            best = valid.sort_values("Risk").iloc[0]
                            portfolios_to_test[method] = best["Weights"]

                    st.session_state["portfolios_to_test"] = portfolios_to_test

                    analyzer = RobustnessAnalyzer(returns_sel)
                    df_rob = analyzer.compare_portfolios_robustness(portfolios_to_test, n_bootstrap=n_bootstrap)
                    st.session_state["robustness_df"] = df_rob

                    distributions = {}
                    for name, w in portfolios_to_test.items():
                        distributions[name] = analyzer.run_bootstrap_analysis(w, n_bootstrap=n_bootstrap)
                    st.session_state["robustness_distributions"] = distributions

                st.success("Analyse de robustesse terminee !")

        if st.session_state["robustness_df"] is not None:
            df_rob = st.session_state["robustness_df"]
            st.markdown("### Resultats de Robustesse")
            st.dataframe(df_rob.round(3), use_container_width=True, hide_index=True)

            if st.session_state["robustness_distributions"] is not None:
                distributions = st.session_state["robustness_distributions"]
                st.markdown("### Distribution des Rendements (Bootstrap)")

                hist_data = []
                for name, dist in distributions.items():
                    for ret in dist["returns_distribution"]:
                        hist_data.append({"Methode": name, "Rendement (%)": ret * 100})
                df_hist = pd.DataFrame(hist_data)

                fig_hist = px.histogram(df_hist, x="Rendement (%)", color="Methode", barmode="overlay", opacity=0.6, nbins=30)
                fig_hist.update_layout(height=400, margin=dict(l=50, r=50, b=50, t=30), plot_bgcolor=COLORS["white"])
                st.plotly_chart(fig_hist, use_container_width=True)

                st.markdown("### Interpretation")
                most_stable = df_rob.loc[df_rob["Stabilite Rendement"].idxmax(), "Portfolio"]
                best_worst_case = df_rob.loc[df_rob["Rendement Worst-Case (%)"].idxmax(), "Portfolio"]

                render_callout(
                    f"<strong>Portefeuille le plus stable :</strong> {most_stable}<br>"
                    f"<strong>Meilleur worst-case :</strong> {best_worst_case}<br><br>"
                    "Un portefeuille robuste maintient ses performances meme quand les conditions "
                    "de marche different des donnees historiques utilisees pour l'optimisation.",
                    "success",
                )

                st.markdown("### Selection Automatique du Portefeuille Robuste")

                df_valid = df_rob[df_rob["Rendement Moyen (%)"] >= r_min * 100]
                if not df_valid.empty:
                    df_sorted = df_valid.sort_values(by=["Rendement Worst-Case (%)", "Risque Worst-Case (%)"], ascending=[False, True])
                    winner = df_sorted.iloc[0]["Portfolio"]

                    if st.session_state["portfolios_to_test"] and winner in st.session_state["portfolios_to_test"]:
                        exact_weights = np.asarray(st.session_state["portfolios_to_test"][winner], dtype=float)

                        # ✅ FIX 4: recheck dimension avant dot
                        if len(exact_weights) != len(tickers):
                            st.error("Poids du portefeuille robustesse incompatibles avec l'univers courant. Relance la comparaison.")
                            st.stop()

                        optimizer_new = PortfolioOptimizer(mu_sel, sigma_sel, transaction_cost=c_prop)
                        ret, vol = optimizer_new.compute_performance(exact_weights)
                        cost = optimizer_new.compute_transaction_cost(exact_weights)

                        st.session_state["selected_portfolio"] = {
                            "Return": ret,
                            "Risk": vol,
                            "Sharpe": ret / vol if vol > 0 else 0,
                            "Weights": exact_weights,
                            "Cost": cost,
                        }

                        render_callout(
                            f"<strong>Regle de selection :</strong><br>"
                            f"1. Filtrer les methodes avec rendement moyen >= r_min ({r_min * 100:.1f}%)<br>"
                            f"2. Maximiser le rendement worst-case (5eme percentile)<br>"
                            f"3. En cas d'egalite, minimiser le risque worst-case<br><br>"
                            f"<strong>-> Methode selectionnee : {winner}</strong><br>"
                            f"Le portefeuille a ete mis a jour dans l'onglet Analyse.<br>"
                            f"<em>Note : Les couts affiches ici sont calcules pour un nouvel investisseur.</em>",
                            "success",
                        )
                    else:
                        st.error("Erreur de recuperation des poids.")
                else:
                    render_callout(f"Aucune methode ne respecte r_min = {r_min * 100:.1f}%. Reduisez le seuil de rendement minimum.", "warning")

    # ────────────────────────────────
    # TAB 5
    # ────────────────────────────────
    with tab5:
        render_section("05", "Cadre Mathematique", "Theorie et reference des formules")

        st.markdown("### Les Trois Objectifs")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(
                """
                <div class="formula-block">
                    <div class="formula-label">f1 : Rendement</div>
                    <div>Objectif : <strong>Maximiser</strong></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.latex(r"f_1(w) = -w^\top \mu")

        with col2:
            st.markdown(
                """
                <div class="formula-block">
                    <div class="formula-label">f2 : Risque</div>
                    <div>Objectif : <strong>Minimiser</strong></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.latex(r"f_2(w) = w^\top \Sigma w")

        with col3:
            st.markdown(
                """
                <div class="formula-block">
                    <div class="formula-label">f3 : Couts</div>
                    <div>Objectif : <strong>Minimiser</strong></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.latex(r"f_3(w) = c \sum_{i=1}^{N} |w_i - w_{t,i}|")

        st.markdown("---")
        st.markdown("### Contraintes")
        st.caption("Application des contraintes : Top-K -> seuil delta -> renormalisation.")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                """
                <div class="formula-block">
                    <div class="formula-label">Contraintes de Base</div>
                    <div>Plein investissement, pas de vente a decouvert</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.latex(r"\sum_{i=1}^{N} w_i = 1 \quad \text{et} \quad w_i \geq 0")

        with col2:
            st.markdown(
                """
                <div class="formula-block">
                    <div class="formula-label">Cardinalite et Seuil</div>
                    <div>Max K actifs, seuil minimum delta_tol</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.latex(r"\sum_{i=1}^{N} \mathbb{1}(w_i \ge \delta_{tol}) \leq K")
            st.latex(r"w_i \in \{0\} \cup [\delta_{tol}, 1]")

        st.markdown("---")
        st.markdown("### Methodes d'Optimisation")

        render_callout(
            "<strong>Monte Carlo :</strong><br>"
            "Generation aleatoire de portefeuilles respectant les contraintes. "
            "Simple mais exploration non-dirigee de l'espace des solutions.",
            "success",
        )

        render_callout(
            "<strong>Scalarisation ponderee :</strong><br>"
            "Transformation du probleme multi-objectif en mono-objectif : "
            "F(w) = lambda1f1(w) + lambda2f2(w) + lambda3f3(w). Variation des lambda pour explorer le front de Pareto.",
            "success",
        )

        render_callout(
            "<strong>NSGA-II :</strong><br>"
            "Algorithme evolutionnaire utilisant la dominance de Pareto "
            "pour maintenir une population diversifiee de solutions non-dominees.",
            "success",
        )

        st.markdown("---")
        st.markdown("### Analyse de Robustesse")

        render_callout(
            "<strong>Bootstrap :</strong><br>"
            "Reechantillonnage avec remise des rendements historiques pour tester "
            "la stabilite du portefeuille face a l'incertitude des estimations (mu, Sigma).",
            "success",
        )

        render_callout(
            "<strong>Limites du Modele :</strong><br>"
            "• Les rendements passes ne predisent pas le futur<br>"
            "• Les estimations de mu et Sigma sont incertaines<br>"
            "• Les couts de transaction sont simplifies<br>"
            "• Les correlations peuvent changer en periode de crise",
            "warning",
        )


if __name__ == "__main__":
    main()
