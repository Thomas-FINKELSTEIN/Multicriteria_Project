import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from .styles import CUSTOM_CSS
from .config import COLORS
from .ui_components import render_masthead, render_section, render_callout, render_metrics_strip
from .data_loader import load_market_data_impl
from .optimizer import PortfolioOptimizer, compute_pareto_mask
from .robustness import RobustnessAnalyzer
from .charts import (
    create_efficient_frontier_chart,
    create_sector_bar,
    create_backtest_chart,
    create_allocation_donut,
)

def main():
    # =============================================================================
    # CONFIGURATION DE LA PAGE (DOIT ETRE AVANT TOUT OUTPUT STREAMLIT)
    # =============================================================================
    st.set_page_config(
        page_title="Portfolio Optimizer Pro",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Cache Streamlit défini ici (après set_page_config)
    @st.cache_data
    def load_market_data(data_dir: str = "data", json_path: str = "tick.json"):
        return load_market_data_impl(data_dir=data_dir, json_path=json_path)

    # =============================================================================
    # APPLICATION
    # =============================================================================
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

        # reset si l'univers change (evite weights vs tickers desynchronises)
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

        # eviter slider min==max quand n_sel==2 (RangeError JS)
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
                    import plotly.graph_objects as go
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
