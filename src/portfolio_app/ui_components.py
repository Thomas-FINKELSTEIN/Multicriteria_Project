import streamlit as st
from .config import COLORS

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
