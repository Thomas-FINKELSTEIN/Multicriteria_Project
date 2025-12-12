from .config import COLORS

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
