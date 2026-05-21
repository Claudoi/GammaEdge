# app/Home.py
from __future__ import annotations

import inspect
import sys
from pathlib import Path
from typing import Any

# Add project root to sys.path to allow imports from 'app'
root_path = str(Path(__file__).parent.parent)
if root_path not in sys.path:
    sys.path.append(root_path)

import streamlit as st  # noqa: E402  # Import after sys.path setup

# Design System
from app.design_system import COLORS, get_global_styles  # noqa: E402

# ============================================================================
# Plotly compatibility monkey-patch
# ============================================================================

_ORIG_PLOTLY_CHART = st.plotly_chart

# Deprecated config keys that Streamlit no longer accepts as direct kwargs
_DEPRECATED_CFG_KEYS = {
    "displayModeBar",
    "displaylogo",
    "modeBarButtons",
    "staticPlot",
    "scrollZoom",
    "editable",
}

# To avoid spamming logs: remember which caller locations we already notified
_notified_callers: set[tuple[str, tuple[str, ...], bool]] = set()


def _plotly_chart_compat(fig: Any, *args: Any, **kwargs: Any):
    """
    Backwards-compatible wrapper around st.plotly_chart.

    - If a positional dict is passed after `fig`, it is treated as `config`.
    - Deprecated config kwargs are migrated into a `config` dict.
    - Shows a one-time info message per caller location when migration happens.
    """
    # 1) Positional config dict
    cfg_from_pos: dict[str, Any] = {}
    if len(args) >= 1 and isinstance(args[0], dict):
        cfg_from_pos = dict(args[0])

    # 2) Merge with explicit config kwarg
    cfg: dict[str, Any] = dict(kwargs.pop("config", {}))
    if cfg_from_pos:
        # kwargs.config has priority
        cfg = {**cfg_from_pos, **cfg}

    # 3) Move deprecated kwargs into config
    moved: dict[str, Any] = {}
    for key in list(kwargs.keys()):
        if key in _DEPRECATED_CFG_KEYS:
            moved[key] = kwargs.pop(key)
    if moved:
        cfg = {**moved, **cfg}

    # 4) One-time info per caller if we migrated anything
    if moved or cfg_from_pos:
        where = ""
        try:
            for fr in inspect.stack():
                fname = fr.filename
                if "/streamlit/" not in fname:
                    where = f"{fname}:{fr.lineno}"
                    break
        except Exception:
            where = ""

        key = (where or "unknown", tuple(sorted(moved.keys())), bool(cfg_from_pos))
        if key not in _notified_callers:
            _notified_callers.add(key)
            msg = "Plotly compat: migrated deprecated arguments to `config`"
            if where:
                msg += f" in {where}"
            if moved:
                msg += f". Keys: {', '.join(sorted(moved.keys()))}"
            if cfg_from_pos:
                msg += "; positional dict treated as `config`"
            st.info(msg)

    # 5) Final call
    if cfg:
        kwargs["config"] = cfg
    return _ORIG_PLOTLY_CHART(fig, **kwargs)


# Activate global patch
st.plotly_chart = _plotly_chart_compat  # type: ignore[assignment]

# ============================================================================
# Page configuration
# ============================================================================

st.set_page_config(
    page_title="GammaEdge – Portfolio Analytics",
    page_icon="📊",
    layout="wide",
)

# ============================================================================
# Layout
# ============================================================================

# Apply global styles
st.markdown(get_global_styles(), unsafe_allow_html=True)

# Hero Section - Clean and Spacious
st.markdown(
    f"""
<div style="padding: 80px 0 64px 0; text-align: center; max-width: 1200px; margin: 0 auto;">
<h1 style="font-size: 3.5rem; font-weight: 700; letter-spacing: -0.02em; margin-bottom: 16px; color: {COLORS['text_primary']};">
GammaEdge
</h1>
<p style="font-size: 1.25rem; font-weight: 400; color: {COLORS['text_secondary']}; margin-bottom: 64px;">
Quantitative Portfolio Analytics · Precise. Powerful. Professional.
</p>
</div>
""",
    unsafe_allow_html=True,
)

# Module Cards Grid
st.markdown(
    f"""
<div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 24px; max-width: 1200px; margin: 0 auto 64px auto;">
<div class="module-card" style="background: {COLORS['bg_secondary']}; border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 12px; padding: 32px 24px; transition: box-shadow 0.3s ease;">
<div style="font-size: 2.5rem; margin-bottom: 16px; text-align: center;">📊</div>
<h3 style="font-size: 1.25rem; font-weight: 600; margin-bottom: 12px; text-align: center;">
Data & Metrics
</h3>
<p style="font-size: 0.875rem; color: {COLORS['text_secondary']}; line-height: 1.5; text-align: center;">
Load returns, clean series, analyze data quality, and export quantitative metrics
</p>
</div>
<div class="module-card" style="background: {COLORS['bg_secondary']}; border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 12px; padding: 32px 24px; transition: box-shadow 0.3s ease;">
<div style="font-size: 2.5rem; margin-bottom: 16px; text-align: center;">🧠</div>
<h3 style="font-size: 1.25rem; font-weight: 600; margin-bottom: 12px; text-align: center;">
Model & Optimize
</h3>
<p style="font-size: 0.875rem; color: {COLORS['text_secondary']}; line-height: 1.5; text-align: center;">
Build covariance, run optimizers, backtest with transaction costs and constraints
</p>
</div>
<div class="module-card" style="background: {COLORS['bg_secondary']}; border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 12px; padding: 32px 24px; transition: box-shadow 0.3s ease;">
<div style="font-size: 2.5rem; margin-bottom: 16px; text-align: center;">🚀</div>
<h3 style="font-size: 1.25rem; font-weight: 600; margin-bottom: 12px; text-align: center;">
Deploy & Monitor
</h3>
<p style="font-size: 0.875rem; color: {COLORS['text_secondary']}; line-height: 1.5; text-align: center;">
Attribute performance, generate reports, stress-test with regime detection
</p>
</div>
</div>
<style>
.module-card:hover {{
box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
}}
</style>
""",
    unsafe_allow_html=True,
)

# Quick Start Section
st.markdown("---")

st.markdown(
    f"""
<div style="margin: 48px 0 24px 0;">
<h2 style="font-size: 2rem; font-weight: 600; margin-bottom: 32px; text-align: center; color: {COLORS['text_primary']};">
Quick Start
</h2>
</div>
""",
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.page_link(
        "pages/01_Data.py",
        label="📦 Data Module",
        use_container_width=True,
    )
    st.caption("Load historical data, clean series, compute quantitative metrics")

with col2:
    st.page_link(
        "pages/03_Optimizer.py",
        label="🔬 Optimizer",
        use_container_width=True,
    )
    st.caption("Run HRP, Risk Parity, Mean-Variance, Black-Litterman, CVaR optimization")

with col3:
    st.page_link(
        "pages/04_Backtest.py",
        label="📈 Backtest",
        use_container_width=True,
    )
    st.caption("Rolling rebalance, transaction costs, bootstrap metrics, grid search")

st.markdown("---")

# All Modules Section
st.markdown(
    f"""
<div style="margin: 48px 0 24px 0;">
<h2 style="font-size: 1.5rem; font-weight: 600; margin-bottom: 24px; color: {COLORS['text_primary']};">
All Modules
</h2>
</div>
""",
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.markdown("**Data & Risk Model**")
    st.page_link("pages/01_Data.py", label="01 – Data")
    st.page_link("pages/02_RiskModel.py", label="02 – Risk Model")

with col2:
    st.markdown("**Construction & Backtest**")
    st.page_link("pages/03_Optimizer.py", label="03 – Optimizer")
    st.page_link("pages/04_Backtest.py", label="04 – Backtest")
    st.page_link("pages/08_RegimeDetection.py", label="08 – Regime Detection")

with col3:
    st.markdown("**Attribution & Reports**")
    st.page_link("pages/05_Attribution.py", label="05 – Attribution")
    st.page_link("pages/06_Reporting.py", label="06 – Reporting")
    st.page_link("pages/07_Scenarios.py", label="07 – Scenarios")

st.markdown("---")

# Footer
st.markdown(
    f"""
<div style="text-align: center; margin-top: 64px; padding: 24px 0; color: {COLORS['text_tertiary']}; font-size: 0.875rem;">
GammaEdge is a research sandbox. Results are for educational and prototyping purposes only – not investment advice.
</div>
""",
    unsafe_allow_html=True,
)
