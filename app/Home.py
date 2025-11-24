# app/Home.py
from __future__ import annotations

import inspect
from typing import Any

import streamlit as st

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
# UI helpers
# ============================================================================


def _pill(text: str) -> None:
    """Small rounded label."""
    st.markdown(
        f"""
        <span style="
            display:inline-block;
            padding:2px 10px;
            border-radius:999px;
            background:rgba(255,255,255,0.08);
            border:1px solid rgba(255,255,255,0.12);
            font-size:0.78rem;
        ">{text}</span>
        """,
        unsafe_allow_html=True,
    )


# ============================================================================
# Layout
# ============================================================================

# Subtle background
st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at top left, #101520 0, #05060a 40%, #020308 100%);
        color: #e5e7eb;
    }
    section.main > div {
        padding-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Hero
col_left, col_right = st.columns([2.2, 1.3], gap="large")

with col_left:
    st.markdown("### GammaEdge")
    st.markdown(
        """
        <div style="font-size:2.2rem; font-weight:700; line-height:1.1;">
            Quant Portfolio Analytics &nbsp;<span style="opacity:0.75;">Playground</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("")
    _pill("Black-Litterman • HRP • CVaR • Factor Attribution")

    st.markdown(
        """
        <p style="margin-top:1rem; max-width:560px; opacity:0.85;">
        Build, backtest and attribute multi-asset portfolios with a fully local,
        research-grade toolbox. Data in, risk out. No magic, only math.
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("")
    c1, c2 = st.columns([1.2, 1.0])
    with c1:
        st.page_link(
            "pages/01_Data.py",
            label="🚀 Start: Load & Inspect Data",
        )
    with c2:
        st.page_link(
            "pages/03_Optimizer.py",
            label="🧠 Go to Optimizer",
        )

    st.markdown("")
    st.caption(
        "Tip: run from repo root with `poetry run streamlit run app/Home.py` so imports and pages resolve correctly."
    )

with col_right:
    st.markdown(
        """
        <div style="
            border-radius:24px;
            padding:18px 18px 14px 18px;
            background:linear-gradient(145deg, rgba(96,165,250,0.16), rgba(56,189,248,0.04));
            border:1px solid rgba(148,163,184,0.6);
            box-shadow:0 20px 40px rgba(15,23,42,0.65);
        ">
          <div style="font-size:0.9rem; opacity:0.85; margin-bottom:0.2rem;">Session status</div>
          <div style="font-size:1.15rem; font-weight:600; margin-bottom:0.6rem;">
            Environment checks
          </div>
          <ul style="padding-left:1.1rem; margin:0; font-size:0.88rem; line-height:1.5;">
            <li>✅ Linting: <code>ruff check .</code></li>
            <li>✅ Types: <code>mypy portfolio app</code></li>
            <li>✅ Tests: <code>pytest -q</code> (coverage ≥ 65%)</li>
          </ul>
          <div style="margin-top:0.7rem; font-size:0.8rem; opacity:0.78;">
            Ready to experiment with scenarios, reporting and attribution.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# ============================================================================
# Navigation cards
# ============================================================================

st.markdown("#### Modules")

col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.markdown("##### 1. Data & Risk Model")
    st.page_link("pages/01_Data.py", label=" 01 – Data")
    st.page_link("pages/02_RiskModel.py", label=" 02 – Risk Model")
    st.markdown(
        "<p style='font-size:0.83rem; opacity:0.8;'>Load returns, clean series, "
        "and build covariance structures (sample / EWMA).</p>",
        unsafe_allow_html=True,
    )

with col2:
    st.markdown("##### 2. Construction & Backtest")
    st.page_link("pages/03_Optimizer.py", label=" 03 – Optimizer")
    st.page_link("pages/04_Backtest.py", label=" 04 – Backtest")
    st.markdown(
        "<p style='font-size:0.83rem; opacity:0.8;'>Run HRP, min-var, risk parity or "
        "tracking-error portfolios and backtest with turnover controls.</p>",
        unsafe_allow_html=True,
    )

with col3:
    st.markdown("##### 3. Attribution & Reports")
    st.page_link("pages/05_Attribution.py", label=" 05 – Attribution")
    st.page_link("pages/06_Reporting.py", label=" 06 – Reporting")
    st.page_link("pages/07_Scenarios.py", label=" 07 – Scenarios")
    st.markdown(
        "<p style='font-size:0.83rem; opacity:0.8;'>Decompose performance by factors, "
        "generate PDF/HTML reports and stress-test the portfolio.</p>",
        unsafe_allow_html=True,
    )

st.markdown("---")

st.caption(
    "GammaEdge is a research sandbox. Results are for educational and prototyping "
    "purposes only – not investment advice."
)
