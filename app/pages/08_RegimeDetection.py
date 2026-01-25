# app/pages/08_RegimeDetection.py
"""
Regime Detection Page - TIER 1 Enhancement

HMM-based market regime classification with interactive visualizations.
"""

from __future__ import annotations

import os
import sys

import polars as pl
import streamlit as st

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from portfolio.features.regime_detection import (
    RegimeDetector,
    compute_regime_performance,
)
from portfolio.viz.regime_plots import (
    plot_regime_duration_histogram,
    plot_regime_performance,
    plot_regime_probabilities,
    plot_regime_states,
    plot_regime_transitions,  # Fixed: was plot_transition_matrix
)
from portfolio.viz.plot_utils import show_plot

# Design System
from app.design_system import COLORS, get_global_styles, metric_grid, section_header

# =============================================================================
# Page Config
# =============================================================================
st.set_page_config(page_title="Regime Detection", layout="wide")

# Apply global styles
st.markdown(get_global_styles(), unsafe_allow_html=True)

# Page title with Apple-style
st.markdown(f"""
<div style="margin-bottom: 32px;">
<h1 style="font-size: 2.5rem; font-weight: 600; color: {COLORS['text_primary']}; margin-bottom: 8px;">
🌊 Market Regime Detection (HMM)
</h1>
<p style="font-size: 1rem; color: {COLORS['text_secondary']}; line-height: 1.5;">
Detect market regimes (Bull/Bear/Crisis) using Hidden Markov Models • 3-state HMM • 5 interactive visualizations
</p>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# Data Loading
# =============================================================================
if "returns_wide" not in st.session_state:
    st.warning("⚠️ Please load data first in the **Data** page.")
    st.stop()

df_ret_wide: pl.DataFrame = st.session_state["returns_wide"]

# Validate
if "date" not in df_ret_wide.columns:
    st.error("Returns data must include a 'date' column.")
    st.stop()

tickers = [c for c in df_ret_wide.columns if c != "date"]
if not tickers:
    st.error("No return columns found.")
    st.stop()

# =============================================================================
# Configuration Sidebar
# =============================================================================
with st.sidebar:
    st.header("⚙️ Regime Detection Settings")
    
    # Asset selection
    selected_asset = st.selectbox(
        "Asset for regime detection",
        tickers,
        index=0,
        help="Select which asset to use for detecting market regimes",
    )
    
    # HMM parameters
    st.subheader("HMM Parameters")
    n_regimes = st.slider(
        "Number of regimes",
        min_value=2,
        max_value=4,
        value=3,
        help="Typically 3: Bull, Bear, Crisis",
    )
    
    n_iter = st.slider(
        "Max iterations",
        min_value=50,
        max_value=500,
        value=100,
        step=50,
        help="More iterations = better convergence",
    )
    
    random_state = st.number_input(
        "Random seed",
        min_value=0,
        max_value=9999,
        value=42,
        help="For reproducibility",
    )
    
    st.markdown("---")
    st.caption("**Features used**:")
    st.caption("• Returns")
    st.caption("• Realized volatility (20d)")
    st.caption("• Drawdown depth")

# =============================================================================
# Run Regime Detection
# =============================================================================
st.subheader("1️⃣ Detect Regimes")

if st.button("🚀 Run Regime Detection", type="primary"):
    with st.spinner("Detecting regimes with HMM..."):
        # Prepare data
        df_asset = df_ret_wide.select(["date", selected_asset]).rename(
            {selected_asset: "returns"}
        )
        
        # Remove nulls
        df_asset = df_asset.drop_nulls()
        
        if len(df_asset) < 100:
            st.error(f"Insufficient data for {selected_asset}: only {len(df_asset)} rows.")
            st.stop()
        
        # Fit and predict
        detector = RegimeDetector(
            n_regimes=n_regimes,
            n_iter=n_iter,
            random_state=random_state,
        )
        
        detector.fit(df_asset, returns_col="returns")
        df_regimes = detector.predict(df_asset, returns_col="returns")
        
        # Store in session state
        st.session_state["regime_detector"] = detector
        st.session_state["regime_results"] = df_regimes
        st.session_state["regime_asset"] = selected_asset
        
        st.success(f"✅ Regimes detected for **{selected_asset}**!")

# =============================================================================
# Display Results
# =============================================================================
if "regime_results" in st.session_state:
    detector = st.session_state["regime_detector"]
    df_regimes = st.session_state["regime_results"]
    asset_name = st.session_state["regime_asset"]
    
    # ==========================================================================
    # Regime Statistics
    # ==========================================================================
    st.subheader("2️⃣ Regime Statistics")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**Regime Characteristics**")
        stats = detector.get_regime_stats()  # Returns pandas DataFrame
        st.dataframe(
            stats.round(4),  # Already pandas, no to_pandas() needed
            use_container_width=True,
            hide_index=True,
        )
    
    with col2:
        st.markdown("**Transition Matrix**")
        trans_mat = detector.get_transition_matrix()
        trans_df = pl.DataFrame(
            trans_mat,
            schema=[f"To_{i}" for i in range(detector.n_regimes)],
        )
        st.dataframe(
            trans_df.to_pandas().round(3),
            use_container_width=True,
        )
        st.caption("Rows = current regime, Columns = next regime")
    
    # ==========================================================================
    # Performance by Regime
    # ==========================================================================
    st.subheader("3️⃣ Performance by Regime")
    
    perf = compute_regime_performance(
        df_regimes,
        regime_col="regime_label",
        returns_col="returns",
    )
    
    # Performance metrics as grid
    st.markdown(metric_grid([
        {
            'label': f"{row['regime']} Regime",
            'value': f"{row['mean_return']:.2%}",
            'icon': '🐂' if 'Bull' in str(row['regime']) else ('🐻' if 'Bear' in str(row['regime']) else '⚠️'),
        }
        for _, row in perf.iterrows()
    ], columns=3), unsafe_allow_html=True)
    
    st.dataframe(
        perf.round(4),  # Already pandas, no to_pandas() needed
        use_container_width=True,
        hide_index=True,
    )
    
    # ==========================================================================
    # Interactive Visualizations
    # ==========================================================================
    st.markdown(section_header(
        "Interactive Visualizations",
        "Explore regime dynamics through multiple perspectives",
        "📊"
    ), unsafe_allow_html=True)
    
    # Apply tab styling
    st.markdown(f"""
<style>
.stTabs [data-baseweb="tab-list"] {{
gap: 8px;
background-color: {COLORS['bg_secondary']};
border-radius: 12px;
padding: 4px;
}}
.stTabs [data-baseweb="tab"] {{
padding: 12px 24px;
border-radius: 8px;
font-weight: 500;
transition: background-color 0.2s ease;
color: {COLORS['text_secondary']};
}}
.stTabs [aria-selected="true"] {{
background-color: {COLORS['accent_primary']} !important;
color: {COLORS['text_primary']} !important;
}}
</style>
""", unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Regime States",
        "📊 Probabilities",
        "🔄 Transitions",
        "💰 Performance",
        "⏱️ Duration",
    ])
    
    with tab1:
        st.markdown("**Market Regime Timeline**")
        fig = plot_regime_states(
            df_regimes,
            date_col="date",
            regime_col="regime_label",
            returns_col="returns",
            title=f"Market Regimes: {asset_name}",
        )
        show_plot(fig, key="regime-states")
    
    with tab2:
        st.markdown("**Regime Probability Evolution**")
        fig = plot_regime_probabilities(
            df_regimes,
            date_col="date",
            n_regimes=detector.n_regimes,
            title="Regime Probabilities Over Time",
        )
        show_plot(fig, key="regime-probs")
    
    with tab3:
        st.markdown("**Transition Probability Heatmap**")
        fig = plot_regime_transitions(  # Fixed function name
            trans_mat,
            regime_labels=[f"Regime {i}" for i in range(detector.n_regimes)],
            title="Regime Transition Matrix",
        )
        show_plot(fig, key="regime-trans")
    
    with tab4:
        st.markdown("**Performance Comparison by Regime**")
        fig = plot_regime_performance(
            perf,  # Already pandas, pass directly
            title="Risk-Return by Regime",
        )
        show_plot(fig, key="regime-perf")
    
    with tab5:
        st.markdown("**Regime Duration Distribution**")
        fig = plot_regime_duration_histogram(
            df_regimes,
            regime_col="regime_label",  # Fixed: was "regime"
            title="Regime Duration Histogram",
        )
        show_plot(fig, key="regime-duration")
    
    # ==========================================================================
    # Export
    # ==========================================================================
    st.subheader("5️⃣ Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Export regime labels
        csv_data = df_regimes.select(["date", "regime", "regime_label"]).write_csv()
        st.download_button(
            "⬇️ Download Regime Labels (CSV)",
            csv_data,
            file_name=f"regime_labels_{asset_name}.csv",
            mime="text/csv",
        )
    
    with col2:
        # Export statistics (stats is already pandas)
        stats_csv = stats.to_csv(index=False)
        st.download_button(
            "⬇️ Download Regime Stats (CSV)",
            stats_csv,
            file_name=f"regime_stats_{asset_name}.csv",
            mime="text/csv",
        )
    
    # Current Regime Hero Card
    current_regime = df_regimes.tail(1)["regime_label"].item()
    st.session_state["current_regime"] = current_regime
    
    regime_colors = {
        'Bull': ('#30D158', '🐂'),
        'Bear': ('#FF9F0A', '🐻'),
        'Crisis': ('#FF453A', '⚠️'),
    }
    
    color, icon = regime_colors.get(current_regime, ('#0A84FF', '📊'))
    
    st.markdown(f"""
<div style="background: linear-gradient(135deg, {color}22, {color}11); border: 2px solid {color}; border-radius: 16px; padding: 48px; margin: 32px 0; text-align: center;">
<div style="font-size: 4rem; margin-bottom: 16px;">{icon}</div>
<div style="font-size: 0.875rem; color: rgba(235,235,245,0.6); margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.1em;">
CURRENT MARKET REGIME
</div>
<div style="font-size: 3rem; font-weight: 700; color: {color}; margin-bottom: 8px;">
{current_regime}
</div>
<div style="font-size: 1rem; color: rgba(235,235,245,0.8); margin-top: 16px;">
Detected for {asset_name} as of latest data point
</div>
</div>
""", unsafe_allow_html=True)

else:
    st.info("👆 Click **Run Regime Detection** to get started.")

# =============================================================================
# Help Section
# =============================================================================
with st.expander("ℹ️ How to interpret regimes"):
    st.markdown("""
### Regime Interpretation

**Bull Market** (Regime 0):
- Positive returns
- Low volatility
- Small drawdowns
- High Sharpe ratio

**Bear Market** (Regime 1):
- Negative returns
- Medium volatility
- Moderate drawdowns
- Negative or low Sharpe

**Crisis** (Regime 2):
- Large negative returns
- High volatility
- Deep drawdowns
- Strongly negative Sharpe

### Transition Matrix

- **Diagonal values**: Regime persistence probability
- **Off-diagonal**: Regime switch probability
- Higher diagonal = more persistent regimes

### Applications

1. **Regime-conditional strategies**: Adjust allocation based on current regime
2. **Risk management**: Reduce exposure in Bear/Crisis regimes
3. **Performance attribution**: Decompose returns by market state
""", unsafe_allow_html=True)
