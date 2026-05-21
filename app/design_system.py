"""
GammaEdge Design System - Apple-Inspired Components

Provides a cohesive design system for financial analytics dashboards with:
- Clean, spacious layouts
- Data-centric visual hierarchy
- Apple HIG principles (Clarity, Deference, Depth, Consistency)
"""

from __future__ import annotations

from typing import Any

# =============================================================================
# Design Tokens
# =============================================================================

COLORS = {
    # Base (dark mode refinado)
    "bg_primary": "#000000",  # Pure black
    "bg_secondary": "#1C1C1E",  # System gray 6 (dark)
    "bg_tertiary": "#2C2C2E",  # System gray 5 (dark)
    "bg_elevated": "#3A3A3C",  # Elevated surfaces
    # Text (alta legibilidad)
    "text_primary": "#FFFFFF",  # Primary labels
    "text_secondary": "rgba(235,235,245,0.6)",  # Secondary labels
    "text_tertiary": "rgba(235,235,245,0.3)",  # Tertiary labels
    # Data visualization (Financial context)
    "data_positive": "#30D158",  # Green (gains)
    "data_negative": "#FF453A",  # Red (losses)
    "data_neutral": "#0A84FF",  # Blue (neutral)
    "data_warning": "#FFD60A",  # Yellow (caution)
    # Accent colors (subtle, purposeful)
    "accent_primary": "#0A84FF",  # System blue
    "accent_secondary": "#64D2FF",  # Light blue
    "accent_tertiary": "#BF5AF2",  # Purple
    # Chart palette (6 distinct colors)
    "chart": ["#0A84FF", "#30D158", "#FFD60A", "#FF9F0A", "#FF375F", "#BF5AF2"],
}

TYPOGRAPHY = {
    # SF Pro inspired scale
    "hero": {
        "size": "3.5rem",  # 56px - Portfolio value, main KPIs
        "weight": 700,
        "line_height": 1.1,
        "letter_spacing": "-0.02em",
    },
    "title_1": {
        "size": "2rem",  # 32px - Section titles
        "weight": 600,
        "line_height": 1.2,
    },
    "title_2": {
        "size": "1.5rem",  # 24px - Subsections
        "weight": 600,
        "line_height": 1.3,
    },
    "body": {
        "size": "1rem",  # 16px - Body text
        "weight": 400,
        "line_height": 1.5,
    },
    "caption": {
        "size": "0.875rem",  # 14px - Captions, labels
        "weight": 400,
        "line_height": 1.4,
    },
    "mono": {
        "family": "SF Mono, JetBrains Mono, Consolas, monospace",
        "size": "0.875rem",
        "weight": 500,
    },
}

SPACING = {
    "xs": "4px",
    "sm": "8px",
    "md": "16px",
    "lg": "24px",
    "xl": "32px",
    "2xl": "48px",
    "3xl": "64px",
}

# =============================================================================
# Component Functions
# =============================================================================


def data_hero_card(
    title: str,
    value: float | str,
    subtitle: str = "",
    trend: float | None = None,
    icon: str = "",
    format_value: bool = True,
) -> str:
    """
    Large, prominent card for displaying KEY quantitative metrics.

    Args:
        title: Metric name (e.g., "Portfolio Sharpe Ratio")
        value: Main value to display
        subtitle: Additional context (e.g., "μ=0.12 | σ=0.18")
        trend: Optional trend indicator (positive/negative)
        icon: Emoji or symbol icon
        format_value: Whether to format numeric values

    Returns:
        HTML/CSS string for st.markdown(unsafe_allow_html=True)

    Example:
        >>> st.markdown(data_hero_card(
        ...     title="Sharpe Ratio",
        ...     value=2.34,
        ...     subtitle="Annualized",
        ...     icon="📈"
        ... ), unsafe_allow_html=True)
    """
    # Format value if needed
    if format_value and isinstance(value, (int, float)):
        abs_val = abs(float(value))
        if abs_val < 0.01:
            value_str = f"{value:.4f}"
        elif abs_val < 10:
            value_str = f"{value:.3f}"
        else:
            value_str = f"{value:,.2f}"
    else:
        value_str = str(value)

    # Trend indicator
    trend_html = ""
    if trend is not None:
        trend_color = COLORS["data_positive"] if trend >= 0 else COLORS["data_negative"]
        trend_symbol = "↑" if trend >= 0 else "↓"
        trend_html = f"""
<div style="font-size: 1.25rem; color: {trend_color}; margin-top: 8px;">
{trend_symbol} {abs(trend):.2%}
</div>
"""

    return f"""
<div style="background: {COLORS['bg_secondary']}; border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 16px; padding: 32px; margin: 24px 0; text-align: center;">
{f'<div style="font-size: 2.5rem; margin-bottom: 12px;">{icon}</div>' if icon else ''}
<div style="font-size: {TYPOGRAPHY['caption']['size']}; color: {COLORS['text_secondary']}; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 12px;">
{title}
</div>
<div style="font-size: {TYPOGRAPHY['hero']['size']}; font-weight: {TYPOGRAPHY['hero']['weight']}; line-height: {TYPOGRAPHY['hero']['line_height']}; letter-spacing: {TYPOGRAPHY['hero']['letter_spacing']}; color: {COLORS['text_primary']}; margin-bottom: 8px;">
{value_str}
</div>
{f'<div style="font-size: {TYPOGRAPHY["body"]["size"]}; color: {COLORS["text_secondary"]};">{subtitle}</div>' if subtitle else ''}
{trend_html}
</div>
"""


def metric_grid(metrics: list[dict[str, Any]], columns: int = 3) -> str:
    """
    Grid layout for multiple related metrics.

    Args:
        metrics: List of metric dictionaries with keys:
            - label: str (required)
            - value: str | float (required)
            - icon: str (optional)
            - trend: float (optional)
            - negative: bool (optional, highlight in red)
        columns: Grid columns (2, 3, 4, or 5)

    Returns:
        HTML/CSS string for st.markdown(unsafe_allow_html=True)

    Example:
        >>> st.markdown(metric_grid([
        ...     {'label': 'CAGR', 'value': 0.156, 'icon': '📈'},
        ...     {'label': 'Sharpe', 'value': 2.34, 'icon': '⚡'},
        ...     {'label': 'Max DD', 'value': -0.23, 'icon': '⚠️', 'negative': True},
        ... ], columns=3), unsafe_allow_html=True)
    """
    grid_html = f"""
<div style="display: grid; grid-template-columns: repeat({columns}, 1fr); gap: 16px; margin: 24px 0;">
"""

    for metric in metrics:
        label = metric.get("label", "")
        value = metric.get("value", "")
        icon = metric.get("icon", "")
        trend = metric.get("trend")
        is_negative = metric.get("negative", False)

        # Format value
        if isinstance(value, (int, float)):
            abs_val = abs(float(value))
            if abs_val < 0.01:
                value_str = f"{value:.4f}"
            elif abs_val < 10:
                value_str = f"{value:.3f}"
            else:
                value_str = f"{value:,.2f}"
        else:
            value_str = str(value)

        # Value color
        if is_negative:
            value_color = COLORS["data_negative"]
        elif trend is not None:
            value_color = COLORS["data_positive"] if trend >= 0 else COLORS["data_negative"]
        else:
            value_color = COLORS["text_primary"]

        # Trend indicator
        trend_html = ""
        if trend is not None:
            trend_symbol = "↑" if trend >= 0 else "↓"
            trend_color = COLORS["data_positive"] if trend >= 0 else COLORS["data_negative"]
            trend_html = (
                f'<span style="color: {trend_color}; margin-left: 8px;">{trend_symbol}</span>'
            )

        grid_html += f"""
<div style="background: {COLORS['bg_secondary']}; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 20px 16px; text-align: center; transition: box-shadow 0.2s ease;">
{f'<div style="font-size: 1.5rem; margin-bottom: 8px;">{icon}</div>' if icon else ''}
<div style="font-size: {TYPOGRAPHY['caption']['size']}; color: {COLORS['text_secondary']}; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.05em;">
{label}
</div>
<div style="font-size: 1.75rem; font-weight: 600; color: {value_color}; line-height: 1.2;">
{value_str}{trend_html}
</div>
</div>
"""

    grid_html += "</div>"
    return grid_html


def section_header(title: str, subtitle: str = "", icon: str = "") -> str:
    """
    Clean section header with optional subtitle and icon.

    Args:
        title: Section title
        subtitle: Optional subtitle/description
        icon: Optional emoji icon

    Returns:
        HTML/CSS string for st.markdown(unsafe_allow_html=True)
    """
    return f"""
<div style="margin-top: {SPACING['3xl']}; margin-bottom: {SPACING['lg']};">
<h2 style="font-size: {TYPOGRAPHY['title_1']['size']}; font-weight: {TYPOGRAPHY['title_1']['weight']}; line-height: {TYPOGRAPHY['title_1']['line_height']}; color: {COLORS['text_primary']}; margin-bottom: {SPACING['sm']};">
{icon + ' ' if icon else ''}{title}
</h2>
{f'<p style="font-size: {TYPOGRAPHY["caption"]["size"]}; color: {COLORS["text_secondary"]}; line-height: {TYPOGRAPHY["caption"]["line_height"]};">{subtitle}</p>' if subtitle else ''}
</div>
"""


def info_panel(content: str, panel_type: str = "info") -> str:
    """
    Styled panel for info/warning/success/error messages.

    Args:
        content: Panel content (supports markdown)
        panel_type: 'info', 'warning', 'success', 'error'

    Returns:
        HTML/CSS string for st.markdown(unsafe_allow_html=True)
    """
    type_config = {
        "info": {"color": COLORS["accent_primary"], "icon": "ℹ️"},
        "warning": {"color": COLORS["data_warning"], "icon": "⚠️"},
        "success": {"color": COLORS["data_positive"], "icon": "✅"},
        "error": {"color": COLORS["data_negative"], "icon": "❌"},
    }

    config = type_config.get(panel_type, type_config["info"])

    return f"""
<div style="background: linear-gradient(135deg, {config['color']}22, {config['color']}11); border-left: 4px solid {config['color']}; border-radius: 8px; padding: 16px 20px; margin: 16px 0;">
<div style="font-size: 1rem; color: {COLORS['text_primary']}; line-height: 1.5;">
<span style="font-size: 1.25rem; margin-right: 8px;">{config['icon']}</span>
{content}
</div>
</div>
"""


def get_global_styles() -> str:
    """
    Returns global CSS styles to be applied once per page.

    Should be called with st.markdown(get_global_styles(), unsafe_allow_html=True)
    at the top of each page.
    """
    return f"""
<style>
/* Global resets and base styles */
.stApp {{
background: {COLORS['bg_primary']};
color: {COLORS['text_primary']};
font-family: system-ui, -apple-system, BlinkMacSystemFont, "SF Pro", sans-serif;
}}
/* Streamlit element overrides */
.stMarkdown, .stText {{
color: {COLORS['text_primary']};
}}
/* Tab styling */
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
/* Button overrides */
.stButton > button {{
background-color: {COLORS['accent_primary']};
color: {COLORS['text_primary']};
border: none;
border-radius: 8px;
padding: 12px 24px;
font-weight: 500;
transition: box-shadow 0.2s ease;
}}
.stButton > button:hover {{
box-shadow: 0 4px 16px rgba(10, 132, 255, 0.3);
}}
/* Metric overrides */
[data-testid="stMetricValue"] {{
font-size: 2rem;
font-weight: 600;
}}
/* Dataframe styling */
.dataframe {{
border-radius: 8px;
overflow: hidden;
}}
.dataframe tbody tr:nth-child(even) {{
background-color: rgba(255, 255, 255, 0.02);
}}
.dataframe tbody tr:hover {{
background-color: rgba(255, 255, 255, 0.05);
}}
/* Subtle hover effects */
.metric-card:hover {{
box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
}}
</style>
"""
