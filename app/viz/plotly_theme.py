"""
Plotly Theme for GammaEdge - Apple-Inspired Data Visualization

Provides a cohesive, premium Plotly theme matching the GammaEdge design system.
Dark mode optimized with clean grids, subtle colors, and generous spacing.
"""

from __future__ import annotations

import plotly.graph_objects as go

# =============================================================================
# Theme Configuration
# =============================================================================

GAMMAEDGE_THEME = {
    'layout': {
        # Backgrounds
        'paper_bgcolor': '#000000',        # Pure black
        'plot_bgcolor': '#1C1C1E',         # Elevated dark
        
        # Typography (SF Pro inspired)
        'font': {
            'family': 'system-ui, -apple-system, BlinkMacSystemFont, "SF Pro", sans-serif',
            'size': 14,
            'color': '#FFFFFF',
        },
        'title': {
            'font': {'size': 20, 'color': '#FFFFFF'},
            'x': 0.02,  # Left-aligned
            'xanchor': 'left',
        },
        
        # Axes (minimal, clean)
        'xaxis': {
            'gridcolor': 'rgba(255, 255, 255, 0.05)',   # Subtle grid
            'linecolor': 'rgba(255, 255, 255, 0.1)',
            'tickfont': {'size': 12, 'color': 'rgba(235,235,245,0.6)'},
            'title': {'font': {'size': 14, 'color': 'rgba(235,235,245,0.8)'}},
            'showgrid': True,
            'zeroline': False,
        },
        'yaxis': {
            'gridcolor': 'rgba(255, 255, 255, 0.05)',
            'linecolor': 'rgba(255, 255, 255, 0.1)',
            'tickfont': {'size': 12, 'color': 'rgba(235,235,245,0.6)'},
            'title': {'font': {'size': 14, 'color': 'rgba(235,235,245,0.8)'}},
            'showgrid': True,
            'zeroline': False,
        },
        
        # Color palette (financial context)
        'colorway': [
            '#0A84FF',  # System blue
            '#30D158',  # System green
            '#FFD60A',  # System yellow
            '#FF9F0A',  # System orange
            '#FF375F',  # System pink
            '#BF5AF2',  # System purple
        ],
        
        # Hover
        'hoverlabel': {
            'bgcolor': '#2C2C2E',
            'bordercolor': 'rgba(255, 255, 255, 0.1)',
            'font': {'family': 'SF Mono, monospace', 'size': 12, 'color': '#FFFFFF'},
        },
        'hovermode': 'x unified',
        
        # Legend
        'legend': {
            'bgcolor': 'rgba(28, 28, 30, 0.8)',
            'bordercolor': 'rgba(255, 255, 255, 0.1)',
            'borderwidth': 1,
            'font': {'size': 12},
        },
        
        # Margins (generous spacing)
        'margin': {'t': 60, 'r': 40, 'b': 60, 'l': 60},
    }
}


def apply_gammaedge_theme(fig: go.Figure) -> go.Figure:
    """
    Apply the GammaEdge Apple-inspired theme to a Plotly figure.
    
    Args:
        fig: Plotly Figure object
        
    Returns:
        Modified figure with theme applied
        
    Example:
        >>> fig = go.Figure(data=go.Scatter(x=[1, 2, 3], y=[4, 5, 6]))
        >>> fig = apply_gammaedge_theme(fig)
        >>> fig.show()
    """
    # Apply base theme
    fig.update_layout(**GAMMAEDGE_THEME['layout'])
    
    return fig


def enhance_equity_curve(fig: go.Figure, fill_color: str = 'rgba(10, 132, 255, 0.1)') -> go.Figure:
    """
    Enhance equity curve visualization with area fill and rich hover info.
    
    Args:
        fig: Plotly Figure with equity curve
        fill_color: RGBA color for area under curve
        
    Returns:
        Enhanced figure
        
    Example:
        >>> fig = go.Figure(data=go.Scatter(x=dates, y=equity, mode='lines'))
        >>> fig = enhance_equity_curve(fig)
    """
    # Add gradient fill under curve
    fig.update_traces(
        fill='tozeroy',
        fillcolor=fill_color,
        line=dict(width=2, color='#0A84FF'),
        selector=dict(mode='lines')
    )
    
    # Rich hover template
    fig.update_traces(
        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>' +
                      'Value: %{y:.4f}<br>' +
                      '<extra></extra>',
        selector=dict(mode='lines')
    )
    
    return fig


def enhance_bar_chart(
    fig: go.Figure,
    positive_color: str = '#30D158',
    negative_color: str = '#FF453A'
) -> go.Figure:
    """
    Enhance bar chart with conditional coloring for positive/negative values.
    
    Args:
        fig: Plotly Figure with bar chart
        positive_color: Color for positive bars
        negative_color: Color for negative bars
        
    Returns:
        Enhanced figure
    """
    # This would need to be applied per-trace basis depending on data
    # For now, just update styling
    fig.update_traces(
        marker=dict(
            line=dict(width=1, color='rgba(255, 255, 255, 0.1)'),
        ),
        selector=dict(type='bar')
    )
    
    return fig


def enhance_heatmap(fig: go.Figure) -> go.Figure:
    """
    Enhance heatmap with better color scale and styling.
    
    Args:
        fig: Plotly Figure with heatmap
        
    Returns:
        Enhanced figure
    """
    fig.update_traces(
        colorscale=[
            [0.0, '#FF453A'],   # Red (negative)
            [0.5, '#FFFFFF'],   # White (neutral)
            [1.0, '#30D158'],   # Green (positive)
        ],
        selector=dict(type='heatmap')
    )
    
    return fig


def create_financial_subplot_fig(
    n_rows: int = 2,
    row_heights: list[float] | None = None,
    subplot_titles: tuple[str, ...] | None = None,
) -> go.Figure:
    """
    Create a subplot figure with GammaEdge theme pre-applied.
    
    Useful for equity + drawdown, or multi-panel financial charts.
    
    Args:
        n_rows: Number of rows
        row_heights: Relative heights for each row
        subplot_titles: Titles for each subplot
        
    Returns:
        Configured subplot figure
        
    Example:
        >>> from plotly.subplots import make_subplots
        >>> fig = create_financial_subplot_fig(
        ...     n_rows=2,
        ...     row_heights=[0.7, 0.3],
        ...     subplot_titles=("Equity", "Drawdown")
        ... )
    """
    from plotly.subplots import make_subplots
    
    if row_heights is None:
        row_heights = [1.0 / n_rows] * n_rows
    
    fig = make_subplots(
        rows=n_rows,
        cols=1,
        row_heights=row_heights,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=subplot_titles,
    )
    
    # Apply theme
    fig = apply_gammaedge_theme(fig)
    
    return fig
