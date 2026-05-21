"""GammaEdge Plotly theme — backward-compatibility shim.

The canonical implementation lives in ``portfolio.viz.theme`` (core layer).
This module re-exports it so existing imports of ``app.viz.plotly_theme``
keep working without violating the architectural rule that the UI layer
(``app/``) depends on the core layer (``portfolio/``) and never the reverse.
"""

from portfolio.viz.theme import (  # noqa: F401
    GAMMAEDGE_THEME,
    apply_gammaedge_theme,
    create_financial_subplot_fig,
    enhance_bar_chart,
    enhance_equity_curve,
    enhance_heatmap,
)

__all__ = [
    "GAMMAEDGE_THEME",
    "apply_gammaedge_theme",
    "create_financial_subplot_fig",
    "enhance_bar_chart",
    "enhance_equity_curve",
    "enhance_heatmap",
]
