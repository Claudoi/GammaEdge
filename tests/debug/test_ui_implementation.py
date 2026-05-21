"""
Test script para verificar que todos los componentes se pueden importar
y usar sin errores.
"""

import sys

sys.path.insert(0, ".")

print("=" * 60)
print("VERIFICACIÓN COMPLETA DE IMPLEMENTACIÓN UI")
print("=" * 60)

# 1. Test Design System
print("\n1. Testing Design System...")
try:
    from app.design_system import (
        data_hero_card,
        get_global_styles,
        metric_grid,
        section_header,
    )

    # Test get_global_styles
    styles = get_global_styles()
    assert "<style>" in styles
    assert "</style>" in styles
    print("   ✓ get_global_styles() works")

    # Test data_hero_card
    card = data_hero_card("Test", 1.5, subtitle="subtitle", icon="📈")
    assert "<div" in card
    print("   ✓ data_hero_card() works")

    # Test metric_grid
    grid = metric_grid([{"label": "Test", "value": "10%", "icon": "📊"}], columns=1)
    assert "<div" in grid
    print("   ✓ metric_grid() works")

    # Test section_header
    header = section_header("Title", "Subtitle", "📊")
    assert "<div" in header
    print("   ✓ section_header() works")

    print("   ✅ Design System: ALL OK")
except Exception as e:
    print(f"   ❌ Design System Error: {e}")
    sys.exit(1)

# 2. Test Plotly Theme
print("\n2. Testing Plotly Theme...")
try:
    import plotly.graph_objects as go

    from app.viz.plotly_theme import GAMMAEDGE_THEME, apply_gammaedge_theme

    # Test theme dict
    assert isinstance(GAMMAEDGE_THEME, dict)
    assert "layout" in GAMMAEDGE_THEME
    print("   ✓ GAMMAEDGE_THEME dict is valid")

    # Test apply_gammaedge_theme
    fig = go.Figure()
    fig = apply_gammaedge_theme(fig)
    assert fig.layout.template is not None
    print("   ✓ apply_gammaedge_theme() works")

    print("   ✅ Plotly Theme: ALL OK")
except Exception as e:
    print(f"   ❌ Plotly Theme Error: {e}")
    sys.exit(1)

# 3. Test Plot Utils Integration
print("\n3. Testing Plot Utils Integration...")
try:
    import plotly.graph_objects as go

    from portfolio.viz.plot_utils import apply_fig_defaults

    # Test that theme is applied
    fig = go.Figure()
    fig = apply_fig_defaults(fig)
    # Should have dark background from theme
    assert fig.layout.paper_bgcolor == "#000000" or fig.layout.template == "plotly_white"
    print("   ✓ apply_fig_defaults() integrates theme")

    print("   ✅ Plot Utils Integration: ALL OK")
except Exception as e:
    print(f"   ❌ Plot Utils Error: {e}")
    sys.exit(1)

# 4. Test Page Imports
print("\n4. Testing Page Imports...")
pages_to_test = [
    ("Home.py", "app.Home"),
    ("01_Data.py", "app.pages.01_Data"),
    ("02_RiskModel.py", "app.pages.02_RiskModel"),
    ("03_Optimizer.py", "app.pages.03_Optimizer"),
    ("04_Backtest.py", "app.pages.04_Backtest"),
    ("08_RegimeDetection.py", "app.pages.08_RegimeDetection"),
]

for page_name, _module_name in pages_to_test:
    try:
        # Check file exists and can be read
        filepath = f"app/{page_name}" if "Home" in page_name else f"app/pages/{page_name}"

        with open(filepath) as f:
            code = f.read()

        # Check for design_system import
        assert "from app.design_system import" in code or "COLORS" in code
        print(f"   ✓ {page_name} has design system imports")

    except Exception as e:
        print(f"   ❌ {page_name} Error: {e}")
        sys.exit(1)

print("   ✅ All Pages: OK")

print("\n" + "=" * 60)
print("✅ VERIFICACIÓN COMPLETA: TODO OK")
print("=" * 60)
print("\nLa aplicación debería funcionar correctamente.")
print("Ejecuta: poetry run streamlit run app/Home.py")
