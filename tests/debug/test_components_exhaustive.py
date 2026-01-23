"""
Prueba exhaustiva de componentes de design_system
para detectar errores de runtime
"""
import sys
sys.path.insert(0, '.')

print("=" * 70)
print("PRUEBA EXHAUSTIVA DE COMPONENTES")
print("=" * 70)

from app.design_system import (
    COLORS, TYPOGRAPHY, SPACING,
    get_global_styles, data_hero_card, metric_grid, section_header, info_panel
)

print("\n1. Probando data_hero_card con diferentes valores...")
test_cases = [
    # (title, value, subtitle, trend, icon, format_value)
    ("Test Básico", 1.5, "", None, "", True),
    ("Con Subtitle", 2.34, "Annualized", None, "📈", True),
    ("Con Trend Positivo", 100.5, "Test", 0.05, "📊", True),
    ("Con Trend Negativo", -10.2, "Loss", -0.15, "📉", True),
    ("Valor String", "Custom", "No format", None, "💰", False),
    ("Valor Pequeño", 0.0012, "Tiny", None, "🔬", True),
    ("Valor Grande", 1234567.89, "Big", None, "💎", True),
]

for i, (title, value, subtitle, trend, icon, fmt) in enumerate(test_cases, 1):
    try:
        result = data_hero_card(title, value, subtitle, trend, icon, fmt)
        assert "<div" in result
        assert "</div>" in result
        print(f"  ✓ Test {i}: {title}")
    except Exception as e:
        print(f"  ✗ Test {i} FAILED: {e}")
        sys.exit(1)

print("\n2. Probando metric_grid con diferentes configuraciones...")
grid_tests = [
    # Simple 1 column
    ([{'label': 'Test', 'value': 123.45, 'icon': '📊'}], 1),
    # 3 columns with different value types
    ([
        {'label': 'Returns', 'value': 0.156, 'icon': '📈'},
        {'label': 'Vol', 'value': 0.182, 'icon': '📊'},
        {'label': 'Sharpe', 'value': 2.34, 'icon': '⚡'},
    ], 3),
    # With trends and negatives
    ([
        {'label': 'CAGR', 'value': 0.12, 'icon': '📈', 'trend': 0.05},
        {'label': 'Max DD', 'value': -0.23, 'icon': '⚠️', 'negative': True},
    ], 2),
    # String values
    ([
        {'label': 'Status', 'value': 'Active', 'icon': '✅'},
        {'label': 'Mode', 'value': 'Live', 'icon': '🔴'},
    ], 2),
    # 5 columns (max)
    ([
        {'label': f'Metric {i}', 'value': i * 10.5, 'icon': '📊'}
        for i in range(1, 6)
    ], 5),
]

for i, (metrics, cols) in enumerate(grid_tests, 1):
    try:
        result = metric_grid(metrics, cols)
        assert "<div" in result
        assert "grid-template-columns" in result
        print(f"  ✓ Grid test {i}: {len(metrics)} metrics, {cols} columns")
    except Exception as e:
        print(f"  ✗ Grid test {i} FAILED: {e}")
        sys.exit(1)

print("\n3. Probando section_header...")
header_tests = [
    ("Simple Title", "", ""),
    ("With Subtitle", "This is a description", ""),
    ("With Icon", "", "📊"),
    ("Full Header", "Complete with all", "🎯"),
]

for title, subtitle, icon in header_tests:
    try:
        result = section_header(title, subtitle, icon)
        assert "<div" in result
        assert title in result
        print(f"  ✓ Header: {title[:20]}...")
    except Exception as e:
        print(f"  ✗ Header test FAILED: {e}")
        sys.exit(1)

print("\n4. Probando info_panel...")
panel_types = ['info', 'warning', 'success', 'error']
for panel_type in panel_types:
    try:
        result = info_panel(f"Test {panel_type} panel", panel_type)
        assert "<div" in result
        assert panel_type in result or "Test" in result
        print(f"  ✓ Panel type: {panel_type}")
    except Exception as e:
        print(f"  ✗ Panel {panel_type} FAILED: {e}")
        sys.exit(1)

print("\n5. Probando get_global_styles...")
try:
    styles = get_global_styles()
    assert "<style>" in styles
    assert "</style>" in styles
    assert ".stApp" in styles
    assert "background:" in styles
    # Check for critical styles
    assert COLORS['bg_primary'] in styles
    assert "stTabs" in styles  # Tab styling present
    assert "stButton" in styles  # Button styling present
    print("  ✓ Global styles complete")
except Exception as e:
    print(f"  ✗ Global styles FAILED: {e}")
    sys.exit(1)

print("\n6. Verificando que COLORS/TYPOGRAPHY/SPACING son consistentes...")
try:
    # Verify essential color keys
    essential_colors = [
        'bg_primary', 'bg_secondary', 'text_primary', 'text_secondary',
        'data_positive', 'data_negative', 'accent_primary'
    ]
    for key in essential_colors:
        assert key in COLORS, f"Missing color: {key}"
        assert isinstance(COLORS[key], str), f"Color {key} not string"
    print(f"  ✓ {len(COLORS)} colors defined")
    
    # Verify typography
    essential_typo = ['hero', 'title_1', 'title_2', 'body', 'caption']
    for key in essential_typo:
        assert key in TYPOGRAPHY, f"Missing typography: {key}"
    print(f"  ✓ {len(TYPOGRAPHY)} typography scales defined")
    
    # Verify spacing
    assert len(SPACING) >= 7, "Not enough spacing units"
    print(f"  ✓ {len(SPACING)} spacing units defined")
    
except Exception as e:
    print(f"  ✗ Tokens verification FAILED: {e}")
    sys.exit(1)

print("\n" + "=" * 70)
print("✅ TODOS LOS COMPONENTES FUNCIONAN CORRECTAMENTE")
print("=" * 70)
print("\nSe probaron:")
print(f"  - {len(test_cases)} casos de data_hero_card")
print(f"  - {len(grid_tests)} configuraciones de metric_grid")
print(f"  - {len(header_tests)} variantes de section_header")
print(f"  - {len(panel_types)} tipos de info_panel")
print(f"  - Estilos globales completos")
print(f"  - {len(COLORS)} color tokens")
print(f"  - {len(TYPOGRAPHY)} typography scales")
print(f"  - {len(SPACING)} spacing units")
