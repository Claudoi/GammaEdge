import inspect

import streamlit as st

# --- Monkey patch: compat Plotly config ---
_ORIG_PLOTLY_CHART = st.plotly_chart

# claves de config que Streamlit deprecó como kwargs directos
_DEPRECATED_CFG_KEYS = {
    "displayModeBar",
    "displaylogo",
    "modeBarButtons",
    "staticPlot",
    "scrollZoom",
    "editable",
}

# para no spamear logs, recuerde ya avisados por (file:line)
_notified_callers = set()

def _plotly_chart_compat(fig, *args, **kwargs):
    """
    Acepta usos antiguos y los migra a config={...} sin romper nada.
    - Si hay dict posicional (segundo arg), lo trata como config.
    - Si hay kwargs deprecados, los mueve a config.
    - Loguea una nota 1 vez por origen (archivo:línea).
    """
    # 1) Si el primer arg posicional tras fig es un dict → es config
    cfg_from_pos = {}
    if len(args) >= 1 and isinstance(args[0], dict):
        cfg_from_pos = dict(args[0])

    # 2) Extraer/crear config kwarg
    cfg = dict(kwargs.pop("config", {}))
    if cfg_from_pos:
        # merge con prioridad a kwargs.config
        cfg = {**cfg_from_pos, **cfg}

    # 3) Mover kwargs deprecados → config
    moved = {}
    for k in list(kwargs.keys()):
        if k in _DEPRECATED_CFG_KEYS:
            moved[k] = kwargs.pop(k)
    if moved:
        cfg = {**moved, **cfg}

    # 4) Aviso (solo una vez por llamador) si migramos algo o había dict posicional
    if moved or cfg_from_pos:
        where = ""
        try:
            for fr in inspect.stack():
                fname = fr.filename
                if "/streamlit/" not in fname:
                    where = f"{fname}:{fr.lineno}"
                    break
        except Exception:
            pass

        key = (where or "unknown", tuple(sorted(moved.keys())), bool(cfg_from_pos))
        if key not in _notified_callers:
            _notified_callers.add(key)
            st.info(
                "Plotly compat: migré argumentos deprecados a `config`"
                + (f" en {where}" if where else "")
                + (f". Claves: {', '.join(sorted(moved.keys()))}" if moved else "")
                + ("; dict posicional tratado como `config`" if cfg_from_pos else "")
            )

    # 5) Llamada final con config limpio
    if cfg:
        kwargs["config"] = cfg
    return _ORIG_PLOTLY_CHART(fig, **kwargs)

# Activar parche global
st.plotly_chart = _plotly_chart_compat  # type: ignore
# --- fin parche ---