# app/pages/02_RiskModel.py
from __future__ import annotations

import io
import json
import hashlib
import numpy as np
import polars as pl
import streamlit as st
from datetime import datetime, timezone
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from portfolio.features.risk_models import (
    compute_mu_sigma,
    correlation_from_cov,
    black_litterman_mu,
    capm_mu,
    pca_factor_cov,
)
from portfolio.viz.plot_utils import (
    HeatmapOrder,
    corr_heatmap,
    corr_heatmap_gl,
    corr_dendrogram,
    covariance_spectrum,
    scree_plot,
    network_corr_graph,
    risk_contributions_bar,
)

# ──────────────────────────────────────────────────────────────────────────────
# Config & guards
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Risk Model", layout="wide")
st.title("📐 Risk Model")

if "returns_wide" not in st.session_state:
    st.warning("Load data first in the Data page.")
    st.stop()

df_ret_wide: pl.DataFrame = st.session_state["returns_wide"]

def _validate_returns_wide(df: pl.DataFrame) -> pl.DataFrame:
    # 0) Tipo y columna fecha
    if not isinstance(df, pl.DataFrame):
        st.error("returns_wide in session_state is not a Polars DataFrame."); st.stop()
    if "date" not in df.columns:
        st.error("returns_wide must include a 'date' column."); st.stop()

    # 1) Normaliza fecha + orden + unicidad
    df = (df
          .with_columns(pl.col("date").cast(pl.Datetime, strict=False))
          .sort("date"))
    if df["date"].n_unique() < df.height:
        st.warning("Duplicate dates detected — keeping last per timestamp.")
        df = df.unique(subset=["date"], keep="last")

    # 2) Fuerza numérico en todas las columnas ≠ date
    value_cols = [c for c in df.columns if c != "date"]
    df = df.with_columns([pl.col(c).cast(pl.Float64, strict=False).alias(c) for c in value_cols])

    # 3) Limpieza de no finitos: ±inf → null
    df = df.with_columns([pl.when(pl.col(c).is_finite()).then(pl.col(c)).otherwise(None).alias(c)
                          for c in value_cols])

    # 4) Elimina columnas completamente vacías
    null_counts = df.select([pl.col(c).is_null().sum().alias(c) for c in value_cols]).row(0)
    n_rows = df.height
    drop_cols = [c for c, nnull in zip(value_cols, null_counts) if (nnull == n_rows)]
    if drop_cols:
        st.warning(f"Dropping empty return columns: {', '.join(drop_cols)}")
        df = df.drop(drop_cols)
        value_cols = [c for c in value_cols if c not in drop_cols]

    # 5) Detecta columnas constantes (σ≈0) que rompen Σ
    if value_cols:
        stds = df.select([pl.col(c).std(ddof=1).alias(c) for c in value_cols]).row(0)
        const_cols = [c for c, s in zip(value_cols, stds) if (s is None) or (not np.isfinite(s)) or (s <= 1e-14)]
        if const_cols:
            st.warning(f"Dropping near-constant columns (σ≈0): {', '.join(const_cols)}")
            df = df.drop(const_cols)

    # 6) Asegura datos mínimos
    if df.height < 2 or len([c for c in df.columns if c != "date"]) == 0:
        st.error("Not enough valid data after validation for risk modeling."); st.stop()

    return df

df_ret_wide = _validate_returns_wide(df_ret_wide)

tickers = [c for c in df_ret_wide.columns if c != "date"]
if not tickers:
    st.error("No return columns found.")
    st.stop()

# Estado persistente
st.session_state.setdefault("risk_payload", None)
st.session_state.setdefault("risk_ready", False)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _json_default(o):
    if hasattr(o, "isoformat"):
        try:
            return o.isoformat()
        except Exception:
            pass
    import numpy as np
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    return str(o)

def _infer_per_year(dates: pl.Series) -> int:
    """Infer periods/year from median spacing (day/week/month granularity)."""
    s = dates.sort()
    if s.len() < 2:
        return 252
    dt_days = (s.diff().dt.total_days()).drop_nulls()
    med = float(dt_days.median())
    if med <= 3.0:
        return 252  # daily
    elif med <= 9.0:
        return 52   # weekly
    else:
        return 12   # monthly approx

def _apply_fill_policy(df_wide: pl.DataFrame, policy: str):
    """Devuelve df_wide_filled y un reporte de imputación por ticker."""
    if policy == "drop":
        original_h = df_wide.height
        df_filled = df_wide.drop_nulls()
        dropped = original_h - df_filled.height
        report = pl.DataFrame(
            {"policy": ["drop"], "rows_dropped": [int(dropped)], "imputed_pct": [0.0]}
        )
        return df_filled, report
    else:
        value_cols = [c for c in df_wide.columns if c != "date"]
        means = df_wide.select([pl.col(c).mean().alias(c) for c in value_cols])
        na_counts = df_wide.select([pl.col(c).is_null().sum().alias(c) for c in value_cols])
        df_filled = df_wide.clone()
        for c in value_cols:
            m = means.select(c).item()
            df_filled = df_filled.with_columns(
                pl.when(pl.col(c).is_null()).then(pl.lit(m)).otherwise(pl.col(c)).alias(c)
            )
        total_cells = df_wide.height * len(value_cols)
        total_na = int(sum(na_counts.row(0)))
        imputed_pct = (100.0 * total_na / total_cells) if total_cells else 0.0
        report = pl.DataFrame(
            {"policy": ["mean"], "rows_dropped": [0], "imputed_pct": [imputed_pct]}
        )
        return df_filled, report

def _annualize(mu: np.ndarray, Sigma: np.ndarray, per_year: int) -> tuple[np.ndarray, np.ndarray]:
    mu_a = mu * float(per_year)
    Sigma_a = Sigma * float(per_year)
    return mu_a, Sigma_a

def _apply_ridge(Sigma: np.ndarray, eps: float) -> np.ndarray:
    if eps <= 0:
        return Sigma
    n = Sigma.shape[0]
    return Sigma + np.eye(n) * float(eps)

def _cond_number(S: np.ndarray) -> float:
    if S.size == 0:
        return float("nan")
    S_sym = 0.5 * (S + S.T)
    vals = np.linalg.eigvalsh(S_sym)
    lam_min = float(np.min(vals)) if vals.size else np.nan
    lam_max = float(np.max(vals)) if vals.size else np.nan
    return float(lam_max / max(lam_min, 1e-16)) if vals.size else np.nan

def _fingerprint(names: list[str], params: dict) -> str:
    blob = json.dumps({"tickers": names, "params": params}, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]

def _ewma_default(per_year: int) -> float:
    # Defaults coherentes con granularidad
    if per_year >= 250:
        return 0.94
    if per_year >= 50:
        return 0.80
    return 0.60

# ──────────────────────────────────────────────────────────────────────────────
# Core pipeline
# ──────────────────────────────────────────────────────────────────────────────
def _risk_pipeline(
    df_ret_wide: pl.DataFrame,
    *,
    mu_method: str,
    mu_span: int,
    mu_shrink_target: str | None,
    mu_rf_annual: float | None,
    cov_method: str,
    ewma_lambda: float,
    n_factors: int,
    fill_policy: str,
    per_year: int,
    enforce_psd: bool,
    ridge_eps: float,
    stress_test: bool,
    heatmap_method: str,
    heatmap_optimal: bool,
):
    # 1) Fill policy
    df_clean, fill_report = _apply_fill_policy(df_ret_wide, fill_policy)
    names = [c for c in df_clean.columns if c != "date"]
    X = df_clean.select(pl.exclude("date")).to_numpy()

    # 2) μ & Σ (no anualizar aquí)
    mu_shrink_to_vec = None
    if mu_method == "shrunk" and mu_shrink_target is not None:
        # Construimos target: zero / equal-mean / rf (per-period)
        if mu_shrink_target == "zero":
            mu_shrink_to_vec = np.zeros(len(names))
        elif mu_shrink_target == "equal":
            base = np.nanmean(X, axis=0)  # por activo
            target_val = float(np.nanmean(base))
            mu_shrink_to_vec = np.full(len(names), target_val, dtype=float)
        elif mu_shrink_target == "rf":
            rf_ann = float(mu_rf_annual or 0.0)
            rf_per = rf_ann / max(per_year, 1)
            mu_shrink_to_vec = np.full(len(names), rf_per, dtype=float)

    if mu_method == "capm":
        mu = capm_mu(df_clean, market="SPY", annualize=False)  # por periodo
        Sigma = np.cov(X, rowvar=False)
    elif mu_method == "black-litterman":
        # Prior de equilibrio π = δ Σ w_mkt (equal-weight si no hay pesos)
        mu = black_litterman_mu(df_clean, Sigma=None, annualize=False)
        Sigma = np.cov(X, rowvar=False)
    elif cov_method == "pca":
        mu, Sigma, names_out, _ = pca_factor_cov(
            df_clean,
            mu_method=mu_method,
            mu_span=int(mu_span),
            n_factors=int(n_factors),
            annualize=False,
        )
        names = list(names_out)
    else:
        mu, Sigma, names_out = compute_mu_sigma(
            df_clean,
            mu_method=mu_method,
            mu_span=int(mu_span),
            mu_shrink_to=mu_shrink_to_vec,
            cov_method=cov_method,
            ewma_lambda=float(ewma_lambda),
            fill="none",       # ya hicimos fill
            annualize=False,
            psd=enforce_psd,
        )
        names = list(names_out)

    # 3) Annualización consistente
    mu, Sigma = _annualize(mu, Sigma, per_year)

    # 4) Condicionamiento antes de ridge (para diagnóstico)
    cond_pre = _cond_number(Sigma)

    # 5) Ridge opcional
    Sigma = _apply_ridge(Sigma, ridge_eps)

    # 6) Stress test opcional
    if stress_test:
        vol = np.sqrt(np.diag(Sigma))
        Corr = correlation_from_cov(Sigma)
        Sigma = np.outer(vol * 1.2, vol * 1.2) * (Corr * 0.5)

    # 7) Diagnósticos post
    S_sym = 0.5 * (Sigma + Sigma.T)
    eigvals = np.linalg.eigvalsh(S_sym)
    lam_min = float(np.min(eigvals)) if eigvals.size else np.nan
    lam_max = float(np.max(eigvals)) if eigvals.size else np.nan
    cond_post = float(lam_max / max(lam_min, 1e-16)) if eigvals.size else np.nan
    eff_rank = float((eigvals.sum() ** 2) / np.sum(eigvals ** 2)) if eigvals.size else np.nan

    params = {
        "mu_method": mu_method, "mu_span": int(mu_span),
        "mu_shrink_target": mu_shrink_target,
        "mu_rf_annual": float(mu_rf_annual or 0.0),
        "cov_method": cov_method, "ewma_lambda": float(ewma_lambda),
        "n_factors": int(n_factors), "fill_policy": fill_policy,
        "per_year": int(per_year), "enforce_psd": bool(enforce_psd),
        "ridge_eps": float(ridge_eps), "stress_test": bool(stress_test),
        "heatmap_method": heatmap_method, "heatmap_optimal": bool(heatmap_optimal),
    }
    meta = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "params": params,
        "diagnostics": {
            "lambda_min": lam_min, "lambda_max": lam_max,
            "cond_kappa_pre": cond_pre, "cond_kappa_post": cond_post,
            "effective_rank": eff_rank,
        },
        "tickers": names,
        "fingerprint": _fingerprint(names, params),
    }

    return {
        "mu": mu,
        "Sigma": Sigma,
        "names": names,
        "eigvals": eigvals,
        "cond_pre": cond_pre,
        "cond_post": cond_post,
        "eff_rank": eff_rank,
        "fill_report": fill_report,
        "meta": meta,
    }

# ──────────────────────────────────────────────────────────────────────────────
# Inputs (UI)
# ──────────────────────────────────────────────────────────────────────────────
default_per_year = _infer_per_year(df_ret_wide["date"])
ewma_default = _ewma_default(default_per_year)
st.caption(
    f"Annualization default inferred: **{default_per_year}** periods/year · "
    f"EWMA λ default: **{ewma_default:.2f}**"
)

c1, c2, c3 = st.columns(3)
with c1:
    mu_method = st.selectbox(
        "Expected returns (μ)",
        ["historical", "ema", "shrunk", "capm", "black-litterman"],
        index=0,
    )
    mu_span = st.number_input("EMA span (if μ=ema)", min_value=5, max_value=360, value=60, step=5)
with c2:
    cov_method = st.selectbox("Covariance (Σ)", ["sample", "oas", "lw", "ewma", "pca"], index=0)
    ewma_lambda = st.slider("EWMA λ", min_value=0.80, max_value=0.995, value=float(ewma_default), step=0.005)
    n_factors = st.slider("PCA factors (if Σ=pca)", min_value=1, max_value=len(tickers), value=min(5, len(tickers)))
with c3:
    fill_policy = st.selectbox("NaN policy", ["drop", "mean"], index=0)
    enforce_psd = st.checkbox("Enforce PSD (clip eigenvalues)", value=True)

# Controles extra para μ-shrunk
mu_shrink_target = None
mu_rf_annual = 0.0
if mu_method == "shrunk":
    st.markdown("**μ shrinkage target**")
    _tcol1, _tcol2 = st.columns(2)
    with _tcol1:
        mu_shrink_target = st.selectbox("Target", ["zero", "equal", "rf"], index=0)
    with _tcol2:
        mu_rf_annual = st.number_input(
            "Risk-free (annual, in return units)",
            min_value=-1.0, max_value=1.0, value=0.0, step=0.001, format="%.3f"
        )

c4, c5, c6 = st.columns(3)
with c4:
    per_year = st.selectbox(
        "Annualization: periods/year",
        [252, 260, 52, 12],
        index=[252, 260, 52, 12].index(default_per_year if default_per_year in (252, 260, 52, 12) else 252),
    )
with c5:
    ridge_eps = st.number_input("Ridge ε on Σ (0 = off)", min_value=0.0, max_value=1.0, value=0.0, step=0.0001, format="%.4f")
with c6:
    stress_test = st.checkbox("Enable Stress Testing (+20% vol, -50% corr)", value=False)

# Orden del heatmap y WebGL
h1, h2, h3 = st.columns(3)
with h1:
    heatmap_method = st.selectbox("Heatmap linkage", ["single", "complete", "average", "ward"], index=2)
with h2:
    heatmap_optimal = st.checkbox("Optimal leaf ordering", value=True)
with h3:
    show_dendro = st.checkbox("Show dendrogram", value=False)

# ──────────────────────────────────────────────────────────────────────────────
# Acción (cálculo) → guarda en session_state
# ──────────────────────────────────────────────────────────────────────────────
if st.button("Estimate μ and Σ", type="primary"):
    payload = _risk_pipeline(
        df_ret_wide,
        mu_method=mu_method,
        mu_span=int(mu_span),
        mu_shrink_target=mu_shrink_target,
        mu_rf_annual=float(mu_rf_annual or 0.0),
        cov_method=cov_method,
        ewma_lambda=float(ewma_lambda),
        n_factors=int(n_factors),
        fill_policy=fill_policy,
        per_year=int(per_year),
        enforce_psd=bool(enforce_psd),
        ridge_eps=float(ridge_eps),
        stress_test=bool(stress_test),
        heatmap_method=heatmap_method,
        heatmap_optimal=bool(heatmap_optimal),
    )
    st.session_state["risk_payload"] = payload
    st.session_state["risk_ready"] = True

# ──────────────────────────────────────────────────────────────────────────────
# Render desde session_state (persistente)
# ──────────────────────────────────────────────────────────────────────────────
if st.session_state.get("risk_ready"):
    p = st.session_state["risk_payload"]
    mu, Sigma, names = p["mu"], p["Sigma"], p["names"]
    eigvals, cond_pre, cond_post = p["eigvals"], p["cond_pre"], p["cond_post"]
    fill_report, meta = p["fill_report"], p["meta"]

    # Fill policy report
    st.subheader("NaN Policy Report")
    st.dataframe(fill_report.to_pandas(), use_container_width=True)

    # Diagnostics
    st.subheader("Diagnostics")
    lam_min = float(eigvals.min()) if eigvals.size else np.nan
    lam_max = float(eigvals.max()) if eigvals.size else np.nan
    cA, cB, cC, cD, cE = st.columns(5)
    cA.metric("Assets (N)", len(names))
    cB.metric("Obs (T)", int(df_ret_wide.height))
    cC.metric("λ_min", f"{lam_min:.2e}")
    cD.metric("κ (pre-ridge)", f"{cond_pre:.2e}")
    cE.metric("κ (post-ridge)", f"{cond_post:.2e}")

    # μ
    st.subheader("Expected returns (μ, annualized)")
    mu_df = pl.DataFrame({"ticker": names, "mu": mu}).sort("mu", descending=True)
    st.dataframe(mu_df.to_pandas().round(6), use_container_width=True)

    # Σ/ρ visualizations
    st.subheader("Correlation heatmap (clustered)")
    order_cfg = HeatmapOrder(
        clustered=True,
        method=meta["params"]["heatmap_method"],
        optimal=bool(meta["params"]["heatmap_optimal"])
    )
    if len(names) > 200:
        st.plotly_chart(corr_heatmap_gl(Sigma, labels=names, is_cov=True, order=order_cfg), use_container_width=True)
    else:
        st.plotly_chart(corr_heatmap(Sigma, labels=names, is_cov=True, order=order_cfg), use_container_width=True)

    if show_dendro:
        st.subheader("Correlation dendrogram")
        st.plotly_chart(
            corr_dendrogram(Sigma, labels=names, is_cov=True, method=meta["params"]["heatmap_method"]),
            use_container_width=True
        )

    st.subheader("Covariance spectrum")
    st.plotly_chart(covariance_spectrum(Sigma), use_container_width=True)

    st.subheader("Scree Plot (explained variance)")
    st.plotly_chart(scree_plot(eigvals), use_container_width=True)

    st.subheader("Correlation Network Graph")
    st.plotly_chart(network_corr_graph(Sigma, names), use_container_width=True)

    # Risk contributions (benchmark equal-weight)
    st.subheader("Risk Contributions (Equal-Weight Benchmark)")
    if len(names) > 0:
        w_eq = np.full(len(names), 1.0 / len(names))
        rc = w_eq * (Sigma @ w_eq)  # contribución absoluta
        st.plotly_chart(
            risk_contributions_bar(rc, names, sort=True, topn=min(30, len(names))),
            use_container_width=True
        )

    # Handoff & exports
    st.session_state["mu_vec"] = mu
    st.session_state["cov_mat"] = Sigma
    st.session_state["asset_names"] = names

    st.subheader("📤 Export artifacts")
    colx, coly, colz, colw, colm = st.columns(5)

    with colx:
        buf_mu = io.StringIO()
        pl.DataFrame({"ticker": names, "mu": mu}).write_csv(buf_mu)
        st.download_button("Download μ (CSV)", buf_mu.getvalue(), file_name="mu.csv", mime="text/csv")

    with coly:
        buf_cov = io.BytesIO()
        np.save(buf_cov, Sigma)
        st.download_button("Download Σ (.npy)", buf_cov.getvalue(), file_name="covariance.npy")

    with colz:
        Corr = correlation_from_cov(Sigma)
        buf_corr = io.BytesIO()
        np.save(buf_corr, Corr)
        st.download_button("Download ρ (.npy)", buf_corr.getvalue(), file_name="correlation.npy")

    with colw:
        buf_cov_csv = io.StringIO()
        pl.DataFrame(Sigma, schema=names).write_csv(buf_cov_csv)
        st.download_button("Download Σ (CSV wide)", buf_cov_csv.getvalue(), file_name="covariance.csv", mime="text/csv")

    with colm:
        meta_blob = json.dumps(
            {**meta, "exported_at_utc": datetime.now(timezone.utc).isoformat()},
            ensure_ascii=False, indent=2, sort_keys=True, default=_json_default
        )
        st.download_button(
            f"Download risk_model_{meta['fingerprint']}.json",
            meta_blob.encode("utf-8"),
            file_name=f"risk_model_{meta['fingerprint']}.json",
            mime="application/json"
        )

    # Alertas útiles de conditioning
    if cond_post > 1e6:
        st.warning("High condition number κ (post); consider increasing Ridge ε or using OAS/PCA.")
    if eigvals.size and eigvals.min() < 0:
        st.warning("Σ has negative eigenvalues; PSD clip or larger Ridge ε recommended.")
