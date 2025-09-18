"""Error Correction Model (ECM) construction utilities."""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
from scipy import stats
import warnings

def _build_ecm_model(
    aligned_data,
    cointegration_results,
    hac_maxlags=None,
    allow_contemporaneous=False,   # set False to force q,r ≥ 1 (forecast-friendly)
    ic_kind="BIC",                 # "AIC", "BIC", or "AICc" for lag selection
    include_L3=True,               # include ΔL3 terms in the short run
    max_lags_cap=3                 # small-sample guard for max p/q/r
):
    """
    Build Error Correction Model (ECM) with information-criterion lag selection and diagnostics.

    ECM spec (chosen via grid search):
      ΔL1_t = α + Σ_{i=1..p} φ_i ΔL1_{t-i} + Σ_{j=q_start..q} θ_j ΔL13_{t-j}
                  + [Σ_{m=r_start..r} ψ_m ΔL3_{t-m}] + γ·u_{t-1} + ε_t

    Where u_{t-1} is from cointegration of L1 ~ L13 (+ optional trend).
    If allow_contemporaneous=False, q_start=r_start=1 (no contemporaneous Δ exogs).

    Returns:
      dict with 'model', 'specification', 'diagnostics', 'summary'
    """
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
    from scipy import stats
    import warnings
    warnings.filterwarnings("ignore")

    # ---- helpers ----
    def _ic_stat(model, nobs, k, kind="BIC"):
        kind = kind.upper()
        if kind == "BIC":
            return model.bic
        if kind == "AIC":
            return model.aic
        if kind == "AICc":
            aic = model.aic
            denom = max(nobs - k - 1, 1)
            return aic + (2 * k * (k + 1)) / denom
        return model.aic

    def _ljung_box_safe(resid):
        # Clean and cap lag conservatively
        r = pd.Series(resid).replace([np.inf, -np.inf], np.nan).dropna()
        T = len(r)
        if T <= 5:
            return {"status": "skipped", "lag": np.nan, "stat": np.nan, "pvalue": np.nan}
        L = max(1, min(int(np.sqrt(T)), 12, T - 2))
        out = acorr_ljungbox(r, lags=[L], model_df=0, return_df=True)
        return {
            "status": "ok",
            "lag": int(L),
            "stat": float(out["lb_stat"].iloc[-1]),
            "pvalue": float(out["lb_pvalue"].iloc[-1]),
        }

    # ---- unpack & basic checks ----
    if "residuals" not in cointegration_results:
        raise ValueError("Missing 'residuals' in cointegration_results")

    required_cols = ["Date", "L1", "L3", "L13"]
    if not all(c in aligned_data.columns for c in required_cols):
        missing = [c for c in required_cols if c not in aligned_data.columns]
        raise ValueError(f"Missing required columns in aligned_data: {missing}")

    print("=" * 80)
    print("🔧 BUILDING ERROR CORRECTION MODEL (ECM)")
    print("=" * 80)

    # ---- Step 1: Clean, index, differences, align u_{t-1} ----
    print("📈 Step 1: Index Alignment & First Differences")
    print("-" * 40)
    clean_data = aligned_data[required_cols].dropna().copy().set_index("Date")
    residuals_indexed = cointegration_results["residuals"]
    # Ensure same index on residuals
    if getattr(residuals_indexed, "index", None) is None or len(residuals_indexed) != len(clean_data):
        residuals_indexed = pd.Series(np.asarray(residuals_indexed).ravel(), index=clean_data.index)
    else:
        residuals_indexed = residuals_indexed.copy()
        residuals_indexed.index = clean_data.index

    diff_data = clean_data.diff().dropna()
    diff_data.columns = ["dL1", "dL3", "dL13"]
    u_lag1 = residuals_indexed.shift(1).rename("u_lag1")

    model_data_base = diff_data.join(u_lag1).dropna()
    print(f"   ✅ Created differences: {len(model_data_base)} observations")
    print(f"   ✅ Error correction term u_{{t-1}} aligned: {len(model_data_base)} observations")

    # ---- Step 2: Basic validation ----
    print("Step 2: Data Validation")
    print("-" * 40)
    stds = model_data_base[["dL1", "dL13", "dL3"]].std()
    for k, v in stds.items():
        print(f"   {k} std: {v:.6f}")
    if any(stds < 1e-8):
        z = list(stds[stds < 1e-8].index)
        raise ValueError(f"Differenced series near-constant: {z}")

    # ---- Step 3: HAC setup ----
    print("⚙️  Step 3: HAC Covariance Configuration")
    print("-" * 40)
    if hac_maxlags is None:
        T = len(model_data_base)
        hac_lags = max(1, min(8, int(4 * (T / 100.0) ** (2 / 9))))
        print(f"   📊 Auto-computed HAC maxlags: {hac_lags} (based on T={T})")
    else:
        hac_lags = int(hac_maxlags)
        print(f"   📊 Using provided HAC maxlags: {hac_lags}")

    # ---- Step 4: IC-based lag selection ----
    print("🔍 Step 4: IC-Based Lag Selection")
    print("-" * 40)
    T_eff = len(model_data_base)
    max_lags = min(int(max_lags_cap), max(1, T_eff // 8))  # conservative cap for small T
    p_range = range(1, max_lags + 1)

    q_start = 0 if allow_contemporaneous else 1
    q_range = range(q_start, max_lags + 1) if max_lags >= q_start else []

    if include_L3:
        r_start = 0 if allow_contemporaneous else 1
        r_range = range(r_start, max_lags + 1) if max_lags >= r_start else []
    else:
        r_range = [-1]  # sentinel = no ΔL3 in short run

    print(f"   🔎 Search grid: p∈{list(p_range)}, q∈{list(q_range)}, r∈{list(r_range)}")
    print(f"   📐 Selection criterion: {ic_kind.upper()}")

    best_ic = np.inf
    best_lags = None
    tested = 0
    singular_skipped = 0

    for p in p_range:
        for q in (q_range if q_range else [1e9]):  # if empty and contemporaneous disallowed: skip
            for r in r_range:
                try:
                    model_data = model_data_base.copy()

                    # ΔL1 lags
                    for i in range(1, p + 1):
                        model_data[f"dL1_lag{i}"] = model_data["dL1"].shift(i)

                    # ΔL13 terms (q may start at 1)
                    if q != 1e9:
                        for j in range(q_start, q + 1):
                            col = f"dL13_lag{j}"
                            if j == 0:
                                model_data[col] = model_data["dL13"]
                            else:
                                model_data[col] = model_data["dL13"].shift(j)

                    # ΔL3 terms
                    if include_L3 and r >= 0:
                        for m in range(r_start, r + 1):
                            col = f"dL3_lag{m}"
                            if m == 0:
                                model_data[col] = model_data["dL3"]
                            else:
                                model_data[col] = model_data["dL3"].shift(m)

                    model_data = model_data.dropna()
                    if len(model_data) < 10:
                        continue

                    y = model_data["dL1"]
                    X_cols = [c for c in model_data.columns if c != "dL1"]
                    X = sm.add_constant(model_data[X_cols])

                    # rank check
                    if np.linalg.matrix_rank(X) < X.shape[1]:
                        singular_skipped += 1
                        continue

                    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
                    k_params = len(model.params)
                    ic_val = _ic_stat(model, nobs=len(model_data), k=k_params, kind=ic_kind)
                    tested += 1

                    if ic_val < best_ic:
                        best_ic = ic_val
                        best_lags = (p, int(q if q != 1e9 else -1), int(r))
                except Exception:
                    continue

    if best_lags is None:
        raise ValueError("No valid lag combination found. Try relaxing settings or check data quality.")

    p_opt, q_opt, r_opt = best_lags
    print(f"   ✅ Optimal lags: (p={p_opt}, q={q_opt}, r={r_opt})  (q/r=-1 means term excluded)")
    print(f"   🔄 Combinations tested: {tested}, singular skipped: {singular_skipped}")
    print(f"   🏅 Best {ic_kind.upper()}: {best_ic:.4f}")

    # ---- Step 5: Final model with optimal lags ----
    print("🎯 Step 5: Final ECM Model Estimation")
    print("-" * 40)
    model_data = model_data_base.copy()

    # ΔL1 lags
    for i in range(1, p_opt + 1):
        model_data[f"dL1_lag{i}"] = model_data["dL1"].shift(i)

    # ΔL13 terms
    if q_opt >= 0:
        for j in range(q_start, q_opt + 1):
            col = f"dL13_lag{j}"
            if j == 0:
                model_data[col] = model_data["dL13"]
            else:
                model_data[col] = model_data["dL13"].shift(j)

    # ΔL3 terms
    if include_L3 and r_opt >= 0:
        for m in range(r_start, r_opt + 1):
            col = f"dL3_lag{m}"
            if m == 0:
                model_data[col] = model_data["dL3"]
            else:
                model_data[col] = model_data["dL3"].shift(m)

    model_data = model_data.dropna()
    y_final = model_data["dL1"]

    X_vars = [f"dL1_lag{i}" for i in range(1, p_opt + 1)]
    if q_opt >= 0:
        X_vars += [f"dL13_lag{j}" for j in range(q_start, q_opt + 1)]
    if include_L3 and r_opt >= 0:
        X_vars += [f"dL3_lag{m}" for m in range(r_start, r_opt + 1)]
    X_vars += ["u_lag1"]

    X_final = sm.add_constant(model_data[X_vars])
    final_model = sm.OLS(y_final, X_final).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})

    print("✅ Final ECM Model Specification:")
    print(f"   • Dependent variable: ΔL1_t")
    print(f"   • Sample size: {len(model_data)} observations")
    print(f"   • Variables included: {len(X_final.columns)} (incl. intercept)")
    print(f"   • R-squared: {final_model.rsquared:.4f}, Adj R²: {final_model.rsquared_adj:.4f}")
    print(f"   • HAC maxlags: {hac_lags}")

    # ---- Step 6: Diagnostics ----
    print("🔬 Step 6: Model Diagnostics")
    print("-" * 40)
    resid = final_model.resid
    fitted = final_model.fittedvalues

    # Ljung-Box
    lb = _ljung_box_safe(resid)
    if lb["status"] == "ok":
        print(f"      • Ljung-Box (lag={lb['lag']}): {lb['stat']:.4f} (p={lb['pvalue']:.4f})")
    else:
        print(f"      • Ljung-Box: skipped (T too small)")

    # Breusch-Pagan
    try:
        bp_lm, bp_lm_pvalue, bp_f, bp_f_pvalue = het_breuschpagan(resid, X_final)
    except Exception:
        bp_lm, bp_lm_pvalue = np.nan, np.nan

    # Jarque-Bera
    try:
        jb_stat, jb_pvalue = stats.jarque_bera(resid)
    except Exception:
        jb_stat, jb_pvalue = np.nan, np.nan

    # Error-correction coefficient
    ec_coeff = float(final_model.params["u_lag1"])
    ec_pvalue = float(final_model.pvalues["u_lag1"])
    ec_tstat = float(final_model.tvalues["u_lag1"])

    print(f"      • Error Correction (γ): {ec_coeff:.6f} (t={ec_tstat:.3f}, p={ec_pvalue:.4f})")
    if not np.isnan(bp_lm):
        print(f"      • Breusch-Pagan: LM={bp_lm:.4f} (p={bp_lm_pvalue:.4f})")
    if not np.isnan(jb_stat):
        print(f"      • Jarque-Bera: JB={jb_stat:.4f} (p={jb_pvalue:.4f})")

    if ec_coeff < 0 and ec_pvalue < 0.05:
        print("   ✅ Error correction mechanism is valid (negative & significant)")
    elif ec_coeff < 0:
        print("   ⚠️  Error correction coefficient negative but not significant")
    else:
        print("   ❌ Error correction coefficient is positive (mechanism invalid)")

    # cointegration coefficients (if present) for convenience in forecasting
    coint_coeffs = cointegration_results.get("coefficients", {})

    # Summary/return
    half_life = np.nan
    if (-1.0 < ec_coeff < 0.0):
        half_life = -np.log(2) / np.log(1 + ec_coeff)

    results = {
        "model": final_model,
        "specification": {
            "lags": {"p": p_opt, "q": q_opt, "r": r_opt},
            "sample_size": len(model_data),
            "aic": final_model.aic,
            "bic": final_model.bic,
            "ic_used": ic_kind.upper(),
            "hac_maxlags": hac_lags,
            "regressors": list(X_final.columns),
            "allow_contemporaneous": bool(allow_contemporaneous),
            "include_L3": bool(include_L3),
            "q_start": q_start if q_opt >= 0 else None,
            "r_start": (r_start if include_L3 and r_opt >= 0 else None),
        },
        "diagnostics": {
            "r_squared": final_model.rsquared,
            "adj_r_squared": final_model.rsquared_adj,
            "error_correction_coeff": ec_coeff,
            "error_correction_pvalue": ec_pvalue,
            "error_correction_tstat": ec_tstat,
            "ljung_box_stat": lb.get("stat", np.nan),
            "ljung_box_pvalue": lb.get("pvalue", np.nan),
            "ljung_box_lags": lb.get("lag", np.nan),
            "breusch_pagan_lm": bp_lm,
            "breusch_pagan_pvalue": bp_lm_pvalue,
            "jarque_bera_stat": jb_stat,
            "jarque_bera_pvalue": jb_pvalue,
            "residuals": resid,
            "fitted_values": fitted,
            "coint_coefficients": coint_coeffs,
        },
        "summary": {
            "model_valid": ec_coeff < 0 and ec_pvalue < 0.05,
            "interpretation": (
                f"ECM with p={p_opt} lags of ΔL1, "
                f"{'no ΔL13' if q_opt<0 else f'ΔL13 lags {q_start}..{q_opt}'}"
                f"{'' if not include_L3 else (', no ΔL3' if r_opt<0 else f', ΔL3 lags {r_start}..{r_opt}')}"
            ),
            "error_correction_speed": abs(ec_coeff) if ec_coeff < 0 else 0.0,
            "half_life": half_life,
        },
    }

    print("🎉" + "=" * 80)
    print("✅ ECM MODEL BUILDING COMPLETED SUCCESSFULLY")
    print("=" * 80)
    return results

__all__ = ['_build_ecm_model']
