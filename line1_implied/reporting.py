"""Reporting helpers for ECM pipeline."""

import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

def _display_results(cointegration_results, ecm_results, eval_out):
    """
    Console-only summary of the ECM pipeline using your actual objects:
      - cointegration_results: {'coefficients': {'β0_intercept','β1_L13','β2_trend'}, 'residuals', ...}
      - ecm_results: {'specification': {'lags', 'aic', 'bic'}, 'diagnostics': {...}, 'summary': {...}}
      - eval_out: {'forecast_results': long DF, 'metrics': DF, 'summary': dict}
    """
    import numpy as np
    import pandas as pd

    print("="*60)
    print("                ECM FORECASTING RESULTS SUMMARY")
    print("="*60)

    # --- Cointegration bits ---
    coeffs = cointegration_results.get("coefficients", {})
    intercept   = float(coeffs.get("β0_intercept", coeffs.get("intercept", np.nan)))
    beta_L13    = float(coeffs.get("β1_L13", coeffs.get("L13", np.nan)))
    beta_trend  = float(coeffs.get("β2_trend", coeffs.get("trend", 0.0)))
    resid       = cointegration_results.get("residuals", [])
    resid_std   = float(np.nanstd(np.asarray(resid).ravel())) if len(resid) else np.nan
    r2          = cointegration_results.get("r_squared", np.nan)

    # --- ECM spec & diagnostics ---
    spec = ecm_results.get("specification", {})
    lags = spec.get("lags", {})
    p = lags.get("p"); q = lags.get("q"); r = lags.get("r")
    aic = spec.get("aic", np.nan); bic = spec.get("bic", np.nan)

    diag = ecm_results.get("diagnostics", {})
    gamma = float(diag.get("error_correction_coeff", np.nan))
    half_life = ecm_results.get("summary", {}).get("half_life", np.nan)
    if (not np.isfinite(half_life)) and (gamma < 0) and (gamma > -1):
        half_life = -np.log(2) / np.log(1 + gamma)

    metrics_df = eval_out["metrics"].copy()

    # --- Print cointegration ---
    print("\n📊 COINTEGRATING RELATIONSHIP")
    trend_txt = f" + {beta_trend:.4f}×trend" if np.isfinite(beta_trend) and abs(beta_trend) > 0 else ""
    print(f"  L1 = {intercept:.4f} + {beta_L13:.4f}×L13{trend_txt} + u_t")
    r2_txt = f"{r2:.4f}" if np.isfinite(r2) else "N/A"
    print(f"  R² = {r2_txt}  |  Residual σ = {resid_std:.4f}")

    # --- Print ECM spec ---
    print("\n⚙️  ECM MODEL SPECIFICATION")
    print(f"  Lags (p,q,r) = ({p},{q},{r})")
    print(f"  AIC = {aic:.3f}, BIC = {bic:.3f}")
    print(f"  γ (ECT coeff) = {gamma:.4f}  |  Half-life = {half_life if np.isfinite(half_life) else 'N/A'}")

    # --- Performance table (RMSE) ---
    print("\n📈 RMSE BY MODEL & HORIZON")
    rmse_table = metrics_df.pivot(index="model", columns="horizon", values="RMSE")
    # Order rows nicely if present
    order = [m for m in ["ECM","Random Walk","ARIMA(0,1,0)"] if m in rmse_table.index]
    rmse_table = rmse_table.loc[order] if order else rmse_table
    print(rmse_table.round(4).to_string())

    # --- Rankings per horizon ---
    print("\n🏆 RANKINGS BY HORIZON (lower RMSE is better)")
    for h in sorted(metrics_df["horizon"].unique()):
        sub = metrics_df[metrics_df["horizon"] == h].sort_values("RMSE")
        best_model = sub.iloc[0]["model"]; best_rmse = sub.iloc[0]["RMSE"]
        print(f"  H{h}: {best_model} (RMSE {best_rmse:.4f})")
        if "ECM" in sub["model"].values:
            ecm_rmse = float(sub[sub["model"] == "ECM"]["RMSE"].iloc[0])
            if np.isfinite(ecm_rmse) and np.isfinite(best_rmse):
                diff = (ecm_rmse - best_rmse) / best_rmse * 100
                tag = "ahead of" if diff < 0 else "behind"
                print(f"      ECM is {abs(diff):.1f}% {tag} the best model")

    print("\n✅ DISPLAY COMPLETE")
    return rmse_table

def _display_plots(aligned_data, cointegration_results, ecm_results, eval_out):
    """
    Inline plots only (no saving). Uses:
      eval_out['forecast_results'] long DF with columns:
        ['date','horizon','y_actual','ecm_forecast','rw_forecast','arima_forecast', ...]
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    print("="*60)
    print("                GENERATING ECM VISUALIZATIONS")
    print("="*60)

    forecast_df = eval_out["forecast_results"].copy()

    # Ensure datetime for nice plotting
    if "date" in forecast_df.columns:
        forecast_df["date"] = pd.to_datetime(forecast_df["date"])

    # 1) Actual vs Predicted (time series) per horizon
    print("\n📈 Actual vs Predicted (time series) per horizon")
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.ravel()
    horizons = sorted(forecast_df["horizon"].unique())
    for idx, h in enumerate([1,2,3,4]):
        ax = axes[idx]
        sub = forecast_df[forecast_df["horizon"] == h].copy()
        if len(sub) == 0:
            ax.set_visible(False); continue
        sub = sub.sort_values("date")
        ax.plot(sub["date"], sub["y_actual"], marker="o", linewidth=1.5, label="Actual")
        if "ecm_forecast" in sub.columns:
            ax.plot(sub["date"], sub["ecm_forecast"], marker="^", linewidth=1.5, label="ECM")
        if "rw_forecast" in sub.columns:
            ax.plot(sub["date"], sub["rw_forecast"], marker="s", linewidth=1.0, label="RW", alpha=0.7)
        if "arima_forecast" in sub.columns:
            ax.plot(sub["date"], sub["arima_forecast"], marker="x", linewidth=1.0, label="ARIMA", alpha=0.7)
        ax.set_title(f"H{h}"); ax.grid(alpha=0.3); ax.legend(fontsize=9)
        ax.tick_params(axis="x", labelrotation=45)
    plt.tight_layout(); plt.show()

    # 2) Actual vs Predicted scatter per horizon (ECM)
    print("\n🔍 Actual vs Predicted (scatter) per horizon — ECM")
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.ravel()
    for idx, h in enumerate([1,2,3,4]):
        ax = axes[idx]
        sub = forecast_df[forecast_df["horizon"] == h].copy()
        if len(sub) == 0 or "ecm_forecast" not in sub.columns:
            ax.set_visible(False); continue
        y_true = sub["y_actual"].values
        y_pred = sub["ecm_forecast"].values
        ax.scatter(y_true, y_pred, alpha=0.7)
        lo = float(np.nanmin([y_true.min(), y_pred.min()]))
        hi = float(np.nanmax([y_true.max(), y_pred.max()]))
        ax.plot([lo, hi], [lo, hi], linestyle="--")
        ax.set_title(f"ECM H{h}: Actual vs Predicted")
        ax.set_xlabel("Actual"); ax.set_ylabel("Predicted"); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.show()

    # 3) Residual diagnostics (ECM residuals)
    print("\n🔬 Residual diagnostics (ECM final model)")
    resid = ecm_results.get("diagnostics", {}).get("residuals", None)
    if resid is not None:
        r = np.asarray(resid).ravel()
        fig, axes = plt.subplots(2, 2, figsize=(14, 11))
        # Time series
        axes[0,0].plot(r); axes[0,0].axhline(0, linestyle="--"); axes[0,0].grid(alpha=0.3)
        axes[0,0].set_title("Residuals (Time Series)")
        # Histogram
        axes[0,1].hist(r, bins=20, density=True, alpha=0.8); axes[0,1].grid(alpha=0.3)
        axes[0,1].set_title("Residuals Distribution")
        # QQ plot
        from scipy import stats as _stats
        _stats.probplot(r, dist="norm", plot=axes[1,0]); axes[1,0].set_title("Q-Q Plot (Normal)")
        # ACF
        from statsmodels.tsa.stattools import acf
        L = max(5, min(20, len(r)//4)); acf_vals = acf(r, nlags=L, fft=True)
        axes[1,1].bar(range(len(acf_vals)), acf_vals)
        ci = 1.96/np.sqrt(len(r))
        axes[1,1].axhline(0, color="k"); axes[1,1].axhline(ci, linestyle="--"); axes[1,1].axhline(-ci, linestyle="--")
        axes[1,1].set_title("ACF of Residuals"); axes[1,1].grid(alpha=0.3)
        plt.tight_layout(); plt.show()
    else:
        print("   (No ECM residuals found.)")

    print("\n✅ VISUALS COMPLETE")
    return None

def _save_outputs(aligned_data, cointegration_results, ecm_results, eval_out, output_root="outputs/ecm_pipeline"):
    """Save key pipeline artifacts to disk and return the output directory."""
    output_root = Path(output_root)
    timestamp_dir = output_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp_dir.mkdir(parents=True, exist_ok=True)

    if aligned_data is not None:
        aligned_path = timestamp_dir / "aligned_data.csv"
        aligned_data.to_csv(aligned_path, index=False)

    if eval_out is not None:
        forecast_df = eval_out.get('forecast_results')
        metrics_df = eval_out.get('metrics')
        if isinstance(forecast_df, pd.DataFrame):
            forecast_df.to_csv(timestamp_dir / "forecast_results.csv", index=False)
        if isinstance(metrics_df, pd.DataFrame):
            metrics_df.to_csv(timestamp_dir / "forecast_metrics.csv", index=False)

    summary_payload = {
        'cointegration_results': cointegration_results,
        'ecm_results': ecm_results,
        'evaluation_summary': eval_out.get('summary') if isinstance(eval_out, dict) else None,
    }

    summary_path = timestamp_dir / "pipeline_summary.json"
    with summary_path.open('w', encoding='utf-8') as fh:
        json.dump(summary_payload, fh, default=_json_serializer, indent=2)

    return timestamp_dir


def _json_serializer(value):
    """Best-effort serializer for numpy/pandas objects."""
    if isinstance(value, (pd.Series, pd.Index)):
        return value.tolist()
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient='records')
    try:
        import numpy as _np
        if isinstance(value, _np.ndarray):
            return value.tolist()
        if isinstance(value, _np.generic):
            return value.item()
    except ImportError:
        pass
    if hasattr(value, 'isoformat'):
        return value.isoformat()
    return str(value)

__all__ = ['_display_results', '_display_plots', '_save_outputs']
