"""Top-level orchestrator for the ECM forecasting pipeline.

This module wires the feature-specific helpers together in the following order:
1. line1_implied.preparation._prepare_aligned_data – align the input series.
2. line1_implied.cointegration._estimate_cointegrating_relation – fit the long-run relation.
3. line1_implied.ecm._build_ecm_model – build the short-run ECM.
4. line1_implied.forecast._forecast_evaluation – roll forecasts and metrics.
5. line1_implied.reporting._display_results/_display_plots/_save_outputs – present and persist outputs.
"""

from datetime import datetime
import time

import numpy as np
import pandas as pd

from .preparation import _prepare_aligned_data
from .cointegration import _estimate_cointegrating_relation
from .ecm import _build_ecm_model
from .forecast import _forecast_evaluation
from .reporting import _display_results, _display_plots, _save_outputs


def run_pipeline_complete(
    pipeline_data,
    correlation_results,
    df_or_path=None,
    n_test=26,
    exog_nowcast="ma",
    exog_ma_lookback=3,
    save_outputs=True,
    display_results=True,
):
    """Run the full Colonial ECM workflow using the helper modules listed above."""

    if pipeline_data is None or correlation_results is None:
        raise ValueError("'pipeline_data' and 'correlation_results' must be provided.")

    if df_or_path is not None:
        raise NotImplementedError("Loading from external file not yet implemented.")

    if n_test < 1:
        raise ValueError(f"n_test must be positive, got {n_test}")

    start_time = time.time()

    print("🚀" + "=" * 80)
    print("           EXECUTING COMPLETE ECM FORECASTING PIPELINE")
    print("🚀" + "=" * 80)

    pipeline_results = {
        "aligned_data": None,
        "cointegration_results": None,
        "ecm_results": None,
        "forecast_evaluation": None,
        "key_metrics": {},
        "model_summary": {},
        "execution_metadata": {
            "start_time": datetime.now().isoformat(),
            "parameters": {
                "n_test": n_test,
                "exog_nowcast": exog_nowcast,
                "exog_ma_lookback": exog_ma_lookback,
                "save_outputs": save_outputs,
                "display_results": display_results,
            },
            "status": "initialized",
        },
    }

    try:
        print("\n📋 STEP 1: DATA ALIGNMENT AND PREPARATION")
        print("=" * 60)
        aligned_data = _prepare_aligned_data(pipeline_data, correlation_results)
        if aligned_data is None or aligned_data.empty:
            raise ValueError("Aligned data is empty; cannot continue.")
        pipeline_results["aligned_data"] = aligned_data

        print("\n🔍 STEP 2: COINTEGRATING RELATIONSHIP ESTIMATION")
        print("=" * 60)
        cointegration_results = _estimate_cointegrating_relation(aligned_data)
        pipeline_results["cointegration_results"] = cointegration_results

        beta1 = cointegration_results.get("coefficients", {}).get("β1_L13")
        beta2 = cointegration_results.get("coefficients", {}).get("β2_trend")
        if beta1 is not None and beta2 is not None:
            print(f"✅ Cointegrating relationship: β₁={beta1:.4f}, β₂={beta2:.4f}")

        print("\n⚙️  STEP 3: ECM MODEL BUILDING")
        print("=" * 60)
        ecm_results = _build_ecm_model(
            aligned_data=aligned_data,
            cointegration_results=cointegration_results,
            exog_nowcast=exog_nowcast,
            exog_ma_lookback=exog_ma_lookback,
        )
        pipeline_results["ecm_results"] = ecm_results

        ecm_gamma = (
            ecm_results.get("diagnostics", {}).get("error_correction_coeff", np.nan)
        )
        chosen_lags = ecm_results.get("specification", {}).get("lags", {})
        print(
            "✅ ECM model: γ={:.4f}, lags={}".format(
                ecm_gamma if np.isfinite(ecm_gamma) else float("nan"), chosen_lags
            )
        )

        print("\n📈 STEP 4: FORECAST EVALUATION")
        print("=" * 60)
        eval_out = _forecast_evaluation(
            aligned_data=aligned_data,
            cointegration_results=cointegration_results,
            ecm_results=ecm_results,
            n_test=n_test,
            exog_nowcast=exog_nowcast,
            exog_ma_lookback=exog_ma_lookback,
        )
        pipeline_results["forecast_evaluation"] = eval_out
        print("✅ Forecast evaluation complete")

        if display_results:
            print("\n📊 STEP 5: RESULTS DISPLAY")
            print("=" * 60)
            try:
                _display_results(cointegration_results, ecm_results, eval_out)
                _display_plots(aligned_data, cointegration_results, ecm_results, eval_out)
                print("✅ Results display complete")
            except Exception as err:
                print(f"⚠️  Display warning: {err}")

        if save_outputs:
            print("\n💾 STEP 6: OUTPUT SAVING")
            print("=" * 60)
            try:
                output_dir = _save_outputs(
                    aligned_data, cointegration_results, ecm_results, eval_out
                )
                pipeline_results["execution_metadata"]["output_directory"] = str(output_dir)
                print(f"✅ Outputs saved to: {output_dir}")
            except Exception as err:
                print(f"⚠️  Save warning: {err}")

        print("\n📋 STEP 7: COMPILING KEY METRICS")
        print("=" * 60)
        key_metrics = {
            "beta1": cointegration_results.get("coefficients", {}).get("β1_L13", np.nan),
            "beta2": cointegration_results.get("coefficients", {}).get("β2_trend", np.nan),
            "r_squared": cointegration_results.get("r_squared", np.nan),
            "gamma": ecm_gamma,
            "chosen_lags": chosen_lags,
            "half_life": (
                -np.log(2) / np.log(1 + ecm_gamma)
                if np.isfinite(ecm_gamma) and ecm_gamma < 0
                else np.nan
            ),
            "n_observations": len(aligned_data),
            "n_test_periods": n_test,
        }

        metrics_df = eval_out.get("metrics") if isinstance(eval_out, dict) else None
        improvements = []
        if isinstance(metrics_df, pd.DataFrame) and not metrics_df.empty:
            for horizon in sorted(metrics_df["horizon"].unique()):
                ecm_row = metrics_df[
                    (metrics_df["model"] == "ECM") & (metrics_df["horizon"] == horizon)
                ]
                rw_row = metrics_df[
                    (metrics_df["model"] == "Random Walk")
                    & (metrics_df["horizon"] == horizon)
                ]

                ecm_rmse = float(ecm_row["RMSE"].iloc[0]) if not ecm_row.empty else np.nan
                rw_rmse = float(rw_row["RMSE"].iloc[0]) if not rw_row.empty else np.nan

                key_metrics[f"ecm_rmse_h{horizon}"] = ecm_rmse
                key_metrics[f"rw_rmse_h{horizon}"] = rw_rmse

                if np.isfinite(rw_rmse) and not np.isclose(rw_rmse, 0):
                    improvement = (rw_rmse - ecm_rmse) / rw_rmse * 100.0
                    key_metrics[f"improvement_vs_rw_h{horizon}"] = improvement
                    improvements.append(improvement)
                else:
                    key_metrics[f"improvement_vs_rw_h{horizon}"] = np.nan

        pipeline_results["key_metrics"] = key_metrics

        avg_improvement = np.nanmean(improvements) if improvements else np.nan
        best_horizon = None
        if improvements:
            horizon_improvements = {
                h: key_metrics.get(f"improvement_vs_rw_h{h}", np.nan)
                for h in [1, 2, 3, 4]
            }
            finite_improvements = {
                h: val for h, val in horizon_improvements.items() if np.isfinite(val)
            }
            if finite_improvements:
                best_horizon = max(finite_improvements, key=finite_improvements.get)

        pipeline_results["model_summary"] = {
            "model_type": "Error Correction Model (ECM)",
            "cointegration_detected": bool(np.isfinite(ecm_gamma) and abs(ecm_gamma) > 0.1),
            "avg_improvement_vs_random_walk": avg_improvement,
            "best_horizon": best_horizon,
        }

        end_time = time.time()
        execution_time = end_time - start_time

        pipeline_results["execution_metadata"].update(
            {
                "end_time": datetime.now().isoformat(),
                "execution_time_seconds": execution_time,
                "status": "completed_successfully",
            }
        )

        print("\n🎉" + "=" * 80)
        print("           🎯 ECM PIPELINE EXECUTION COMPLETED 🎯")
        print("🎉" + "=" * 80)
        print(f"\n⏱️  EXECUTION SUMMARY:")
        print(f"   • Total time: {execution_time:.2f} seconds")
        print(f"   • Data points: {len(aligned_data)}")
        print(f"   • Test periods: {n_test}")
        print(
            f"   • Error correction speed: {ecm_gamma:.3f}"
            if np.isfinite(ecm_gamma)
            else "   • Error correction speed: N/A"
        )
        if np.isfinite(avg_improvement):
            print(f"   • Average improvement vs RW: {avg_improvement:.1f}%")
        else:
            print("   • Average improvement vs RW: N/A")
        print("\n✅ Pipeline completed successfully!")

    except Exception as err:
        print(f"❌ PIPELINE ERROR: {err}")
        pipeline_results["execution_metadata"]["status"] = "failed"
        pipeline_results["execution_metadata"]["error"] = str(err)
        import traceback

        traceback.print_exc()

    return pipeline_results

__all__ = ["run_pipeline_complete"]
