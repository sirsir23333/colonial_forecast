"""Forecast evaluation routines for ECM pipeline."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
import warnings

def _forecast_evaluation(
    aligned_data,
    cointegration_results,
    ecm_results,
    n_test=26,
    exog_nowcast="ma",       # "ma" (moving average) or "ar1"
    exog_ma_lookback=3       # lookback window if exog_nowcast == "ma"
):
    """
    Comprehensive ECM forecast evaluation with rolling window methodology.

    Performs 1-4 step ahead out-of-sample forecasts using an Error Correction Model (ECM),
    aligned with the estimation spec even when contemporaneous exogenous differences
    (ΔL13_t, ΔL3_t) appear on the RHS.

    Parameters
    ----------
    aligned_data : pd.DataFrame
        Must contain columns ['Date','L1','L3','L13'] and (optionally) 'trend'
    cointegration_results : dict
        Must contain cointegrating coefficients under cointegration_results['coefficients']
        with keys like 'β0_intercept', 'β1_L13', 'β2_trend' (trend is optional)
    ecm_results : dict
        Fitted ECM results (only used to read chosen lags (p,q,r), no refit required)
    n_test : int
        Out-of-sample size
    exog_nowcast : {"ma", "ar1"}
        How to nowcast ΔL13_{t+1}, ΔL3_{t+1} for horizon h=1 when the ECM uses contemporaneous exogs.
    exog_ma_lookback : int
        Window for moving-average nowcast of exog differences.

    Returns
    -------
    dict with:
        - 'forecast_results' : DataFrame (date, horizon, y_actual, ecm_forecast, rw_forecast, arima_forecast, forecast_origin)
        - 'metrics'          : DataFrame of RMSE/MAE/MAPE by model and horizon
        - 'plots'            : None (placeholder)
        - 'summary'          : Dict with best-per-horizon RMSE and counts
    """
    print("🔮 STARTING ECM FORECAST EVALUATION")
    print("="*60)

    # Imports
    import numpy as np
    import pandas as pd
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from scipy import stats   # noqa: F401 (kept for parity with earlier code)
    import statsmodels.api as sm
    from statsmodels.tsa.arima.model import ARIMA
    import warnings
    warnings.filterwarnings('ignore')

    # --- Helpers ---
    def mape(y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    def _safe_ma_of_diff(series, lookback=3):
        d = series.diff().dropna()
        if len(d) == 0:
            return 0.0
        lookback = max(1, min(lookback, len(d)))
        return float(d.tail(lookback).mean())

    def _safe_ar1_of_diff(series):
        d = series.diff().dropna()
        if len(d) == 0:
            return 0.0
        # AR(1) one-step nowcast: phi * last_diff, where phi is lag-1 autocorr
        last = float(d.iloc[-1])
        phi = float(d.autocorr(lag=1)) if len(d) > 1 else 0.0
        phi = 0.0 if np.isnan(phi) else phi
        return phi * last

    # --- Data overview ---
    print(f"📊 Data Overview:")
    print(f"  • Total observations: {len(aligned_data)}")
    print(f"  • Test period: {n_test} observations")
    print(f"  • Training period: {len(aligned_data) - n_test} observations")

    if len(aligned_data) < n_test + 10:
        raise ValueError(f"Insufficient data: need at least {n_test + 10} observations")

    # Index & trend handling
    if 'Date' in aligned_data.columns:
        dates = aligned_data['Date']
        aligned_data = aligned_data.set_index('Date')
    else:
        dates = aligned_data.index
        if not isinstance(dates, (pd.DatetimeIndex, pd.PeriodIndex)):
            dates = pd.period_range(start='2020-01', periods=len(aligned_data), freq='M')
            aligned_data.index = dates

    if 'trend' not in aligned_data.columns:
        aligned_data['trend'] = np.arange(len(aligned_data))

    # Extract cointegration coefficients (with sensible fallbacks)
    beta = cointegration_results.get('coefficients', {})
    if not beta:
        raise ValueError("Cointegration coefficients not found in cointegration_results['coefficients'].")

    beta_intercept = float(beta.get('β0_intercept', beta.get('intercept', 0.0)))
    beta_L13       = float(beta.get('β1_L13', beta.get('L13', -1.0)))   # fallback sign only
    beta_trend     = float(beta.get('β2_trend', beta.get('trend', 0.0)))

    print(f"\n🧭 Cointegrating equation used for ECT:")
    trend_txt = f" + {beta_trend:.4f}*trend_t" if beta_trend != 0 else ""
    print(f"  L1_t = {beta_intercept:.4f} + {beta_L13:.4f}*L13_t{trend_txt} + u_t")

    # Read chosen lags from ECM results
    p_opt = int(ecm_results['specification']['lags']['p'])
    q_opt = int(ecm_results['specification']['lags']['q'])
    r_opt = int(ecm_results['specification']['lags']['r'])

    print(f"\n🎛️ ECM lag structure from training:")
    print(f"  • (p,q,r) = ({p_opt},{q_opt},{r_opt})")
    print("  • Equation: ΔL1_t = α + Σ φ_i ΔL1_{t-i} + Σ θ_j ΔL13_{t-j} + Σ ψ_m ΔL3_{t-m} + γ u_{t-1} + ε_t")

    horizons = [1, 2, 3, 4]
    n_train = len(aligned_data) - n_test
    forecast_results = []

    print(f"\n🎯 Rolling Window Forecast Setup:")
    print(f"  • Forecast origins: {n_test}")
    print(f"  • Forecast horizons: {horizons}")
    print(f"  • Exog nowcast method: {exog_nowcast.upper()}")

    printed_design_matrix = False

    # Iterate forecast origins; require enough room for max horizon
    for i in range(n_train, len(aligned_data) - max(horizons) + 1):
        origin_idx = i - 1
        origin_date = dates[origin_idx]
        date_str = origin_date.strftime('%Y-%m') if hasattr(origin_date, 'strftime') else str(origin_date)
        print(f"\rForecast origin: {date_str} ({i - n_train + 1}/{len(aligned_data) - max(horizons) + 1 - n_train})", end="")

        train_data = aligned_data.iloc[:i].copy()

        # Actual L1 values to verify at horizons
        y_actual = {}
        for h in horizons:
            target_idx = i + h - 1
            if target_idx < len(aligned_data):
                y_actual[h] = float(aligned_data['L1'].iloc[target_idx])

        # ----- ECM one- through four-step forecasts -----
        try:
            # Construct ECT from cointegration: u_t = L1_t - (β0 + β1 L13_t + β2 trend_t)
            ect = train_data['L1'] - (beta_intercept + beta_L13 * train_data['L13'] + beta_trend * train_data['trend'])

            # Differences for design matrix
            dL1  = train_data['L1'].diff()
            dL13 = train_data['L13'].diff()
            dL3  = train_data['L3'].diff()

            # Build in-sample ECM design matrix (matches selected (p,q,r))
            ecm_df = pd.DataFrame({'dL1': dL1, 'ECT_lag': ect.shift(1)})

            # ΔL1 lags
            for lag_i in range(1, p_opt + 1):
                ecm_df[f'dL1_lag{lag_i}'] = dL1.shift(lag_i)

            # ΔL13 terms
            for lag_j in range(0, q_opt + 1):
                if lag_j == 0:
                    ecm_df['dL13_lag0'] = dL13  # contemporaneous if q_opt==0
                else:
                    ecm_df[f'dL13_lag{lag_j}'] = dL13.shift(lag_j)

            # ΔL3 terms
            for lag_m in range(0, r_opt + 1):
                if lag_m == 0:
                    ecm_df['dL3_lag0'] = dL3  # contemporaneous if r_opt==0
                else:
                    ecm_df[f'dL3_lag{lag_m}'] = dL3.shift(lag_m)

            ecm_df = ecm_df.dropna()

            min_required = 1 + p_opt + max(q_opt, r_opt)
            if len(ecm_df) < min_required + 3:
                raise ValueError(f"Insufficient data after alignment: need at least {min_required + 3}, got {len(ecm_df)}")

            if not printed_design_matrix:
                expected_cols = ['dL1', 'ECT_lag']
                expected_cols.extend([f'dL1_lag{lag_i}' for lag_i in range(1, p_opt + 1)])
                expected_cols.extend([f'dL13_lag{lag_j}' for lag_j in range(0, q_opt + 1)])
                expected_cols.extend([f'dL3_lag{lag_m}'  for lag_m in range(0, r_opt + 1)])
                print("\n🔍 ECM Design Matrix Validation:")
                print(f"  • Expected columns: {expected_cols}")
                print(f"  • Actual columns:   {list(ecm_df.columns)}")
                print(f"  • (p,q,r) = ({p_opt},{q_opt},{r_opt})")
                print(f"  • Min required obs: {min_required + 3}, Available: {len(ecm_df)}")
                printed_design_matrix = True

            # Fit ECM (OLS is fine; HAC affects inference not point forecasts)
            y_ecm = ecm_df['dL1']
            X_cols = [c for c in ecm_df.columns if c != 'dL1']
            X_ecm  = sm.add_constant(ecm_df[X_cols])
            ecm_model = sm.OLS(y_ecm, X_ecm).fit()

            # Prepare paths for recursion
            current_L1    = float(train_data['L1'].iloc[-1])
            current_L13   = float(train_data['L13'].iloc[-1])
            current_L3    = float(train_data['L3'].iloc[-1])
            current_trend = float(train_data['trend'].iloc[-1])

            L1_path    = [current_L1]
            L13_path   = [current_L13]
            L3_path    = [current_L3]
            trend_path = [current_trend]

            dL1_path, dL13_path, dL3_path = [], [], []

            # Pre-compute h=1 exog-difference nowcasts if contemporaneous terms are present
            dL13_next = 0.0
            dL3_next  = 0.0
            if q_opt >= 0 and 'dL13_lag0' in X_ecm.columns:
                if exog_nowcast == "ma":
                    dL13_next = _safe_ma_of_diff(train_data['L13'], lookback=exog_ma_lookback)
                elif exog_nowcast == "ar1":
                    dL13_next = _safe_ar1_of_diff(train_data['L13'])
            if r_opt >= 0 and 'dL3_lag0' in X_ecm.columns:
                if exog_nowcast == "ma":
                    dL3_next = _safe_ma_of_diff(train_data['L3'], lookback=exog_ma_lookback)
                elif exog_nowcast == "ar1":
                    dL3_next = _safe_ar1_of_diff(train_data['L3'])

            ecm_forecasts = {}

            for h in horizons:
                if h not in y_actual:
                    continue

                # Compute ECT at t+h-1 using level paths
                ect_h = L1_path[h-1] - (beta_intercept + beta_L13 * L13_path[h-1] + beta_trend * trend_path[h-1])

                # Build forecast feature vector aligned to training design
                X_forecast = {'const': 1.0, 'ECT_lag': ect_h}

                # ΔL1 lags from history / recursion
                for lag_i in range(1, p_opt + 1):
                    if h - 1 - lag_i >= 0 and len(dL1_path) >= lag_i:
                        X_forecast[f'dL1_lag{lag_i}'] = dL1_path[h - 1 - lag_i]
                    else:
                        # pull from actual history for first step(s)
                        if len(train_data) >= lag_i + 1:
                            # ΔL1_{t - (lag_i-1)} = L1_{t - (lag_i-1)} - L1_{t - lag_i}
                            v = float(train_data['L1'].iloc[-(lag_i)] - train_data['L1'].iloc[-(lag_i + 1)])
                            X_forecast[f'dL1_lag{lag_i}'] = v
                        else:
                            X_forecast[f'dL1_lag{lag_i}'] = 0.0

                # ΔL13 terms
                for lag_j in range(0, q_opt + 1):
                    col = f'dL13_lag{lag_j}' if lag_j > 0 else 'dL13_lag0'
                    if lag_j == 0:
                        # contemporaneous term: needs a nowcast at h=1, else zero by assumption
                        X_forecast[col] = dL13_next if h == 1 else 0.0
                    else:
                        # use recursive dL13 path if you later model it; for now zero under RW
                        X_forecast[col] = 0.0

                # ΔL3 terms
                for lag_m in range(0, r_opt + 1):
                    col = f'dL3_lag{lag_m}' if lag_m > 0 else 'dL3_lag0'
                    if lag_m == 0:
                        X_forecast[col] = dL3_next if h == 1 else 0.0
                    else:
                        X_forecast[col] = 0.0

                # Predict ΔL1_{t+h}
                X_forecast_df = pd.DataFrame([X_forecast], columns=X_ecm.columns)
                dL1_forecast = float(ecm_model.predict(X_forecast_df).iloc[0])

                # Update paths
                dL1_path.append(dL1_forecast)
                L1_next = L1_path[h-1] + dL1_forecast
                L1_path.append(L1_next)

                # Keep exogs RW in levels (no change); Δ paths zero except h=1 which we already used
                L13_path.append(L13_path[h-1])
                L3_path.append(L3_path[h-1])
                trend_path.append(trend_path[h-1] + 1)

                dL13_path.append(0.0)
                dL3_path.append(0.0)

                ecm_forecasts[h] = L1_next

        except Exception as e:
            print(f"\nECM forecast error at {date_str}: {e}")
            # Fallback: persistence
            ecm_forecasts = {h: float(train_data['L1'].iloc[-1]) for h in horizons if h in y_actual}

        # Baselines
        rw_forecasts = {h: float(train_data['L1'].iloc[-1]) for h in y_actual}

        try:
            arima_model = ARIMA(train_data['L1'], order=(0,1,0), trend='c')
            arima_fit = arima_model.fit()
            arima_vals = arima_fit.forecast(steps=max(y_actual.keys()))
            arima_forecasts = {h: float(arima_vals[h-1]) for h in y_actual}
        except Exception:
            arima_forecasts = rw_forecasts.copy()

        # Store rows
        for h in y_actual:
            target_idx = i + h - 1
            forecast_results.append({
                'date': dates[target_idx],
                'horizon': h,
                'y_actual': y_actual[h],
                'ecm_forecast': ecm_forecasts.get(h, float(train_data['L1'].iloc[-1])),
                'rw_forecast': rw_forecasts[h],
                'arima_forecast': arima_forecasts[h],
                'forecast_origin': origin_date
            })

    print("\n✅ Rolling window forecasts completed!")

    # Results -> DataFrame
    forecast_df = pd.DataFrame(forecast_results)
    if len(forecast_df) == 0:
        raise ValueError("No forecasts generated - check data and parameters")

    print("\n📈 Forecast Results Summary:")
    print(f"  • Total forecasts generated: {len(forecast_df)}")
    print(f"  • By horizon: {dict(forecast_df.groupby('horizon').size())}")

    # Metrics
    print("\n📊 Computing performance metrics...")
    models = ['ecm_forecast', 'rw_forecast', 'arima_forecast']
    model_names = ['ECM', 'Random Walk', 'ARIMA(0,1,0)']

    metrics_list = []
    for h in horizons:
        sub = forecast_df[forecast_df['horizon'] == h]
        if len(sub) == 0:
            continue
        y_true = sub['y_actual'].values
        for col, name in zip(models, model_names):
            y_pred = sub[col].values
            rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            mae  = float(mean_absolute_error(y_true, y_pred))
            mape_val = float(mape(y_true, y_pred))
            metrics_list.append({'model': name, 'horizon': h, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape_val})

    metrics_df = pd.DataFrame(metrics_list)

    # Best by horizon
    summary_list = []
    for h in horizons:
        msub = metrics_df[metrics_df['horizon'] == h]
        if len(msub) == 0:
            continue
        idx = msub['RMSE'].idxmin()
        summary_list.append({
            'horizon': h,
            'best_model_rmse': msub.loc[idx, 'model'],
            'best_rmse': float(msub.loc[idx, 'RMSE'])
        })

    summary_dict = {
        'best_by_horizon': summary_list,
        'total_forecasts': int(len(forecast_df)),
        'horizons_evaluated': horizons,
        'exog_nowcast': exog_nowcast,
        'exog_ma_lookback': exog_ma_lookback
    }

    print("\n🏆 Best models by horizon (RMSE):")
    for item in summary_list:
        print(f"  • Horizon {item['horizon']}: {item['best_model_rmse']} (RMSE: {item['best_rmse']:.4f})")

    return {
        'forecast_results': forecast_df,
        'metrics': metrics_df,
        'plots': None,
        'summary': summary_dict
    }

__all__ = ['_forecast_evaluation']
