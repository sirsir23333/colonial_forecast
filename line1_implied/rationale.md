Line 1 ECM Rationale
=====================

Overview
--------
This note documents why the `line1_implied` package uses a cointegration/ECM
framework instead of a simple OLS model, and what each stage contributes to the
pipeline. The context is Colonial Pipeline Line 1 transit-time forecasting, where
we observe multiple related series (Line 1, Line 3, Line 13) with overlapping
periods and obvious non-stationarity.

Why Cointegration Matters
-------------------------
* Transit times display level shifts and long-memory behaviour; classical unit
  root tests (ADF/KPSS) fail to reject the presence of unit roots in raw levels.
* Despite the individual series being non-stationary, Line 1 and Line 13 share a
  stable long-run relationship: when Line 13 slows or accelerates, Line 1 moves
  proportionally after a lag. Engle–Granger cointegration tests produce a
  stationary residual, validating the equilibrium.
* Running plain OLS on two non-stationary series would risk spurious regression:
  high R² and significant coefficients that merely reflect common trends rather
  than economic linkage. Cointegration allows us to keep the long-run signal
  while modelling deviations explicitly.

Why an Error-Correction Model (ECM)
-----------------------------------
* Once the long-run equilibrium is identified, we need a dynamic model that
  drives short-run adjustments back towards that equilibrium. The ECM captures:
  - Short-run differenced terms (ΔL1, ΔL13, ΔL3) for immediate shocks
  - The lagged residual `u_{t-1}` signalling how far the system is from
    equilibrium
  - A negative γ coefficient that quantifies mean-reversion speed; in our case
    γ ≈ -0.53 implies a half-life of ~1.3 periods.
* Forecasting with the ECM respects both short-run volatility and long-run
  tethering, outperforming Random Walk/ARIMA baselines—especially for multi-step
  horizons where pure differencing models drift quickly.

Stage-by-Stage Rationale
------------------------
1. `preparation._prepare_aligned_data`
   * Aligns Line 1/3/13 data on the intersection of available dates and adds a
     deterministic trend. Using a common sample avoids the need for imputation
     and keeps the cointegration test valid.
2. `stationarity.analyze_pipeline_stationarity`
   * Confirms that raw levels are non-stationary (ADF high p-values, KPSS rejects)
     but differenced series pass stationarity checks. This justifies differencing
     inside the ECM and ensures inference is reliable.
3. `cointegration._estimate_cointegrating_relation`
   * Fits the long-run OLS relationship `L1_t = β0 + β1 L13_t + β2 trend_t + u_t`
     and outputs residuals. Diagnostics (Durbin–Watson, IV tests) verify the
     residual is well-behaved; the residual series becomes the equilibrium error.
4. `ecm._build_ecm_model`
   * Searches lag orders (p, q, r) using an information criterion (BIC) and fits
     the short-run ECM with HAC covariance to account for heteroskedasticity and
     autocorrelation. Diagnostic suite (Ljung-Box, Breusch–Pagan, Jarque–Bera)
     ensures assumptions hold.
5. `forecast._forecast_evaluation`
   * Uses a rolling window to generate 1–4 step forecasts, recalibrating the ECM
     at each origin. Baselines (Random Walk, ARIMA(0,1,0)) provide comparison so
     we can report improvements and avoid a false sense of precision.
6. `reporting._display_results/_display_plots/_save_outputs`
   * Present key metrics (RMSE, MAE, MAPE, improvement vs RW) and persist
     aligned datasets plus evaluation results for auditability.
7. `run_all.run_pipeline_complete`
   * Orchestrates the workflow, captures metadata (execution time, best horizon),
     and returns a structured result dict so notebooks/scripts can consume it
     without re-running the full notebook logic.

When to Consider Alternatives
-----------------------------
* If Line 3 transit times become cointegrated with Line 1 (e.g., after structural
  changes), extend the cointegration step to include multiple regressors or adopt
  a VECM.
* If residual diagnostics flag heavy autocorrelation or heteroskedasticity that
  HAC adjustments cannot correct, consider adding exogenous break dummies or
  switching to regime-switching models.
* When stationarity improves (e.g., through de-seasonalisation) and cointegration
  fails, a VAR in differences may be more appropriate—ECM is only justified while
  a long-run equilibrium holds.

Key Takeaways
-------------
* Cointegration lets us retain the economic link between Lines 1 and 13 without
  falling into spurious regression traps.
* The ECM decomposes movements into persistent deviations versus short-run noise,
  enabling faster, more stable forecasts that revert toward realistic levels.
* Modularising the workflow (data prep → stationarity checks → cointegration →
  ECM → evaluation → reporting) keeps the notebook clean and allows reuse from
  scripts or batch jobs.
