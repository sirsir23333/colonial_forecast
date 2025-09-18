"""Cointegration estimation utilities."""

import numpy as np
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from scipy import stats
import matplotlib.pyplot as plt

def _estimate_cointegrating_relation(aligned_data):
    """
    Estimate the long-run cointegrating relationship between L1 and L13 transit times.

    This function estimates the cointegrating equation: L1_t = β0 + β1*L13_t + β2*trend_t + u_t
    using OLS regression, where the residuals (u_t) serve as the error-correction term for 
    subsequent ECM modeling.

    Parameters:
    -----------
    aligned_data : pd.DataFrame
        DataFrame with columns ['Date', 'L1', 'L3', 'L13', 'trend'] containing aligned time series

    Returns:
    --------
    dict
        Dictionary containing:
        - 'model': fitted statsmodels OLS model object
        - 'residuals': error-correction term (u_t) for ECM
        - 'coefficients': dict with β0 (intercept), β1 (L13 coef), β2 (trend coef)
        - 'fitted_values': predicted L1 values from the cointegrating relationship
        - 'summary': comprehensive model diagnostics and interpretation
        - 'r_squared': coefficient of determination
        - 'f_statistic': F-test results for overall model significance
        - 'durbin_watson': test for serial correlation in residuals

    Notes:
    ------
    - Uses statsmodels OLS for robust regression estimation
    - Includes comprehensive model diagnostics and statistical tests
    - Provides economic interpretation of the cointegrating relationship
    - Handles missing data and validates input assumptions
    - Creates diagnostic plots for model validation
    """

    import numpy as np
    import statsmodels.api as sm
    from statsmodels.stats.stattools import durbin_watson
    from scipy import stats
    import matplotlib.pyplot as plt
    # (seaborn import optional)

    print("🔍 Estimating Long-Run Cointegrating Relationship")
    print("="*65)
    print("Model: L1_t = β0 + β1*L13_t + β2*trend_t + u_t")
    print()

    # Input validation
    required_cols = ['L1', 'L13', 'trend']
    missing_cols = [col for col in required_cols if col not in aligned_data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Check for missing values
    analysis_data = aligned_data[required_cols].copy()
    if analysis_data.isnull().any().any():
        print("⚠️  Warning: Missing values detected, removing incomplete observations")
        analysis_data = analysis_data.dropna()
        print(f"   Observations after cleaning: {len(analysis_data)}")

    if len(analysis_data) < 10:
        raise ValueError(f"Insufficient observations ({len(analysis_data)}) for reliable estimation")

    print(f"📊 Data Summary:")
    print(f"   • Observations: {len(analysis_data)}")
    print(f"   • L1 range: [{analysis_data['L1'].min():.2f}, {analysis_data['L1'].max():.2f}] days")
    print(f"   • L13 range: [{analysis_data['L13'].min():.2f}, {analysis_data['L13'].max():.2f}] days")
    print(f"   • Trend range: [{analysis_data['trend'].min()}, {analysis_data['trend'].max()}]")
    print()

    # Prepare regression data
    y = analysis_data['L1']  # Dependent variable: Line 1 transit times
    X = analysis_data[['L13', 'trend']]  # Independent variables: Line 13 + trend
    X = sm.add_constant(X)  # Add intercept term

    # Estimate OLS model
    try:
        model = sm.OLS(y, X).fit()

        # Extract key results
        coefficients = {
            'β0_intercept': model.params['const'],
            'β1_L13': model.params['L13'], 
            'β2_trend': model.params['trend']
        }

        residuals = model.resid
        fitted_values = model.fittedvalues

        # Calculate additional diagnostics
        r_squared = model.rsquared
        adj_r_squared = model.rsquared_adj
        f_stat = model.fvalue
        f_pvalue = model.f_pvalue
        dw_stat = durbin_watson(residuals)

        # Model summary statistics
        summary_stats = {
            'n_observations': len(analysis_data),
            'r_squared': r_squared,
            'adj_r_squared': adj_r_squared,
            'f_statistic': f_stat,
            'f_pvalue': f_pvalue,
            'durbin_watson': dw_stat,
            'aic': model.aic,
            'bic': model.bic,
            'log_likelihood': model.llf
        }

        print("📈 Model Estimation Results:")
        print("-" * 40)
        print(f"   L1_t = {coefficients['β0_intercept']:.4f} + {coefficients['β1_L13']:.4f}*L13_t + {coefficients['β2_trend']:.4f}*trend_t + u_t")
        print()
        print("📊 Coefficient Estimates:")
        print(f"   • β0 (Intercept): {coefficients['β0_intercept']:.4f} ± {model.bse['const']:.4f}")
        print(f"     - t-stat: {model.tvalues['const']:.3f}, p-value: {model.pvalues['const']:.4f}")
        print(f"   • β1 (L13 coefficient): {coefficients['β1_L13']:.4f} ± {model.bse['L13']:.4f}")
        print(f"     - t-stat: {model.tvalues['L13']:.3f}, p-value: {model.pvalues['L13']:.4f}")
        print(f"   • β2 (Trend coefficient): {coefficients['β2_trend']:.4f} ± {model.bse['trend']:.4f}")
        print(f"     - t-stat: {model.tvalues['trend']:.3f}, p-value: {model.pvalues['trend']:.4f}")
        print()

        # Economic interpretation
        print("💡 Economic Interpretation:")
        print("-" * 40)
        if model.pvalues['L13'] < 0.05:
            direction = "increases" if coefficients['β1_L13'] > 0 else "decreases"
            print(f"   • L13 Effect: 1-day increase in L13 transit time {direction} L1 by {abs(coefficients['β1_L13']):.3f} days")
            print(f"     (Statistically significant at 5% level)")
        else:
            print(f"   • L13 Effect: No statistically significant relationship detected")

        if model.pvalues['trend'] < 0.05:
            trend_direction = "upward" if coefficients['β2_trend'] > 0 else "downward"
            annual_trend = coefficients['β2_trend'] * 365.25  # Approximate annual effect
            print(f"   • Time Trend: {trend_direction} trend of {coefficients['β2_trend']:.4f} days per period")
            print(f"     (Approximately {annual_trend:.2f} days per year)")
        else:
            print(f"   • Time Trend: No significant long-term trend detected")
        print()

        # Model diagnostics
        print("🔬 Model Diagnostics:")
        print("-" * 40)
        print(f"   • R-squared: {r_squared:.4f} ({r_squared*100:.1f}% of variation explained)")
        print(f"   • Adjusted R-squared: {adj_r_squared:.4f}")
        print(f"   • F-statistic: {f_stat:.3f} (p-value: {f_pvalue:.6f})")
        print(f"   • Durbin-Watson: {dw_stat:.3f}", end="")
        if 1.5 <= dw_stat <= 2.5:
            print(" (No significant autocorrelation)")
        elif dw_stat < 1.5:
            print(" (Possible positive autocorrelation)")
        else:
            print(" (Possible negative autocorrelation)")
        print(f"   • AIC: {model.aic:.2f}, BIC: {model.bic:.2f}")
        print()

        # Model significance assessment
        if f_pvalue < 0.01:
            significance_level = "highly significant (p < 0.01)"
        elif f_pvalue < 0.05:
            significance_level = "significant (p < 0.05)"
        elif f_pvalue < 0.10:
            significance_level = "marginally significant (p < 0.10)"
        else:
            significance_level = "not significant (p ≥ 0.10)"

        print(f"🎯 Overall Model Assessment:")
        print(f"   • Model is {significance_level}")
        print(f"   • Explains {r_squared*100:.1f}% of L1 transit time variation")

        # ECM implications
        print(f"   • Error-correction term (u_t) ready for ECM modeling")
        print(f"   • Residuals represent deviations from long-run equilibrium")
        print()

        # Create diagnostic plots
        print("📊 Generating Diagnostic Plots...")
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Plot 1: Actual vs Fitted
        axes[0,0].scatter(fitted_values, y, alpha=0.7, color='blue')
        axes[0,0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', alpha=0.8)
        axes[0,0].set_xlabel('Fitted Values')
        axes[0,0].set_ylabel('Actual L1 Transit Days')
        axes[0,0].set_title('Actual vs Fitted Values')
        axes[0,0].grid(True, alpha=0.3)

        # Add R-squared to the plot
        axes[0,0].text(0.05, 0.95, f'R² = {r_squared:.3f}', transform=axes[0,0].transAxes,
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Plot 2: Residuals vs Fitted
        axes[0,1].scatter(fitted_values, residuals, alpha=0.7, color='red')
        axes[0,1].axhline(y=0, color='black', linestyle='--', alpha=0.8)
        axes[0,1].set_xlabel('Fitted Values')
        axes[0,1].set_ylabel('Residuals')
        axes[0,1].set_title('Residuals vs Fitted Values')
        axes[0,1].grid(True, alpha=0.3)

        # Plot 3: Residuals over time
        if 'Date' in aligned_data.columns:
            time_index = aligned_data['Date'].iloc[:len(residuals)]
            axes[1,0].plot(time_index, residuals, color='green', alpha=0.7, marker='o', markersize=4)
            axes[1,0].axhline(y=0, color='black', linestyle='--', alpha=0.8)
            axes[1,0].set_xlabel('Date')
            axes[1,0].set_ylabel('Residuals (Error-Correction Term)')
            axes[1,0].set_title('Error-Correction Term Over Time')
            axes[1,0].grid(True, alpha=0.3)
            axes[1,0].tick_params(axis='x', rotation=45)
        else:
            axes[1,0].plot(residuals, color='green', alpha=0.7, marker='o', markersize=4)
            axes[1,0].axhline(y=0, color='black', linestyle='--', alpha=0.8)
            axes[1,0].set_xlabel('Time Period')
            axes[1,0].set_ylabel('Residuals (Error-Correction Term)')
            axes[1,0].set_title('Error-Correction Term Over Time')
            axes[1,0].grid(True, alpha=0.3)

        # Plot 4: Residuals histogram with normal curve
        axes[1,1].hist(residuals, bins=15, density=True, alpha=0.7, color='purple', edgecolor='black')
        # Add normal distribution overlay
        x_norm = np.linspace(residuals.min(), residuals.max(), 100)
        y_norm = stats.norm.pdf(x_norm, residuals.mean(), residuals.std())
        axes[1,1].plot(x_norm, y_norm, 'r-', linewidth=2, label='Normal Distribution')
        axes[1,1].set_xlabel('Residuals')
        axes[1,1].set_ylabel('Density')
        axes[1,1].set_title('Residuals Distribution')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Residual diagnostics
        print("🔬 Residual Diagnostics:")
        print("-" * 40)

        # Normality test
        shapiro_stat, shapiro_p = stats.shapiro(residuals)
        print(f"   • Shapiro-Wilk normality test: W = {shapiro_stat:.4f}, p = {shapiro_p:.4f}")
        if shapiro_p > 0.05:
            print("     Residuals appear normally distributed ✅")
        else:
            print("     Residuals may not be normally distributed ⚠️")

        # Residual summary
        print(f"   • Residual mean: {residuals.mean():.6f} (should be ≈ 0)")
        print(f"   • Residual std: {residuals.std():.4f}")
        print(f"   • Min residual: {residuals.min():.4f}")
        print(f"   • Max residual: {residuals.max():.4f}")
        print()

        # Prepare return dictionary
        results = {
            'model': model,
            'residuals': residuals,
            'coefficients': coefficients,
            'fitted_values': fitted_values,
            'summary': summary_stats,
            'r_squared': r_squared,
            'f_statistic': {'value': f_stat, 'p_value': f_pvalue},
            'durbin_watson': dw_stat,
            'diagnostic_plots': fig,
            'interpretation': {
                'equation': f"L1_t = {coefficients['β0_intercept']:.4f} + {coefficients['β1_L13']:.4f}*L13_t + {coefficients['β2_trend']:.4f}*trend_t + u_t",
                'l13_effect_significant': model.pvalues['L13'] < 0.05,
                'trend_effect_significant': model.pvalues['trend'] < 0.05,
                'model_significant': f_pvalue < 0.05,
                'variance_explained': r_squared
            }
        }

        print("✅ Cointegrating relationship estimation complete!")
        print(f"📋 Error-correction term (u_t) extracted for ECM modeling")
        print(f"📊 Model explains {r_squared*100:.1f}% of L1 transit time variation")

        return results

    except Exception as e:
        print(f"❌ Error in model estimation: {str(e)}")
        raise e

__all__ = ['_estimate_cointegrating_relation']
