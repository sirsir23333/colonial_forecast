"""Stationarity diagnostics for Colonial Pipeline series."""

import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

def _test_stationarity(ts_data, column_name, max_lags=None, trend='c'):
    """
    Test stationarity of a time series using ADF and KPSS tests

    Parameters:
    - ts_data: pandas Series or DataFrame column with time series data
    - column_name: string name for the series (for reporting)
    - max_lags: maximum lags for ADF test (None for auto)
    - trend: trend assumption for tests ('c'=constant, 'ct'=constant+trend, 'n'=none)

    Returns:
    - Dictionary with test results
    """
    # Remove NaN values
    clean_data = ts_data.dropna()

    if len(clean_data) < 10:
        return {'error': f'Insufficient data points ({len(clean_data)}) for stationarity tests'}

    results = {
        'series_name': column_name,
        'n_observations': len(clean_data),
        'mean': clean_data.mean(),
        'std': clean_data.std(),
        'min': clean_data.min(),
        'max': clean_data.max()
    }

    try:
        # Augmented Dickey-Fuller test
        # H0: Unit root exists (non-stationary)
        # H1: No unit root (stationary)
        adf_result = adfuller(clean_data, maxlag=max_lags, regression=trend)

        results['adf'] = {
            'statistic': adf_result[0],
            'p_value': adf_result[1],
            'used_lag': adf_result[2],
            'n_obs': adf_result[3],
            'critical_values': adf_result[4],
            'is_stationary': adf_result[1] < 0.05,
            'conclusion': 'Stationary' if adf_result[1] < 0.05 else 'Non-stationary'
        }

    except Exception as e:
        results['adf'] = {'error': str(e)}

    try:
        # KPSS test  
        # H0: Series is stationary
        # H1: Series is non-stationary
        kpss_result = kpss(clean_data, regression=trend, nlags='auto')

        results['kpss'] = {
            'statistic': kpss_result[0],
            'p_value': kpss_result[1],
            'lags_used': kpss_result[2],
            'critical_values': kpss_result[3],
            'is_stationary': kpss_result[1] > 0.05,
            'conclusion': 'Stationary' if kpss_result[1] > 0.05 else 'Non-stationary'
        }

    except Exception as e:
        results['kpss'] = {'error': str(e)}

    # Overall conclusion
    if 'error' not in results['adf'] and 'error' not in results['kpss']:
        adf_stat = results['adf']['is_stationary']
        kpss_stat = results['kpss']['is_stationary']

        if adf_stat and kpss_stat:
            results['overall_conclusion'] = 'STATIONARY (both tests agree)'
        elif not adf_stat and not kpss_stat:
            results['overall_conclusion'] = 'NON-STATIONARY (both tests agree)'
        else:
            results['overall_conclusion'] = 'INCONCLUSIVE (tests disagree)'

    return results

def _print_stationarity_results(results):
    """Print formatted stationarity test results"""
    print(f"\n{'='*70}")
    print(f"STATIONARITY ANALYSIS: {results['series_name']}")
    print(f"{'='*70}")
    print(f"Data Summary:")
    print(f"  • Observations: {results['n_observations']}")
    print(f"  • Mean: {results['mean']:.4f}")
    print(f"  • Std Dev: {results['std']:.4f}")
    print(f"  • Range: [{results['min']:.4f}, {results['max']:.4f}]")

    if 'adf' in results and 'error' not in results['adf']:
        adf = results['adf']
        print(f"\n🔍 Augmented Dickey-Fuller Test:")
        print(f"  • Test Statistic: {adf['statistic']:.6f}")
        print(f"  • P-value: {adf['p_value']:.6f}")
        print(f"  • Lags Used: {adf['used_lag']}")
        print(f"  • Critical Values:")
        for level, cv in adf['critical_values'].items():
            print(f"    - {level}: {cv:.6f}")
        print(f"  • Conclusion: {adf['conclusion']} (p {'<' if adf['is_stationary'] else '≥'} 0.05)")

    if 'kpss' in results and 'error' not in results['kpss']:
        kpss = results['kpss']
        print(f"\n📊 KPSS Test:")
        print(f"  • Test Statistic: {kpss['statistic']:.6f}")
        print(f"  • P-value: {kpss['p_value']:.6f}")
        print(f"  • Lags Used: {kpss['lags_used']}")
        print(f"  • Critical Values:")
        for level, cv in kpss['critical_values'].items():
            print(f"    - {level}%: {cv:.6f}")
        print(f"  • Conclusion: {kpss['conclusion']} (p {'>' if kpss['is_stationary'] else '≤'} 0.05)")

    if 'overall_conclusion' in results:
        print(f"\n🎯 OVERALL CONCLUSION: {results['overall_conclusion']}")

def analyze_pipeline_stationarity(pipeline_data):
    """Analyze stationarity for all pipeline data"""
    print("🔍 STATIONARITY ANALYSIS FOR ALL PIPELINES")
    print("="*80)

    stationarity_results = {}
    for line_name, data in pipeline_data.items():
        clean_data = data['Gas Transit Days'].dropna()
        if len(clean_data) >= 10:
            results = _test_stationarity(clean_data, f"{line_name} Transit Days")
            stationarity_results[line_name] = results
            _print_stationarity_results(results)
        else:
            print(f"\n⚠️  {line_name}: Insufficient data ({len(clean_data)} points) for stationarity tests")

    return stationarity_results

__all__ = ['_test_stationarity', '_print_stationarity_results', 'analyze_pipeline_stationarity']
