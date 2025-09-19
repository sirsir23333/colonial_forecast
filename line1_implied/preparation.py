"""Data alignment helpers for ECM pipeline."""

import numpy as np
import pandas as pd

def _prepare_aligned_data(pipeline_data, correlation_results):
    """
    Prepare aligned time series data for ECM forecasting from existing pipeline data.

    This function extracts and aligns the time series data from the existing pipeline_data 
    dictionary, creating a clean DataFrame suitable for Error Correction Model (ECM) 
    forecasting and cointegration analysis.

    Parameters:
    -----------
    pipeline_data : dict
        Dictionary containing Line1, Line3, and Line13 DataFrames with Date and Gas Transit Days
    correlation_results : dict
        Dictionary containing correlation and cointegration analysis results

    Returns:
    --------
    pd.DataFrame
        Aligned DataFrame with columns: ['Date', 'L1', 'L3', 'L13', 'trend']
        - Date: datetime index for time series alignment
        - L1: Line 1 Gas Transit Days (HTN→GBJ)
        - L3: Line 3 Gas Transit Days (GBJ→HTN) 
        - L13: Line 13 Gas Transit Days (HTN→LNJ)
        - trend: Linear trend variable (1, 2, 3, ...) for cointegration analysis

    Notes:
    ------
    - Uses inner join to keep only dates with complete data across all three series
    - Disposes of any periods where data is not available for all three series
    - Uses only the common period where all series have data (no missing value filling)
    - Adds linear trend variable required for ECM specification
    - Includes comprehensive data validation and quality checks
    - Leverages existing correlation analysis results for context
    """

    print("🔄 Preparing aligned data for ECM forecasting...")
    print("="*60)

    # Extract time series from pipeline_data dictionary
    extraction_results = {}

    for line_name, line_data in pipeline_data.items():
        # Extract relevant columns and clean data
        if 'Date' in line_data.columns and 'Gas Transit Days' in line_data.columns:
            clean_data = line_data[['Date', 'Gas Transit Days']].copy()
            clean_data = clean_data.dropna()  # Remove rows with missing data
            clean_data['Date'] = pd.to_datetime(clean_data['Date'])  # Ensure datetime format

            extraction_results[line_name] = {
                'data': clean_data,
                'original_count': len(line_data),
                'clean_count': len(clean_data),
                'date_range': (clean_data['Date'].min(), clean_data['Date'].max()),
                'transit_stats': clean_data['Gas Transit Days'].describe()
            }

            print(f"✅ {line_name}: {len(clean_data)} clean records ({clean_data['Date'].min().date()} to {clean_data['Date'].max().date()})")
        else:
            print(f"❌ {line_name}: Missing required columns")
            return None

    # Verify we have all three datasets
    required_lines = ['Line1', 'Line3', 'Line13']
    if not all(line in extraction_results for line in required_lines):
        missing = [line for line in required_lines if line not in extraction_results]
        print(f"❌ Missing required datasets: {missing}")
        return None

    print(f"\n📊 Data extraction summary:")
    for line_name, info in extraction_results.items():
        print(f"  • {line_name}: {info['clean_count']}/{info['original_count']} records, "
              f"mean transit = {info['transit_stats']['mean']:.2f} days")

    # Find common date period across all three series
    all_dates = []
    for line_name in required_lines:
        dates = set(extraction_results[line_name]['data']['Date'].dt.date)
        all_dates.append(dates)

    # Use intersection to find common dates (inner join approach)
    common_dates = set.intersection(*all_dates)
    common_dates = sorted(list(common_dates))

    print(f"\n🔗 Common period analysis:")
    print(f"  • Line1 dates: {len(all_dates[0])}")
    print(f"  • Line3 dates: {len(all_dates[1])}")
    print(f"  • Line13 dates: {len(all_dates[2])}")
    print(f"  • Common dates: {len(common_dates)}")

    if len(common_dates) == 0:
        print("❌ No common dates found across all three series")
        return None

    print(f"  • Common period: {min(common_dates)} to {max(common_dates)}")

    # Create aligned dataset using only common dates
    aligned_data = pd.DataFrame({'Date': pd.to_datetime(common_dates)})

    # Merge each series for the common dates only
    for line_name in required_lines:
        line_data = extraction_results[line_name]['data'].copy()
        line_data['Date'] = pd.to_datetime(line_data['Date'])

        # Filter to common dates only
        common_data = line_data[line_data['Date'].dt.date.isin(common_dates)].copy()

        # Rename column based on line
        if line_name == 'Line1':
            col_name = 'L1'
        elif line_name == 'Line3':
            col_name = 'L3'
        else:  # Line13
            col_name = 'L13'

        common_data = common_data.rename(columns={'Gas Transit Days': col_name})

        # Merge with aligned_data
        aligned_data = pd.merge(aligned_data, common_data[['Date', col_name]], on='Date', how='inner')

    # Sort by date and reset index
    aligned_data = aligned_data.sort_values('Date').reset_index(drop=True)

    # Add linear trend variable for cointegration analysis
    aligned_data['trend'] = range(1, len(aligned_data) + 1)

    # Final validation - ensure no missing values
    missing_check = aligned_data[['L1', 'L3', 'L13']].isna().sum()
    if missing_check.sum() > 0:
        print(f"⚠️  Warning: Missing values detected after alignment: {missing_check.to_dict()}")
        aligned_data = aligned_data.dropna()
        print(f"  • Removed rows with missing values, final count: {len(aligned_data)}")

    print(f"\n📋 Final dataset validation:")
    print(f"  • Final dataset shape: {aligned_data.shape}")
    print(f"  • Columns: {list(aligned_data.columns)}")
    print(f"  • Date range: {aligned_data['Date'].min().date()} to {aligned_data['Date'].max().date()}")
    print(f"  • Complete observations (no missing values): {len(aligned_data)}")

    # Check for sufficient variation in each series
    for col in ['L1', 'L3', 'L13']:
        std_dev = aligned_data[col].std()
        unique_vals = aligned_data[col].nunique()
        print(f"  • {col}: std={std_dev:.3f}, unique_values={unique_vals}")
        if std_dev < 0.001:
            print(f"    ⚠️  Warning: {col} has very low variation")
        if unique_vals < 3:
            print(f"    ⚠️  Warning: {col} has very few unique values")

    # Summary statistics
    print(f"\n📊 Summary statistics (common period only):")
    summary_stats = aligned_data[['L1', 'L3', 'L13']].describe()
    print(summary_stats.round(3))

    # Reference correlation results for context
    print(f"\n🔍 Leveraging existing correlation analysis:")
    significant_pairs = []
    cointegrated_pairs = []

    for pair_name, results in correlation_results.items():
        if results.get('pearson_p', 1.0) < 0.05:
            significant_pairs.append(f"{pair_name} (r={results.get('pearson_r', 0):.3f})")
        if results.get('is_cointegrated', False):
            cointegrated_pairs.append(pair_name)

    print(f"  • Significant correlations: {len(significant_pairs)}")
    for pair in significant_pairs:
        print(f"    - {pair}")

    print(f"  • Cointegrated pairs: {len(cointegrated_pairs)}")
    for pair in cointegrated_pairs:
        print(f"    - {pair}")

    # ECM readiness assessment
    print(f"\n🎯 ECM Forecasting Readiness Assessment:")
    print(f"  • ✅ Data alignment: Complete (common period approach)")
    print(f"  • ✅ Missing value handling: Disposed of incomplete periods") 
    print(f"  • ✅ Trend variable: Added")
    print(f"  • ✅ Data validation: Passed")

    if len(cointegrated_pairs) > 0:
        print(f"  • ✅ Cointegration detected: Ready for ECM modeling")
    else:
        print(f"  • ⚠️  No cointegration detected: Consider VAR modeling")

    if len(aligned_data) >= 30:
        print(f"  • ✅ Sample size: Adequate ({len(aligned_data)} observations)")
    else:
        print(f"  • ⚠️  Sample size: Limited ({len(aligned_data)} observations)")

    print(f"\n✅ Data preparation complete! Ready for ECM forecasting pipeline.")
    print(f"📋 Use only common period: {aligned_data['Date'].min().date()} to {aligned_data['Date'].max().date()}")
    print(f"📋 Robust approach: No missing value interpolation, complete data only")

    return aligned_data

def align_with_l13(pipeline_data, correlation_results=None):
    """Anchor alignment on Line 13 while flagging missing Line 1/Line 3 values.

    Parameters
    ----------
    pipeline_data : dict
        Dictionary containing at least ``Line13`` along with optional ``Line1``
        and ``Line3`` DataFrames. Each DataFrame must include ``Date`` and
        ``Gas Transit Days`` columns.
    correlation_results : dict, optional
        Unused placeholder to mirror the signature of :func:`_prepare_aligned_data`.

    Returns
    -------
    pandas.DataFrame
        DataFrame with the full Line 13 date range and columns:

        - ``Date`` – chronological timeline anchored on Line 13
        - ``L1`` / ``L3`` – observed values where available, NaN otherwise
        - ``L13`` – Line 13 transit days (complete series)
        - ``trend`` – linear trend (1, 2, 3, ...)
        - ``L1_observed`` / ``L3_observed`` – boolean flags for observed vs missing

    Notes
    -----
    - Preserves every Line 13 observation (left join for other series)
    - Designed for historical imputation workflows where missing L1/L3 should be
      retained instead of dropped
    - Keeps :func:`_prepare_aligned_data` unchanged for existing pipelines
    """

    print("🔄 Anchoring alignment on Line 13...")
    print("=" * 60)

    required_columns = {'Date', 'Gas Transit Days'}

    def _clean_line(line_name):
        if line_name not in pipeline_data:
            raise ValueError(f"Missing dataset: {line_name}")
        line_df = pipeline_data[line_name]
        if not required_columns.issubset(line_df.columns):
            missing = required_columns - set(line_df.columns)
            raise ValueError(f"{line_name} missing columns: {missing}")
        cleaned = line_df[['Date', 'Gas Transit Days']].copy()
        cleaned['Date'] = pd.to_datetime(cleaned['Date']).dt.normalize()
        cleaned = cleaned.dropna(subset=['Date'])
        cleaned = cleaned.sort_values('Date').drop_duplicates('Date', keep='last')
        return cleaned

    # Line 13 is mandatory because it provides the anchor timeline
    line13 = _clean_line('Line13')
    line13 = line13.dropna(subset=['Gas Transit Days'])
    if line13.empty:
        raise ValueError("Line13 contains no valid observations after cleaning")

    print(f"✅ Line13 anchor: {len(line13)} records from {line13['Date'].min().date()} to {line13['Date'].max().date()}")

    base_df = line13.rename(columns={'Gas Transit Days': 'L13'}).reset_index(drop=True)
    base_df['Date'] = base_df['Date'].dt.normalize()
    base_df['trend'] = range(1, len(base_df) + 1)

    # Helper to merge a line onto the Line 13 timeline
    def _merge_line(line_name, col_name):
        try:
            cleaned = _clean_line(line_name).dropna(subset=['Gas Transit Days'])
            cleaned = cleaned.rename(columns={'Gas Transit Days': col_name})
            print(f"✅ {line_name}: {len(cleaned)} observed records")
        except ValueError as err:
            print(f"⚠️  {err}; treating as unavailable for alignment")
            cleaned = pd.DataFrame(columns=['Date', col_name])
        return base_df.merge(cleaned[['Date', col_name]], on='Date', how='left')

    base_df = _merge_line('Line1', 'L1')
    base_df['L1_observed'] = base_df['L1'].notna()

    base_df = _merge_line('Line3', 'L3')
    base_df['L3_observed'] = base_df['L3'].notna()

    # Summary diagnostics
    print("\n📋 Alignment summary (Line 13 anchor):")
    print(f"  • Total timeline length: {len(base_df)} observations")
    print(f"  • L1 observed: {int(base_df['L1_observed'].sum())} ({base_df['L1_observed'].mean():.1%})")
    print(f"  • L3 observed: {int(base_df['L3_observed'].sum())} ({base_df['L3_observed'].mean():.1%})")

    for col in ['L13', 'L1', 'L3']:
        stats = base_df[col].dropna().describe()
        print(f"  • {col} stats (observed only): count={stats.get('count', 0):.0f}, mean={stats.get('mean', float('nan')):.3f}")

    ordered_columns = ['Date', 'L1', 'L3', 'L13', 'trend', 'L1_observed', 'L3_observed']
    base_df = base_df[ordered_columns]

    print("\n✅ Alignment complete (Line 13 preserved, observation flags added)")
    return base_df

def impute_missing_l1(aligned_data, cointegration_results, residual_mode="zero"):
    """Impute missing Line 1 values using the cointegrating relationship.

    Parameters
    ----------
    aligned_data : pandas.DataFrame
        Output from :func:`align_with_l13` containing columns ``L1``, ``L13``,
        ``trend`` and the boolean flag ``L1_observed``.
    cointegration_results : dict
        Result dictionary returned by
        :func:`line1_implied.cointegration._estimate_cointegrating_relation`.
        Must contain ``coefficients`` (with ``β0_intercept``, ``β1_L13``,
        ``β2_trend``) and optionally ``residuals``.
    residual_mode : {'zero', 'mean', 'sampled'}, default 'zero'
        How to treat the error-correction residual when imputing:

        * ``'zero'`` – set the residual adjustment to zero.
        * ``'mean'`` – add the mean of historical residuals.
        * ``'sampled'`` – randomly sample residuals (with replacement) from the
          historical residual distribution.

    Returns
    -------
    pandas.DataFrame
        Copy of *aligned_data* with an additional ``L1_implied`` column that
        contains the original ``L1`` where observed and imputed values elsewhere.

    Notes
    -----
    - The function never overwrites the original ``L1`` column.
    - Residual sampling falls back to zero adjustment if residuals are missing.
    - Designed to be idempotent: re-running will overwrite the ``L1_implied``
      column with the same logic.
    """

    required_cols = {'L1', 'L13', 'trend', 'L1_observed'}
    missing_cols = required_cols - set(aligned_data.columns)
    if missing_cols:
        raise ValueError(f"aligned_data missing required columns: {missing_cols}")

    coeffs = cointegration_results.get('coefficients', {})
    try:
        beta0 = float(coeffs['β0_intercept'])
        beta1 = float(coeffs['β1_L13'])
        beta2 = float(coeffs['β2_trend'])
    except KeyError as err:
        raise ValueError(f"Missing coefficient in cointegration_results: {err}")

    residuals = cointegration_results.get('residuals')
    if residuals is not None:
        residual_series = pd.Series(residuals).dropna()
    else:
        residual_series = pd.Series(dtype=float)

    print("🔄 Imputing missing Line 1 observations")
    print("=" * 60)
    print(f"  • Residual mode: {residual_mode}")

    if residual_mode not in {'zero', 'mean', 'sampled'}:
        raise ValueError("residual_mode must be one of {'zero','mean','sampled'}")

    if residual_mode in {'mean', 'sampled'} and residual_series.empty:
        print("⚠️  No residuals provided; falling back to zero adjustment")
        residual_mode = 'zero'

    aligned = aligned_data.copy()

    missing_mask = ~aligned['L1_observed']
    observed_mask = aligned['L1_observed']

    base_values = beta0 + beta1 * aligned['L13'] + beta2 * aligned['trend']

    if residual_mode == 'zero':
        residual_adjustment = 0.0
    elif residual_mode == 'mean':
        residual_adjustment = residual_series.mean()
    else:  # sampled
        sampled = np.random.choice(residual_series.values, size=missing_mask.sum(), replace=True)
        residual_adjustment = pd.Series(sampled, index=aligned[missing_mask].index)

    aligned['L1_implied'] = aligned['L1']

    if missing_mask.any():
        if residual_mode == 'sampled':
            adjustments = residual_adjustment
        else:
            adjustments = residual_adjustment
        aligned.loc[missing_mask, 'L1_implied'] = base_values[missing_mask] + adjustments

    if observed_mask.any():
        aligned.loc[observed_mask, 'L1_implied'] = aligned.loc[observed_mask, 'L1']

    imputed_count = int(missing_mask.sum())
    observed_count = int(observed_mask.sum())
    print(f"  • Observed L1 values preserved: {observed_count}")
    print(f"  • Imputed L1 values created: {imputed_count}")

    if imputed_count:
        stats = aligned.loc[missing_mask, 'L1_implied'].describe()
        print("  • Imputed stats:")
        print(stats[['count', 'mean', 'std', 'min', 'max']].round(4).to_dict())
    else:
        print("  • No missing values detected; L1_implied matches original L1")

    return aligned


__all__ = ['_prepare_aligned_data', 'align_with_l13', 'impute_missing_l1']
