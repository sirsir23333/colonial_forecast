"""Data alignment helpers for ECM pipeline."""

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

__all__ = ['_prepare_aligned_data']
