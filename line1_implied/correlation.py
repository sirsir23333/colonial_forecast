"""Correlation and summary utilities for Colonial Pipeline analysis."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from statsmodels.tsa.stattools import coint

def analyze_correlation_pair(data1, data2, name1, name2, date_col='Date', value_col='Gas Transit Days'):
    """
    Analyze correlation between two pipeline datasets

    Parameters:
    - data1, data2: DataFrames with pipeline data
    - name1, name2: Names for the pipeline lines
    - date_col: Column name for dates
    - value_col: Column name for transit days

    Returns:
    - Dictionary with correlation results and pipeline data
    """
    # Find overlapping period
    start_date = max(data1[date_col].min(), data2[date_col].min())
    end_date = min(data1[date_col].max(), data2[date_col].max())

    # Filter to overlapping period
    data1_filtered = data1[(data1[date_col] >= start_date) & (data1[date_col] <= end_date)].copy()
    data2_filtered = data2[(data2[date_col] >= start_date) & (data2[date_col] <= end_date)].copy()

    # Merge on date for direct comparison - FIXED: Use same suffix format as visualization function
    merged = pd.merge(data1_filtered[[date_col, value_col]], 
                     data2_filtered[[date_col, value_col]], 
                     on=date_col, suffixes=(f'_{name1}', f'_{name2}'), how='inner')

    results = {
        'name1': name1,
        'name2': name2,
        'start_date': start_date,
        'end_date': end_date,
        'n_observations': len(merged),
        'merged_data': merged
    }

    if len(merged) > 2:
        # FIXED: Use correct column names that match the suffixes
        col1, col2 = f'{value_col}_{name1}', f'{value_col}_{name2}'

        # Remove NaN values before correlation calculation
        clean_merged = merged[[col1, col2]].dropna()

        if len(clean_merged) >= 3:  # Need at least 3 points for meaningful correlation
            # Calculate correlations
            pearson_r, pearson_p = pearsonr(clean_merged[col1], clean_merged[col2])
            spearman_r, spearman_p = spearmanr(clean_merged[col1], clean_merged[col2])

            results.update({
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'spearman_r': spearman_r,
                'spearman_p': spearman_p,
                'clean_observations': len(clean_merged)  # Add count of clean observations
            })

            # Test cointegration
            if len(clean_merged) > 10:
                try:
                    coint_stat, coint_p, coint_crit = coint(clean_merged[col1], clean_merged[col2])
                    results.update({
                        'coint_stat': coint_stat,
                        'coint_p': coint_p,
                        'coint_critical': coint_crit,
                        'is_cointegrated': coint_p < 0.05
                    })
                except Exception as e:
                    results['coint_error'] = str(e)
        else:
            results['insufficient_clean_data'] = f"Only {len(clean_merged)} clean observations available"

    return results

def print_correlation_results(results):
    """Print formatted correlation analysis results"""
    name1, name2 = results['name1'], results['name2']
    print(f"\n{'='*60}")
    print(f"ANALYSIS: {name1} vs {name2}")
    print(f"{'='*60}")
    print(f"Period: {results['start_date'].date()} to {results['end_date'].date()}")
    print(f"Total paired observations: {results['n_observations']}")

    if 'clean_observations' in results:
        print(f"Clean observations (no NaN): {results['clean_observations']}")

    if 'pearson_r' in results:
        print(f"\nCorrelation Results:")
        print(f"  • Pearson:  r = {results['pearson_r']:.4f} (p = {results['pearson_p']:.4f})")
        print(f"  • Spearman: ρ = {results['spearman_r']:.4f} (p = {results['spearman_p']:.4f})")

        # Interpret strength
        r = abs(results['pearson_r'])
        if r > 0.7:
            strength = "Strong"
        elif r > 0.3:
            strength = "Moderate"
        else:
            strength = "Weak"
        direction = "positive" if results['pearson_r'] > 0 else "negative"
        print(f"  • Relationship: {strength} {direction} correlation")

        # Cointegration results
        if 'coint_stat' in results:
            print(f"\nCointegration Test:")
            print(f"  • ADF Statistic: {results['coint_stat']:.4f}")
            print(f"  • P-value: {results['coint_p']:.4f}")
            if results['is_cointegrated']:
                print(f"  • Result: 🎯 COINTEGRATED (long-term equilibrium exists)")
            else:
                print(f"  • Result: ❌ Not cointegrated")
        elif 'coint_error' in results:
            print(f"\nCointegration Test: Failed ({results['coint_error']})")
    elif 'insufficient_clean_data' in results:
        print(f"\n⚠️  {results['insufficient_clean_data']}")
    else:
        print("\n⚠️  Insufficient overlapping data for correlation analysis")

def create_transit_correlation_matrix(correlation_results, pipeline_data):
    """Create a comprehensive pairwise plot matrix with the function resul"""

    # Set up the plot grid (4x4 for comprehensive view)
    fig = plt.figure(figsize=(16, 16))

    # Define line info for labeling
    line_info = {
        'Line1': {'name': 'Line 1', 'route': 'HTN→GBJ', 'color': 'blue'},
        'Line3': {'name': 'Line 3', 'route': 'GBJ→HTN', 'color': 'red'},
        'Line13': {'name': 'Line 13', 'route': 'HTN→LNJ', 'color': 'green'}
    }

    # Row 1: Individual time series
    for i, (line_key, line_data) in enumerate(pipeline_data.items()):
        ax = plt.subplot(4, 3, i + 1)
        # Filter out NaN values for plotting
        clean_data = line_data.dropna(subset=['Gas Transit Days'])
        if len(clean_data) > 0:
            ax.plot(clean_data['Date'], clean_data['Gas Transit Days'], 
                   color=line_info[line_key]['color'], alpha=0.7, linewidth=1)
        ax.set_title(f"{line_info[line_key]['name']}: {line_info[line_key]['route']}", 
                    fontweight='bold')
        ax.set_ylabel('Transit Days')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)

    # Row 2: Distribution histograms
    for i, (line_key, line_data) in enumerate(pipeline_data.items()):
        ax = plt.subplot(4, 3, i + 4)
        # Filter out NaN values for histogram
        clean_values = line_data['Gas Transit Days'].dropna()
        if len(clean_values) > 0:
            ax.hist(clean_values, bins=20, alpha=0.7, 
                   color=line_info[line_key]['color'], density=True, 
                   edgecolor='black', linewidth=0.5)
            ax.set_xlabel('Transit Days')
            ax.set_ylabel('Density')
        else:
            ax.text(0.5, 0.5, 'No valid data', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=12, color='gray')
        ax.set_title(f"{line_info[line_key]['name']} Distribution")
        ax.grid(True, alpha=0.3)

    # Row 3: Scatter plots for each pair with correlation
    scatter_positions = [7, 8, 9]  # positions for 3 scatter plots
    pair_keys = ['Line1_vs_Line3', 'Line1_vs_Line13', 'Line3_vs_Line13']

    for pos, key in zip(scatter_positions, pair_keys):
        ax = plt.subplot(4, 3, pos)
        results = correlation_results[key]

        # First, ensure we have proper merged data
        if 'merged_data' not in results or len(results['merged_data']) == 0:
            # Create merged data if missing
            line1_key = results['name1']
            line2_key = results['name2']
            if line1_key in pipeline_data and line2_key in pipeline_data:
                data1 = pipeline_data[line1_key][['Date', 'Gas Transit Days']].copy()
                data2 = pipeline_data[line2_key][['Date', 'Gas Transit Days']].copy()
                merged = pd.merge(data1, data2, on='Date', suffixes=(f'_{line1_key}', f'_{line2_key}'))
                results['merged_data'] = merged

        if 'merged_data' in results and len(results['merged_data']) > 0:
            merged = results['merged_data']
            col1 = f"Gas Transit Days_{results['name1']}"
            col2 = f"Gas Transit Days_{results['name2']}"

            # Remove NaN values and check for sufficient data
            clean_merged = merged[[col1, col2]].dropna()

            if len(clean_merged) >= 3:  # Need at least 3 points for regression
                ax.scatter(clean_merged[col1], clean_merged[col2], alpha=0.6, 
                          color='purple', s=30)

                # Check if correlation is nan and recalculate if needed
                r = results.get('pearson_r', np.nan)
                p_val = results.get('pearson_p', np.nan)

                if pd.isna(r) or pd.isna(p_val):
                    try:
                        from scipy.stats import pearsonr
                        r, p_val = pearsonr(clean_merged[col1], clean_merged[col2])
                        # Update the results dictionary
                        results['pearson_r'] = r
                        results['pearson_p'] = p_val
                        print(f"Recalculated correlation for {results['name1']} vs {results['name2']}: r = {r:.3f}")
                    except Exception as e:
                        print(f"Failed to recalculate correlation for {results['name1']} vs {results['name2']}: {e}")
                        r, p_val = np.nan, np.nan

                # Add regression line with error handling
                try:
                    x_vals = clean_merged[col1].values
                    y_vals = clean_merged[col2].values

                    # Check for sufficient variation in x values
                    if np.std(x_vals) > 1e-10 and len(np.unique(x_vals)) > 1:
                        z = np.polyfit(x_vals, y_vals, 1)
                        p = np.poly1d(z)
                        x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
                        ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)
                    else:
                        print(f"Warning: Insufficient variation in {results['name1']} data for regression")

                except (np.linalg.LinAlgError, np.RankWarning) as e:
                    print(f"Warning: Could not fit regression line for {results['name1']} vs {results['name2']}: {e}")

                ax.set_xlabel(f"{results['name1']} Transit Days")
                ax.set_ylabel(f"{results['name2']} Transit Days")
                ax.set_title(f"{results['name1']} vs {results['name2']}")

                # Add correlation text with proper nan handling
                n_points = len(clean_merged)
                if not pd.isna(r):
                    ax.text(0.05, 0.95, f'r = {r:.3f}\np = {p_val:.3f}\nn = {n_points}', 
                           transform=ax.transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                           fontsize=10)
                else:
                    ax.text(0.05, 0.95, f'r = calculation failed\nn = {n_points}', 
                           transform=ax.transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                           fontsize=10)
            else:
                ax.text(0.5, 0.5, f'Insufficient data\n({len(clean_merged)} points)', 
                       transform=ax.transAxes, ha='center', va='center',
                       fontsize=12, color='gray')
                ax.set_title(f"{results['name1']} vs {results['name2']}")
        else:
            ax.text(0.5, 0.5, 'No overlapping\ndata', 
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=12, color='gray')
            ax.set_title(f"{results['name1']} vs {results['name2']}")

        ax.grid(True, alpha=0.3)

    # Row 4: Combined time series overlay
    ax = plt.subplot(4, 1, 4)

    # Find common date range for overlay (only for lines with data)
    valid_data = {}
    for line_key, line_data in pipeline_data.items():
        clean_data = line_data.dropna(subset=['Gas Transit Days'])
        if len(clean_data) > 0:
            valid_data[line_key] = clean_data

    if len(valid_data) > 1:
        common_start = max(data['Date'].min() for data in valid_data.values())
        common_end = min(data['Date'].max() for data in valid_data.values())

        for line_key, line_data in valid_data.items():
            # Filter to common period for better comparison
            filtered_data = line_data[(line_data['Date'] >= common_start) & 
                                     (line_data['Date'] <= common_end)]
            if len(filtered_data) > 0:
                ax.plot(filtered_data['Date'], filtered_data['Gas Transit Days'], 
                       color=line_info[line_key]['color'], alpha=0.8, linewidth=1.5,
                       label=f"{line_info[line_key]['name']} ({line_info[line_key]['route']})")

        ax.set_title('Transit Times Comparison (Overlapping Period)', fontweight='bold', fontsize=14)
        ax.legend(loc='upper right')
    else:
        # Plot individual series even if no overlap
        for line_key, line_data in valid_data.items():
            ax.plot(line_data['Date'], line_data['Gas Transit Days'], 
                   color=line_info[line_key]['color'], alpha=0.8, linewidth=1.5,
                   label=f"{line_info[line_key]['name']} ({line_info[line_key]['route']})")

        ax.set_title('Transit Times (Individual Series)', fontweight='bold', fontsize=14)
        if valid_data:
            ax.legend(loc='upper right')

    ax.set_xlabel('Date')
    ax.set_ylabel('Gas Transit Days')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()

    return fig

def create_summary_table(correlation_results):
    """Create a summary table of all correlation analyses with robust error handling"""
    summary_data = []

    for key, results in correlation_results.items():
        # Helper function to format numbers or return 'nan'
        def format_number(value, decimals=3):
            if pd.isna(value) or value != value:  # Check for NaN
                return 'nan'
            return f"{value:.{decimals}f}"

        # Helper function to format p-values
        def format_pvalue(value):
            if pd.isna(value) or value != value:
                return 'nan'
            if value < 0.001:
                return '0.000'
            return f"{value:.3f}"

        if 'pearson_r' in results and not pd.isna(results.get('pearson_r')):
            # Has valid correlation data
            summary_data.append({
                'Pair': f"{results['name1']} vs {results['name2']}",
                'Period': f"{results['start_date'].date()} to {results['end_date'].date()}",
                'N': results['n_observations'],
                'Pearson r': format_number(results['pearson_r']),
                'P-value': format_pvalue(results['pearson_p']),
                'Spearman ρ': format_number(results.get('spearman_r', float('nan'))),
                'Cointegrated': '✅' if results.get('is_cointegrated', False) else '❌',
                'Coint p-value': format_pvalue(results.get('coint_p', float('nan')))
            })
        else:
            # No valid correlation data
            summary_data.append({
                'Pair': f"{results['name1']} vs {results['name2']}",
                'Period': f"{results['start_date'].date()} to {results['end_date'].date()}",
                'N': results['n_observations'],
                'Pearson r': 'nan',
                'P-value': 'nan',
                'Spearman ρ': 'nan',
                'Cointegrated': '❌',
                'Coint p-value': 'N/A'
            })

    return pd.DataFrame(summary_data)

def create_stats_table(pipeline_data):
    """Create statistics table for all pipeline lines"""
    stats_data = []
    route_map = {
        'Line1': 'HTN → GBJ',
        'Line3': 'GBJ → HTN', 
        'Line13': 'HTN → LNJ'
    }

    for name, data in pipeline_data.items():
        transit_days = data['Gas Transit Days'].dropna()  # Remove NaN values

        if len(transit_days) > 0:
            stats_data.append({
                'Line': name,
                'Route': route_map[name],
                'Records': len(transit_days),
                'Mean (days)': f"{transit_days.mean():.2f}",
                'Std (days)': f"{transit_days.std():.2f}",
                'Min (days)': f"{transit_days.min():.2f}",
                'Max (days)': f"{transit_days.max():.2f}",
                'Date Range': f"{data['Date'].min().date()} to {data['Date'].max().date()}"
            })
        else:
            stats_data.append({
                'Line': name,
                'Route': route_map[name],
                'Records': 0,
                'Mean (days)': 'N/A',
                'Std (days)': 'N/A',
                'Min (days)': 'N/A',
                'Max (days)': 'N/A',
                'Date Range': 'No data'
            })

    return pd.DataFrame(stats_data)

__all__ = ['analyze_correlation_pair', 'print_correlation_results', 'create_transit_correlation_matrix', 'create_summary_table', 'create_stats_table']
