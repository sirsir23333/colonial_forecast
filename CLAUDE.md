# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project predicts Colonial Pipeline (CPL) gasoline throughput, specifically Line 1 flows into PADD 1, using two complementary analytical methods and transit time data.

## Goal
Estimate CPL Line 1 throughput into PADD 1 using physical constraints and market data validation.

## Data Sources

### Primary Data
- **EIA Pipeline Data**: PADD 3 → PADD 1 flows by pipeline (monthly) in `data/gasoline_pipeline_EIA.csv` and `data/gasoil_pipline_EIA.csv`
- **Transit Times**: Daily cycle-by-cycle transit times for different lines in `data/colonial_transit_time.xlsx`
- **Email Extraction**: Colonial Pipeline transit bulletins scraped from Outlook (`data/ScrapeEmail.py`)

### Pipeline Specifications
- **Line 1**: 1.37 mb/d capacity, gasoline, 1049 miles, 36-40 inch diameter
- **Line 2**: 1.16 mb/d capacity, gasoil  
- **Line 3**: 0.885 mb/d capacity, multi-product

## Methodology

### Method 1: Physical Throughput Bounds
Uses Little's Law: **Linefill = Throughput × Transit Time**

1. **Line Volume Calculation**: 
   - 36" diameter: ~6.9 mbd implied cycle volume
   - 40" diameter: ~8.1 mbd implied cycle volume
2. **Throughput Estimation**: Volume ÷ Average Transit Time
3. **Key Assumption**: Line 1 operates at maximum utilization with no intermediate distribution

### Method 2: EIA Flow Split  
1. Start with total EIA PADD 3 → PADD 1 pipeline volumes
2. Apply 65% share split between Colonial and Plantation Pipeline (PPL)
3. **Share Calculation**: Colonial 1.37 mb/d ÷ (Colonial 1.37 + Plantation 0.72) = 0.656 ≈ 0.65

### Validation & Visualization
- Plot Method 1 vs Method 2 results for consistency analysis
- Identify gaps and inconsistencies between physical bounds and market data

## Development Environment

### Python Setup
**Python Executable**: `/Users/zhangsiwei/miniconda3/envs/cleanenv/bin/python`

```bash
pip install -r requirements.txt
jupyter lab
```

### Key Libraries
- **Data**: pandas, numpy, scipy
- **Visualization**: plotly, matplotlib, seaborn  
- **ML**: scikit-learn, lightgbm, optuna, pmdarima
- **Email Processing**: win32com.client, beautifulsoup4

## Architecture & Key Files

### Analysis Notebooks
- `colonial_pipeline_forcast_line1.ipynb`: Main Line 1 throughput analysis with the two methods
- `data/data_pulling.ipynb`: Data acquisition and preprocessing, only can used in windows
- `correlation.ipynb`: Transit time vs incentive arbitrage correlation analysis (now imports helpers from `line1_implied`), including the `create_transit_correlation_matrix()` visualization matrix
- **ECM Forecasting Pipeline**: Complete Error Correction Model implementation exposed via `line1_implied.run_all.run_pipeline_complete()` ✅
- Robust error handling for polyfit regression analysis and data quality validation

### Core Processing
- `data/ScrapeEmail.py`: Email extraction with configurable From/To parameters (HTN → GBJ) as well as other start and end point, but can only be used in windows
- Transit time processing: Latest cycle per year, gas days + hours conversion
- Monthly aggregation: Average gas transit times by month

## Work in Progress

### Correlation Analysis
- **Cointegration Testing**: Engle-Granger tests for long-term equilibrium relationships between transit times - COMPLETED ✅
- **ECM Model Building**: Full Error Correction Model with AIC lag selection and forecast evaluation - COMPLETED ✅
- **Advanced Visualization**: Built `create_transit_correlation_matrix()` function with ggpairs-style matrix layout - COMPLETED ✅
- **Results Pipeline**: Complete `run_pipeline_complete()` function with structured outputs - COMPLETED ✅
- **Robust Error Handling**: NaN value detection and removal for reliable correlation calculations - COMPLETED ✅
- **Cross-validate**: cross validate transit time relationships against EIA monthly data

### Geographic Refinement  
- Identify PADD 1 delivery points closer than GBR (currently using ATJ)
- Build node-to-node transit time estimates including junction times
- Fix inconsistencies in implied throughput ranges

### Data Expansion
- Increase historical coverage from T4 transit bulletins
- Build implied Line 1 transit times using Line 1 + Line 3 combined data
- Derive implied ranges for comparison with actual EIA data

## Key Parameters & Formulas

### Transit Time Bounds
- Lower bound: 7.0 (36" throughout capacity)
- Upper bound: 8.61 (40" throughout capacity) 
- Formula: **Q = L/T** where L = capacity parameter, T = transit time

### Pipeline Share Split
- **Colonial Pipeline**: 1.37 mb/d capacity
- **Plantation (SE) Pipeline**: 0.72 mb/d capacity  
- **Colonial Share**: 1.37 ÷ (1.37 + 0.72) = 0.656 ≈ **0.65**

## ECM Pipeline Details

### Package Layout (`line1_implied/`)
1. `preparation._prepare_aligned_data` – align common-sample series with trend column
2. `cointegration._estimate_cointegrating_relation` – fit long-run relationship and residuals
3. `ecm._build_ecm_model` – select lags, estimate ECM, run diagnostics
4. `forecast._forecast_evaluation` – rolling multi-horizon evaluation vs baselines
5. `reporting._display_results/_display_plots/_save_outputs` – console/plot/file outputs
6. `run_all.run_pipeline_complete` – orchestrator that wires the stages together

### Key Results:
- **Cointegrating Relationship**: L1 = 5.7113 + 0.1775×L13 + 0.0220×trend (R² = 0.4939)
- **ECM Performance**: Consistently outperforms Random Walk and ARIMA across all horizons (H1-H4)
- **Error Correction**: γ = -0.5278, Half-life = 1.3 periods

### Usage:
```python
from line1_implied.run_all import run_pipeline_complete

# Requires pre-populated `pipeline_data` and `correlation_results` dictionaries
pipeline_results = run_pipeline_complete(pipeline_data, correlation_results)

# Custom parameters example
pipeline_results = run_pipeline_complete(
    pipeline_data,
    correlation_results,
    n_test=30,
    exog_nowcast="arima",
    display_results=False,
    save_outputs=True,
)
```

## Important Notes
- Email extraction requires Windows with Outlook COM interface
- Transit times measured HTN (Houston Terminal) → GBJ (Greensboro Junction)
- Colonial vs Plantation split based on published pipeline capacities
- Method assumes Line 1 maximum utilization scenarios
- ECM pipeline requires existing `pipeline_data` and `correlation_results` variables

# Technical Tool Guidelines
Jupyter Notebook Editing (NotebookEdit Tool)
Critical Known Issue
The NotebookEdit tool has a default insertion behavior that inserts new cells at the very top of the notebook when no cell_id is specified. This can create ordering problems and notebook structure issues.

Required Best Practices
Always Get Cell IDs First

# Extract cell IDs from any notebook before editing
cat notebook.ipynb | grep -E '"id": "[^"]*"'
Always Use cell_id Parameter

NEVER use NotebookEdit with edit_mode="insert" without specifying cell_id
ALWAYS target insertion after a specific existing cell
Use cell_id to control exact placement in notebook structure
Proper Insertion Pattern

# CORRECT: Target specific cell for insertion
NotebookEdit(
    notebook_path="path/to/notebook.ipynb",
    edit_mode="insert", 
    cell_id="existing_cell_id",  # Insert AFTER this cell
    cell_type="markdown",
    new_source="content"
)

# WRONG: No cell_id specified - will insert at top
NotebookEdit(
    notebook_path="path/to/notebook.ipynb", 
    edit_mode="insert",  # This will go to the top!
    cell_type="markdown",
    new_source="content"
)
Before Large Notebook Operations

Read file to understand structure and get cell IDs
Plan insertion sequence to avoid ordering issues
Test with single cell insertion first
Consider using fresh/clean notebooks for complex structures
Why This Matters
Incorrect cell insertion can create notebook structure problems that are difficult to fix, especially with large notebooks that exceed file size reading limits. Following these practices ensures precise control over notebook organization.
