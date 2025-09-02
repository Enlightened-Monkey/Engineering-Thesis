# Results Directory

This directory contains experimental results and analysis outputs.

## Structure

- `performance_metrics/` - Algorithm performance comparisons
- `convergence_analysis/` - Convergence behavior studies  
- `policy_analysis/` - Policy structure and time-inconsistency analysis
- `parameter_studies/` - Effects of σ and γ parameters

## File Formats

Results are stored in various formats:
- `.pkl` - Python pickle files with complete results dictionaries
- `.csv` - Tabular data for statistical analysis
- `.json` - Configuration files and metadata

## Usage

Results can be loaded using the analysis tools:

```python
from src.utils.analysis_tools import load_results
results = load_results('performance_metrics/inventory_experiment.pkl')
```

## Notes

- Results files are excluded from git due to size (see .gitignore)
- Each experiment generates timestamped result files
- Summary statistics are available in CSV format for LaTeX tables