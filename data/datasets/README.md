# Datasets Directory

This directory contains datasets used for experiments in the thesis.

## Structure

- `inventory_data/` - Inventory management datasets
- `gridworld_data/` - GridWorld environment configurations
- `synthetic/` - Synthetically generated test data

## Data Sources

All datasets are either:
1. Synthetically generated within the experimental framework
2. Standard benchmark datasets from the RL literature
3. Custom datasets created for this thesis

## Usage

Datasets are loaded automatically by the experiment runner. See `src/experiments/experiment_runner.py` for examples.

## Notes

- Large data files are excluded from git (see .gitignore)
- Data generation scripts are included in each subdirectory
- All data is in standard formats (CSV, JSON, NPY)