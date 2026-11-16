# Complexity Budgeting for Credit Card Default Prediction

MSMA 5023, Module 4 Assignment. This continues my Assignment 1 baseline
([credit-risk-ml-workflow](https://github.com/amanveerwesley/credit-risk-ml-workflow)).

The notebook compares a default logistic regression with three SGD configurations
(tuned learning rate, manual early stopping, feature reduction) under a fixed
compute budget of 15 minutes on one CPU thread.

## Files
- `a2_complexity_budget.ipynb`: the full experiment.
- `a1_split_ids.csv`: the Assignment 1 train, validation and test assignment.
- `make_report_figures.py`: draws the two report figures from `results.json`.
- `a2_outputs/`: results, tables and figures from the run used in the report.
- `requirements.txt`: pinned library versions.

## How to run
1. `pip install -r requirements.txt`
2. Open `a2_complexity_budget.ipynb` and run all cells. It takes under a minute.

The notebook downloads the UCI dataset (Yeh, 2009, CC BY 4.0,
https://doi.org/10.24432/C55S3H) if `uci_credit_default.zip` is not in the folder.
The Assignment 1 test partition is not used anywhere in this notebook.
Timings depend on the machine, so a rerun reproduces the scores but not the exact times.