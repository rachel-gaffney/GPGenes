# Gaussian Processes for Modelling Double Gene Knockout


### Minimal GRN simulator + GP residual model


Install dependencies:
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt


Run:
python -m experiments.train_gp


This simulates control/single/double gene knockouts with replicates, computes control baseline, and fits per-gene GP models to residuals using a graph-informed multi-hot kernel.
