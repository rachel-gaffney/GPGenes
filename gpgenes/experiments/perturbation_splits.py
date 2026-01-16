import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gpgenes import data
from gpgenes.models.train import *


_experiments = [
    {"train": "single+double", "test": "single+double"},
    {"train": "single+double", "test": "single"},
    {"train": "single+double", "test": "double"},
    {"train": "single", "test": "single+double"},
    {"train": "single", "test": "single"},
    {"train": "single", "test": "double"},
    {"train": "double", "test": "single+double"},
    {"train": "double", "test": "single"},
    {"train": "double", "test": "double"},
]


def test_split_groups(solver_fnc, title):

    plot_exp_order = ["single", "double", "single+double"]
    results = np.ndarray((3, 3), dtype=np.float32)

    n_genes = 5
    n_motifs = 3
    n_sparse = 2

    genes = data.create_genes(
        n_genes=n_genes, n_motif=n_motifs, n_sparse=n_sparse, seed=1
    )

    perturbations = data.make_perturbation_list(
        n_genes=n_genes,
        include_singles=True,
        include_doubles=True,
        n_doubles=80,
        seed=0,
    )

    rows = data.simulate_dataset(
        genes,
        perturbations=perturbations,
        n_reps=5,
        steps=1500,
        delta=0.01,
        tail_steps=200,
        seed=42,
    )
    df = pd.DataFrame(rows)

    for cfg in _experiments:
        print(f"Running experiment: {cfg}")
        df_train, df_test = data.split_by_perturbation(
            df,
            train_frac=0.8,
            train_single_pert="single" in cfg["train"],
            train_double_pert="double" in cfg["train"],
            test_single_pert="single" in cfg["test"],
            test_double_pert="double" in cfg["test"],
            seed=0,
        )

        mu = data.compute_control_baseline(df_train, n_genes=n_genes)

        Xtr, Ytr, _ = data.build_xy_from_df(df_train, n_genes=n_genes)
        Xte, Yte, _ = data.build_xy_from_df(df_test, n_genes=n_genes)

        Rtr = data.residualize(Ytr, mu)
        Rte = data.residualize(Yte, mu)

        solver = solver_fnc(genes, n_genes, Xtr, Rtr)
        rmses = solver(Xte, Rte)

        mean_rmse = np.mean(rmses)
        results_i = plot_exp_order.index(cfg["train"])
        results_j = plot_exp_order.index(cfg["test"])
        results[results_i][results_j] = mean_rmse

        print(f"[GP full] Mean RMSE across genes: {mean_rmse:.4f}")
        print(f"[GP full] Median RMSE across genes: {np.median(rmses):.4f}")

    plt.figure(figsize=(6, 5))
    plt.imshow(results)
    plt.colorbar(label="Mean RMSE")

    plt.xticks([0, 1, 2], plot_exp_order)
    plt.yticks([0, 1, 2], plot_exp_order)

    plt.xlabel("Test")
    plt.ylabel("Train")
    plt.title(f"RMSE by Train/Test Split: {title}")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    result_lr = test_split_groups(solver_fnc=solver_linear, title="Linear")
    result_full = test_split_groups(solver_fnc=solver_full, title="Full GP")
