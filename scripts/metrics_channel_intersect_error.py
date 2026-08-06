import os, sys
import argparse
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import root_mean_squared_error

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None, type=str)
    parser.add_argument("--results_dir", default=None, type=str)
    args = parser.parse_args()
else:
    sys.exit()

base_dir = os.path.dirname(__file__)
if "app" in base_dir:
    base_dir = "./"
data_dir = (
    os.path.abspath(os.path.join(base_dir, "..", "data"))
    if args.data_dir is None
    else args.data_dir
)
results_dir = (
    os.path.abspath(os.path.join(base_dir, "..", "results"))
    if args.results_dir is None
    else args.results_dir
)
os.makedirs(results_dir, exist_ok=True)

print("data_dir:", data_dir)
print("results_dir:", results_dir)


def get_data(alg="DEF", directed=True):
    df_true = os.path.join(
        results_dir,
        f"channel_intersect_cost.{alg}.{'directed' if directed else 'undirected'}.csv",
    )
    df_sampled = os.path.join(
        results_dir,
        f"channel_intersect_cost_bootstrap.{alg}.{'directed' if directed else 'undirected'}.csv",
    )
    if os.path.exists(df_true) and os.path.exists(df_sampled):
        df_true = pd.read_csv(
            df_true,
            index_col=0,
        )
        df_sampled = pd.read_csv(
            df_sampled,
            index_col=0,
        )
        df = df_true.join(df_sampled, lsuffix="_true", rsuffix="_sampled")
        df = df[["edges_intersection_rate_true", "edges_intersection_rate_sampled"]]
        df.dropna(inplace=True)
        df.sort_index(inplace=True)
        return df  # .iloc[
        #:, :2
        # ]  # root_mean_squared_error(df.iloc[:, 0], df.iloc[:, 1]), df.shape[0]
    # return  # np.nan, np.nan


results = {}
total = pd.DataFrame()
for alg in ["DEF", "CAP", "LND", "CLN", "ECL"]:
    results.setdefault(alg, {})
    for dir in [True, False]:
        df = get_data(alg=alg, directed=dir)
        if df is not None and not df.empty:
            total = pd.concat([total, df], axis=0)
            rmse, n = root_mean_squared_error(df.iloc[:, 0], df.iloc[:, 1]), df.shape[0]
            corr, p = spearmanr(df.iloc[:, 0], df.iloc[:, 1])
            results[alg]["directed" if dir else "undirected"] = (
                f"RMSE: {rmse:.3f}({n}), Spearman: {corr:.3f}, p={p:.3f}"
            )

results = pd.DataFrame(results).fillna("")
total_rmse = root_mean_squared_error(total.iloc[:, 0], total.iloc[:, 1])
total_corr, total_p = spearmanr(total.iloc[:, 0], total.iloc[:, 1])
print(f"Total RMSE: {total_rmse:.3f} ({total.shape[0]})")
print(f"Total Spearman: {total_corr:.3f}, p-value: {total_p:.3f}")
total = f"Total RMSE: {total_rmse:.3f} ({total.shape[0]}), Total Spearman: {total_corr:.3f}, p-value: {total_p:.3f}"
results = pd.concat([results.reset_index(), pd.Series([total], name="index")], axis=0)
print(results)
results.to_csv(
    os.path.join(results_dir, "channel_intersect_cost_errors.csv"), index=False
)
