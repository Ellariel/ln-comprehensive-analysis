import os, sys
import argparse
import networkx as nx
import pandas as pd
import numpy as np
from tqdm import tqdm
from itertools import pairwise
from scipy.stats import ks_2samp, wasserstein_distance

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from utils import get_stamp, set_seed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None, type=str)
    parser.add_argument("--results_dir", default=None, type=str)
    parser.add_argument("--shapes_file", default="shapes_fix.csv", type=str)
    parser.add_argument("--directed", default=1, type=int)
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

directed = "directed" if args.directed else "undirected"
results_file = os.path.join(results_dir, f"ks_metrics.CAP.{directed}.csv")


def shared_node_capacities(g):
    node_capacities = []
    for n in g.nodes:
        node_capacities.append(
            np.sum([d["capacity_sat"] / 2 for u, v, d in g.edges(n, data=True)])
        )
    return np.asarray(node_capacities)


def proccess_graphs(g1, g2, seed=13):
    set_seed(seed)
    s = get_stamp(g2)
    g1 = nx.read_gml(g1)
    g2 = nx.read_gml(g2)
    a = shared_node_capacities(g1)
    b = shared_node_capacities(g2)
    a = a[~pd.isnull(a)]
    b = b[~pd.isnull(b)]
    ks_stat, ks_p = ks_2samp(a, b)
    wd = wasserstein_distance(a, b)
    return {
        "timestamp": s,
        "ks_stat": ks_stat,
        "ks_p": ks_p,
        "wasserstein_distance": wd,
    }


graphs = pd.read_csv(
    os.path.join(results_dir, args.shapes_file), parse_dates=True, index_col=0
)


if os.path.exists(results_file):
    results = pd.read_csv(results_file, dtype=str)
else:
    results = pd.DataFrame()

timestamps = set(results.timestamp) if "timestamp" in results else set()
for g1, g2 in tqdm(pairwise(graphs.fname), total=len(graphs.fname) - 1):
    if get_stamp(g2) not in timestamps:
        r = proccess_graphs(
            os.path.join(data_dir, f"{get_stamp(g1)}.{directed}.gml"),
            os.path.join(data_dir, f"{get_stamp(g2)}.{directed}.gml"),
        )
        results = pd.concat([results, pd.DataFrame([r])])
        results.to_csv(results_file, index=False)
