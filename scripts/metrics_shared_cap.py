import os, sys
import ray
import math
import pickle
import argparse
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
import nx_parallel as nxp
from itertools import batched


import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_cpu", default=3, type=int)
    parser.add_argument("--batch_size", default=3, type=int)
    parser.add_argument("--data_dir", default=None, type=str)
    parser.add_argument("--results_dir", default=None, type=str)
    parser.add_argument("--shapes_file", default="shapes_fix.csv", type=str)
    parser.add_argument("--directed", default=0, type=int)
    args = parser.parse_args()
else:
    sys.exit()

from utils import *

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

cpu_count = max(1, os.cpu_count() - 2)
num_cpu = min(args.num_cpu, cpu_count)
print("data_dir:", data_dir)
print("results_dir:", results_dir)
print("num_cpu:", num_cpu)
print("batch_size:", args.batch_size)

directed = "directed" if args.directed else "undirected"
results_file = os.path.join(results_dir, f"shared_cap.{directed}.pkl")


print(f"networkx: {nx.__version__}")
print(f"nx_parallel: {nxp.__version__}")
os.environ["NETWORKX_AUTOMATIC_BACKENDS"] = "parallel"
os.environ["RAY_memory_monitor_refresh_ms"] = "500"
# nx.config.backends.parallel.active = True
# nx.config.backends.parallel.n_jobs = num_cpu
ray.init(num_cpus=num_cpu)


def shared_node_deg_capacities(g):
    node_capacities = []
    for n in g.nodes:
        node_capacities.append(
            (
                g.degree[n],
                np.sum([d["capacity_sat"] / 2 for u, v, d in g.edges(n, data=True)]),
            )
        )
    return np.asarray(node_capacities)


@ray.remote
def proccess_graph(g, seed=13):
    set_seed(seed)
    g = nx.read_gml(g)
    return shared_node_deg_capacities(g)


graphs = pd.read_csv(
    os.path.join(results_dir, args.shapes_file), parse_dates=True, index_col=0
)

if os.path.exists(results_file):
    with open(os.path.join(results_dir, results_file), "rb") as handle:
        results = pickle.load(handle)
else:
    results = {}

for batch in tqdm(
    batched(graphs.fname, args.batch_size),  # .iloc[:7]
    total=math.ceil(len(graphs.fname) / args.batch_size),
):
    batch = {
        get_stamp(g): proccess_graph.remote(
            os.path.join(data_dir, f"{get_stamp(g)}.{directed}.gml")
        )
        for g in batch
        if get_stamp(g) not in results
    }
    if len(batch):
        for k, v in batch.items():
            if isinstance(v, ray.ObjectRef):
                batch[k] = ray.get(v)
        results.update(batch)
        with open(os.path.join(results_dir, results_file), "wb") as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
