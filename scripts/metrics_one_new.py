import os, sys
import ray
import math
import argparse
import pandas as pd
from tqdm import tqdm
import networkx as nx
import nx_parallel as nxp
from itertools import batched

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from utils import *
from proto import cost_function

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None, type=str)
    parser.add_argument("--results_dir", default=None, type=str)
    parser.add_argument("--num_cpu", default=3, type=int)
    parser.add_argument("--batch_size", default=3, type=int)
    parser.add_argument("--metric", default=0, type=str)
    parser.add_argument("--shapes_file", default="shapes_fix.csv", type=str)
    parser.add_argument("--directed", default=0, type=int)
    parser.add_argument(
        "--alg", default="DEF", type=str, choices=["DEF", "CAP", "LND", "ECL", "CLN"]
    )
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

cpu_count = max(1, os.cpu_count() - 2)
num_cpu = min(args.num_cpu, cpu_count)
batch_size = min(args.batch_size, cpu_count)
print("data_dir:", data_dir)
print("results_dir:", results_dir)
print("num_cpu:", num_cpu)
print("batch_size:", batch_size)

directed = "directed" if args.directed else "undirected"

print(f"networkx: {nx.__version__}")
print(f"nx_parallel: {nxp.__version__}")
os.environ["NETWORKX_AUTOMATIC_BACKENDS"] = "parallel"
os.environ["RAY_memory_monitor_refresh_ms"] = "0"
# nx.config.backends.parallel.active = True
# nx.config.backends.parallel.n_jobs = num_cpu

metrics = {
    "density": density,  # 0 weights are ignored # add directed
    "mean_degree": mean_degree,  # 1 weights are ignored # add directed
    "transitivity": transitivity,  # 2 weights are ignored # add directed
    # 'preferential_attachment' : preferential_attachment.remote(ug_ref), # not implemented for directed type # weights are ignored
    # "bridges": ray.remote(bridges).remote(ug_ref), # not implemented for directed type # weights are ignored
    # 'min_edge_cover' : min_edge_cover.remote(ug_ref), # not implemented for directed type # weights are ignored
    # 'global_efficiency' : global_efficiency.remote(ug_ref), # not implemented for directed type # weights are ignored
    # "jaccard_coefficient": jaccard_coefficient # not implemented for directed type # no weighted
    # 'resource_allocation_index' : resource_allocation_index.remote(ug_ref), #not implemented for directed type # weights are ignored
}
metrics_w = {
    "information_centrality": information_centrality,  # 3 first component not implemented for directed type # add weighted
    "effective_size": effective_size,  # 4!!! weight=weight_function # directed equal undirected # add weighted
    "burt_effective_size": burt_effective_size,  # 5!!! weight=weight_function         ),  # directed equal undirected # add weighted
    ########
    "average_clustering": average_clustering,  # 6 weight=weight_function # add weighted # add directed
    "fast_label_propagation_communities": fast_label_communities,  # 7 weight=weight_function, seed=seed # add directed # add weighted
    "greedy_modularity_communities": greedy_modularity_communities,  # 8 weight=weight_function # add directed # add weighted
}  # betweenness_centrality

m = {**metrics, **metrics_w}
if str(args.metric).isnumeric():
    m = list(m.items())[int(args.metric)]
    print("\nmetric:", m[0])
else:
    if args.metric in m:
        m = (args.metric, m[args.metric])
        print("metric:", m[0], "\n")
    else:
        print("no metric defined.\n")
        sys.exit()

if m[0] in metrics_w:
    print(f"weighted with {args.alg}")

ray.init(num_cpus=num_cpu)

results_file = os.path.join(results_dir, f"{m[0]}.{args.alg}.{directed}.csv")


@ray.remote
def proccess_graph(g, alg, m, seed=13):
    set_seed(seed)
    s = get_stamp(g)
    g = nx.read_gml(g)

    for u, v in g.edges:
        g.edges[u, v][alg] = cost_function(g, u, v, amount=100, proto_type=alg)

    components = [
        g.subgraph(c).copy()
        for c in sorted(
            nx.connected_components(g.to_undirected()), key=len, reverse=True
        )
        if len(c) > 10
    ]

    if m[0] in metrics_w:
        r = m[1](g, weight=alg)
    else:
        r = m[1](g)

    if r is None:
        if len(components):
            print("..trying the first component..")
            if m[0] in metrics_w:
                r = m[1](components[0], weight=alg)
            else:
                r = m[1](components[0])

    return {
        "timestamp": s,
        m[0]: r,
    }


graphs = pd.read_csv(
    os.path.join(results_dir, args.shapes_file), parse_dates=True, index_col=0
)

if os.path.exists(results_file):
    results = pd.read_csv(results_file, dtype=str)
else:
    results = pd.DataFrame()

timestamps = set(results.timestamp) if "timestamp" in results else set()
for batch in tqdm(
    batched(graphs.fname, batch_size), total=math.ceil(len(graphs.fname) / batch_size)
):
    batch = [
        proccess_graph.remote(
            os.path.join(data_dir, f"{get_stamp(g)}.{directed}.gml"), args.alg, m
        )
        for g in batch
        if get_stamp(g) not in timestamps
    ]
    if len(batch):
        results = pd.concat([results, pd.DataFrame(ray.get(batch))])
        results.to_csv(results_file, index=False)
