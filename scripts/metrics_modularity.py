import os, sys
import ray
import math
import json
import argparse
import pandas as pd
from tqdm import tqdm
import networkx as nx

# from copy import deepcopy
import nx_parallel as nxp
from itertools import batched

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from utils import set_seed, get_stamp
from proto import cost_function


def fast_label_communities_comparison(g, weight=None, seed=13, n_random=100):
    try:
        g = g.copy()  # deepcopy(g)
        obs_partition = list(
            nx.community.fast_label_propagation_communities(g, weight=weight, seed=seed)
        )
        obs_num_communities = len(obs_partition)
        obs_modularity = nx.community.modularity(g, communities=obs_partition)
        rand_modularity = []
        rand_num_communities = []
        swap_func = nx.double_edge_swap
        if g.is_directed():
            swap_func = nx.directed_edge_swap

        for i in range(n_random):
            try:
                _g = g.copy()  # deepcopy(g)
                swap_func(
                    _g,
                    nswap=10 * _g.number_of_edges(),
                    max_tries=1000 * _g.number_of_edges(),
                    seed=seed + i,
                )
                rand_partition = list(
                    nx.community.fast_label_propagation_communities(
                        _g, weight=weight, seed=seed
                    )
                )
                rand_num_communities.append(len(rand_partition))
                rand_modularity.append(
                    nx.community.modularity(_g, communities=rand_partition)
                )
            except Exception as e:
                print(e)

        return {
            "obs_modularity": obs_modularity,
            "obs_num_communities": obs_num_communities,
            "rand_modularity": rand_modularity,
            "rand_num_communities": rand_num_communities,
        }
    except Exception as e:
        print(e)


def greedy_modularity_communities_comparison(g, weight=None, seed=13, n_random=100):
    try:
        g = g.copy()  # g = deepcopy(g)
        obs_partition = list(
            nx.community.greedy_modularity_communities(g, weight=weight)
        )
        obs_num_communities = len(obs_partition)
        obs_modularity = nx.community.modularity(g, communities=obs_partition)
        rand_modularity = []
        rand_num_communities = []
        swap_func = nx.double_edge_swap
        if g.is_directed():
            swap_func = nx.directed_edge_swap
        for i in range(n_random):
            try:
                _g = g.copy()  # _g = deepcopy(g)
                swap_func(
                    _g,
                    nswap=10 * _g.number_of_edges(),
                    max_tries=1000 * _g.number_of_edges(),
                    seed=seed + i,
                )
                rand_partition = list(
                    nx.community.greedy_modularity_communities(_g, weight=weight)
                )
                rand_num_communities.append(len(rand_partition))
                rand_modularity.append(
                    nx.community.modularity(_g, communities=rand_partition)
                )
            except Exception as e:
                print(e)

        return {
            "obs_modularity": obs_modularity,
            "obs_num_communities": obs_num_communities,
            "rand_modularity": rand_modularity,
            "rand_num_communities": rand_num_communities,
        }
    except Exception as e:
        print(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None, type=str)
    parser.add_argument("--results_dir", default=None, type=str)
    parser.add_argument("--num_cpu", default=4, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
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

metrics_w = {
    "fast_label_communities_comparison": fast_label_communities_comparison,  # 7 weight=weight_function, seed=seed # add directed # add weighted
    "greedy_modularity_communities_comparison": greedy_modularity_communities_comparison,  # 8 weight=weight_function # add directed # add weighted
}

m = {**metrics_w}
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

results_file = os.path.join(results_dir, f"{m[0]}.{args.alg}.{directed}.json")


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

    r = m[1](g, weight=alg)

    if r is None:
        if len(components):
            print("..trying the first component..")
            r = m[1](components[0], weight=alg)

    return {
        s: {
            m[0]: r,
        }
    }


graphs = pd.read_csv(
    os.path.join(results_dir, args.shapes_file), parse_dates=True, index_col=0
)

if os.path.exists(results_file):
    with open(results_file, "r") as f:
        results = json.load(f)
else:
    results = {}

for batch in tqdm(
    batched(graphs.fname, batch_size), total=math.ceil(len(graphs.fname) / batch_size)
):
    batch = [
        proccess_graph.remote(
            os.path.join(data_dir, f"{get_stamp(g)}.{directed}.gml"), args.alg, m
        )
        for g in batch
        if get_stamp(g) not in results
    ]
    if len(batch):
        r_batch = ray.get(batch)
        for b in r_batch:
            results.update(b)
        with open(results_file, "w") as f:
            json.dump(results, f)
