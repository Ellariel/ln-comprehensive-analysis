import os, sys
import ray
import json
import argparse
import networkx as nx
import pandas as pd
import nx_parallel as nxp
from itertools import pairwise, combinations
from itertools import batched
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from utils import get_stamp, set_seed
from proto import get_shortest_path, cost_function


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_cpu", default=5, type=int)
    parser.add_argument("--batch_size", default=5, type=int)
    parser.add_argument("--data_dir", default=None, type=str)
    parser.add_argument("--results_dir", default=None, type=str)
    parser.add_argument("--shapes_file", default="shapes_fix.csv", type=str)
    parser.add_argument("--directed", default=0, type=int)
    parser.add_argument("--verbose", default=0, type=int)
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
print("data_dir:", data_dir)
print("results_dir:", results_dir)
print("num_cpu:", num_cpu)
print("batch_size:", args.batch_size)

directed = "directed" if args.directed else "undirected"

print(f"networkx: {nx.__version__}")
print(f"nx_parallel: {nxp.__version__}")
os.environ["NETWORKX_AUTOMATIC_BACKENDS"] = "parallel"
os.environ["RAY_memory_monitor_refresh_ms"] = "500"
nx.config.backends.parallel.active = True
nx.config.backends.parallel.n_jobs = num_cpu
ray.init(num_cpus=num_cpu)


def intersection(g1, g2):
    return set(g1.nodes).intersection(set(g2.nodes))


def get_path_cost(G, p, amount, proto_type="LND"):
    if len(p) < 2:
        return 0
    if proto_type == "DEF":
        return len(p)
    cost = 0
    for u, v in pairwise(p):
        cost += cost_function(G, u, v, amount, proto_type=proto_type)
    return cost


@ray.remote
def route_and_count(g1, g2, sample, alg="DEF", seed=13, verbose=False):
    set_seed(seed)
    matched_paths = 0
    total_paths = 0
    for u, v in sample:
        try:
            g1_path = get_shortest_path(g1, u, v, 100, alg)
            base_path_cost = get_path_cost(g1, g1_path, 100, alg)
            if base_path_cost > 0:
                total_paths += 1
                g2_path = get_shortest_path(g2, u, v, 100, alg)
                new_path_cost = get_path_cost(g2, g2_path, 100, alg)
                if verbose:
                    print(
                        f"Processing pair ({u}, {v}), base_path_cost: {base_path_cost}, new_path_cost: {new_path_cost}"
                    )
                if new_path_cost > 0 and base_path_cost >= new_path_cost:
                    matched_paths += 1
        except:  # noqa: E722
            pass
    return matched_paths, total_paths


def edges_intersection_rate(
    g1,
    g2,
    sample_size,
    batch_size,
    common_nodes=None,
    alg="DEF",
    seed=13,
    verbose=False,
):
    if common_nodes is None:
        common_nodes = intersection(g1, g2)
    tasks = []
    matched_paths = 0
    total_paths = 0
    task_count = 0
    for sample in batched(combinations(common_nodes, 2), sample_size):
        tasks.append(
            route_and_count.remote(ray.put(g1), ray.put(g2), sample, alg, seed, verbose)
        )
        task_count += 1
        if task_count % batch_size == 0:
            for result in ray.get(tasks):
                matched_paths += result[0]
                total_paths += result[1]
            tasks = []
            if verbose:
                print(
                    f"Processed {task_count} samples, matched_paths: {matched_paths}, total_paths: {total_paths}"
                )
    return (
        matched_paths / total_paths if total_paths > 0 else 0,
        matched_paths,
        total_paths,
    )


def proccess_graphs(
    g1,
    g2,
):
    s = get_stamp(g2)
    g1 = nx.read_gml(g1)
    g2 = nx.read_gml(g2)
    intersection_rate, matched_paths, total_paths = edges_intersection_rate(
        g1,
        g2,
        sample_size=10000,
        batch_size=args.batch_size,
        alg=args.alg,
        seed=13,
        verbose=args.verbose,
    )
    return {
        "timestamp": s,
        "edges_intersection_rate": intersection_rate,
        "matched_paths": matched_paths,
        "total_paths": total_paths,
    }


graphs = pd.read_csv(
    os.path.join(results_dir, args.shapes_file), parse_dates=True, index_col=0
)


print([graphs.fname.iloc[0], graphs.fname.iloc[-1]])
for g1, g2 in [(graphs.fname.iloc[0], graphs.fname.iloc[-1])]:
    r = proccess_graphs(
        os.path.join(data_dir, f"{get_stamp(g1)}.{directed}.gml"),
        os.path.join(data_dir, f"{get_stamp(g2)}.{directed}.gml"),
    )
    print(r)
    with open(
        os.path.join(results_dir, f"channel_intersection.{args.alg}.{directed}.json"),
        "w",
    ) as f:
        json.dump(r, f, indent=4)
    break
