import os, sys
import json
import argparse
import networkx as nx
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from utils import get_stamp, set_seed


def intersection(g1, g2):
    return set(g1.nodes).intersection(set(g2.nodes))


def nodes_intersection_rate(g1, g2, common_nodes=None, use_capacity=False):
    if common_nodes is None:
        common_nodes = intersection(g1, g2)
    if not use_capacity:
        return len(common_nodes) / len(g1.nodes)
    else:
        total_capacity_g1 = np.sum(
            [d["capacity_sat"] / 2 for u, v, d in g1.edges(g1.nodes, data=True)]
        )
        total_capacity_g2 = np.sum(
            [d["capacity_sat"] / 2 for u, v, d in g2.edges(g2.nodes, data=True)]
        )

        shared_capacity_g1 = np.sum(
            [d["capacity_sat"] / 2 for u, v, d in g1.edges(common_nodes, data=True)]
        )
        shared_capacity_g2 = np.sum(
            [d["capacity_sat"] / 2 for u, v, d in g2.edges(common_nodes, data=True)]
        )
        return (shared_capacity_g2 / total_capacity_g2) / (
            shared_capacity_g1 / total_capacity_g1
        ), (shared_capacity_g2 / total_capacity_g2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None, type=str)
    parser.add_argument("--results_dir", default=None, type=str)
    parser.add_argument("--shapes_file", default="shapes_fix.csv", type=str)
    parser.add_argument("--directed", default=0, type=int)
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


def proccess_graphs(g1, g2, seed=13):
    set_seed(seed)
    s = get_stamp(g2)
    g1 = nx.read_gml(g1)
    g2 = nx.read_gml(g2)
    isect = intersection(g1, g2)
    nodes_intersect_rate = nodes_intersection_rate(g1, g2, isect)
    nodes_capacity_intersect_rate, shared_capacity_share = nodes_intersection_rate(
        g1, g2, isect, use_capacity=True
    )
    return {
        "timestamp": s,
        "nodes_intersect_rate": nodes_intersect_rate,
        "nodes_capacity_intersect_rate": nodes_capacity_intersect_rate,
        "shared_capacity_share": shared_capacity_share,
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
        os.path.join(results_dir, f"node_intersection.{directed}.json"), "w"
    ) as f:
        json.dump(r, f, indent=4)
    break
