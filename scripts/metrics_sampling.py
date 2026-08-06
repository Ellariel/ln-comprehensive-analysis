import pickle
import os, sys
import argparse
import networkx as nx
import pandas as pd
from tqdm import tqdm
from littleballoffur import ForestFireSampler

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from utils import (
    get_stamp,
    set_seed,
)

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

sampled_dir = os.path.abspath(os.path.join(results_dir, "sampled"))
os.makedirs(sampled_dir, exist_ok=True)

print("data_dir:", data_dir)
print("results_dir:", results_dir)


directed = "directed" if args.directed else "undirected"


def sample_graph(g, sampler=None):
    g = nx.convert_node_labels_to_integers(g, label_attribute="old_id")
    s = sampler.sample(g)
    mapping = {k: g.nodes[k]["old_id"] for k in s.nodes()}
    return nx.relabel_nodes(s, mapping)


def proccess_graph(g, sample_size=100, seed=13):
    set_seed(seed)
    g = nx.read_gml(g)
    sampler = ForestFireSampler(number_of_nodes=sample_size, seed=seed)
    samples = [sample_graph(g, sampler) for _ in range(sample_size)]
    return samples


graphs = pd.read_csv(
    os.path.join(results_dir, args.shapes_file), parse_dates=True, index_col=0
)

if directed == "undirected":
    for f in tqdm(graphs.fname):
        timestamp = get_stamp(f)
        sample = proccess_graph(os.path.join(data_dir, f"{timestamp}.{directed}.gml"))
        with open(
            os.path.join(sampled_dir, f"{timestamp}.{directed}.pkl"), "wb"
        ) as handle:
            pickle.dump(sample, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    for f in tqdm(graphs.fname):
        timestamp = get_stamp(f)
        g = nx.read_gml(os.path.join(data_dir, f"{timestamp}.{directed}.gml"))
        with open(
            os.path.join(sampled_dir, f"{timestamp}.undirected.pkl"), "rb"
        ) as handle:
            sample = [g.subgraph(i.nodes).copy() for i in pickle.load(handle)]
            with open(
                os.path.join(sampled_dir, f"{timestamp}.{directed}.pkl"), "wb"
            ) as handle:
                pickle.dump(sample, handle, protocol=pickle.HIGHEST_PROTOCOL)
