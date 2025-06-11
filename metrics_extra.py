import os, sys
import argparse
import networkx as nx
import pandas as pd
import numpy as np
from tqdm import tqdm
from itertools import pairwise
from littleballoffur import ForestFireSampler

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from utils import get_stamp, set_seed, intersection, nodes_intersection_rate, edges_intersection_rate, sample_graph

if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--data_dir', default=None, type=str)
        parser.add_argument('--results_dir', default=None, type=str)
        args = parser.parse_args()
else:
     sys.exit()

base_dir = os.path.dirname(__file__)
if 'app' in base_dir:
    base_dir = './'
data_dir = os.path.join(base_dir, "data") if args.data_dir is None else args.data_dir
results_dir = os.path.join(base_dir, "results") if args.results_dir is None else args.results_dir
os.makedirs(results_dir, exist_ok=True)

print('data_dir:', data_dir)
print('results_dir:', results_dir)

results_file = os.path.join(results_dir, f"extra_metrics.csv")

def proccess_graphs(g1, g2, seed=13):
    set_seed(seed)
    s = get_stamp(g2)
    g1 = nx.read_gml(g1).to_undirected()
    g2 = nx.read_gml(g2).to_undirected()

    isect = intersection(g1, g2)  
    sampler = ForestFireSampler(number_of_nodes=100, seed=seed)
    samples = [sample_graph(g1, sampler) for _ in range(100)]

    nodes_intersect_rate = nodes_intersection_rate(g1, g2, isect)
    edges_intersect_rate = edges_intersection_rate(g1, g2, isect)
    sampled_nodes_intersect_rate = [nodes_intersection_rate(g, g2) for g in samples]
    sampled_edges_intersect_rate = [edges_intersection_rate(g, g2) for g in samples]

    return {
        'timestamp': s,
        'nodes_intersect_rate' : nodes_intersect_rate,
        'edges_intersect_rate' : edges_intersect_rate,
        'sampled_nodes_intersect_rate_mean' : np.mean(sampled_nodes_intersect_rate),
        'sampled_edges_intersect_rate_mean' : np.mean(sampled_edges_intersect_rate),
        'sampled_nodes_intersect_rate_std' : np.std(sampled_nodes_intersect_rate),
        'sampled_edges_intersect_rate_std' : np.std(sampled_edges_intersect_rate),
    }


graphs = pd.read_csv(os.path.join(results_dir, 'shapes_fix.csv'), parse_dates=True, index_col=0)


if os.path.exists(results_file):
    results = pd.read_csv(results_file, dtype=str)
else:
    results = pd.DataFrame()

timestamps = set(results.timestamp) if 'timestamp' in results else set()
for g1, g2 in tqdm(pairwise(graphs.fname), total=len(graphs.fname)-1):
    if get_stamp(g2) not in timestamps:
        r = proccess_graphs(os.path.join(data_dir, g1), os.path.join(data_dir, g2)) 
        results = pd.concat([results, pd.DataFrame([r])])
        results.to_csv(results_file, index=False)
