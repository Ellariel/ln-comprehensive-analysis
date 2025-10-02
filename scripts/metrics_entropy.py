import os, sys
import argparse
import networkx as nx
import pandas as pd
import numpy as np
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from utils import get_stamp, set_seed, degree_distribution_entropy

if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--data_dir', default=None, type=str)
        parser.add_argument('--results_dir', default=None, type=str)
        parser.add_argument('--shapes_file', default="shapes_fix.csv", type=str)
        args = parser.parse_args()
else:
     sys.exit()

base_dir = os.path.dirname(__file__)
if 'app' in base_dir:
    base_dir = './'
data_dir = os.path.abspath(os.path.join(base_dir, 
                        "..", "data")) if args.data_dir is None else args.data_dir
results_dir = os.path.abspath(os.path.join(base_dir, 
                        "..", "results")) if args.results_dir is None else args.results_dir
os.makedirs(results_dir, exist_ok=True)

print('data_dir:', data_dir)
print('results_dir:', results_dir)

results_file = os.path.join(results_dir, f"degree_entropy_metrics.csv")

def proccess_graphs(g, seed=13):
    set_seed(seed)
    s = get_stamp(g)
    g = nx.read_gml(g).to_undirected()
    h, h_norm = degree_distribution_entropy(g)

    return {
        'timestamp': s,
        'degree_dist_entropy' : h,
        'degree_dist_entropy_norm' : h_norm,
    }


graphs = pd.read_csv(os.path.join(results_dir, args.shapes_file), parse_dates=True, index_col=0)


if os.path.exists(results_file):
    results = pd.read_csv(results_file, dtype=str)
else:
    results = pd.DataFrame()

timestamps = set(results.timestamp) if 'timestamp' in results else set()
for g in tqdm(graphs.fname):
    if get_stamp(g) not in timestamps:
        r = proccess_graphs(os.path.join(data_dir, g)) 
        results = pd.concat([results, pd.DataFrame([r])])
        results.to_csv(results_file, index=False)
