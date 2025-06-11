import os, sys
import ray
import math
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import networkx as nx
import nx_parallel as nxp
from itertools import batched

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from utils import *

if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--data_dir', default=None, type=str)
        parser.add_argument('--results_dir', default=None, type=str)
        parser.add_argument('--num_cpu', default=35, type=int)
        parser.add_argument('--batch_size', default=10, type=int)
        parser.add_argument('--metric', default=0, type=str)
        args = parser.parse_args()
else:
     sys.exit()

base_dir = os.path.dirname(__file__)
if 'app' in base_dir:
    base_dir = './'
data_dir = os.path.join(base_dir, "data") if args.data_dir is None else args.data_dir
results_dir = os.path.join(base_dir, "results") if args.results_dir is None else args.results_dir
os.makedirs(results_dir, exist_ok=True)
cpu_count = max(1, os.cpu_count() - 2)
num_cpu = min(args.num_cpu, cpu_count)
batch_size = min(args.batch_size, cpu_count)
print('data_dir:', data_dir)
print('results_dir:', results_dir)
print('num_cpu:', num_cpu)
print('batch_size:', batch_size)

print(f"networkx: {nx.__version__}")
print(f"nx_parallel: {nxp.__version__}")
os.environ['NETWORKX_AUTOMATIC_BACKENDS'] = "parallel"
os.environ['RAY_memory_monitor_refresh_ms'] = '0'
# nx.config.backends.parallel.active = True
# nx.config.backends.parallel.n_jobs = num_cpu

metrics = {
        'mean_degree' : mean_degree,
        'burt_effective_size' : burt_effective_size,
        'effective_size' : effective_size,
        'min_edge_cover' : min_edge_cover,
        'global_efficiency' : global_efficiency,
        'average_node_connectivity' : average_node_connectivity,
        'mean_betweenness_centrality' : mean_betweenness_centrality,
        'resource_allocation_index' : resource_allocation_index,
        'jaccard_coefficient' : jaccard_coefficient,
        'preferential_attachment' : preferential_attachment,
        'common_neighbor_centrality' : common_neighbor_centrality,
        'shortest_path_length' : shortest_path_length_approximation,
        'information_centrality' : information_centrality, # first component
        'gini_betweenness_centrality' : gini_betweenness_centrality,
        'constraint' : constraint,
        'communicability_betweenness_centrality' : communicability_betweenness_centrality,
        'closeness_vitality' : closeness_vitality, # first component
        'wiener_index' : wiener_index,
    }

if str(args.metric).isnumeric():
    m = list(metrics.items())[int(args.metric)]
    print('\nmetric:', m[0])
else:
    if args.metric in metrics:
        m = (args.metric, metrics[args.metric])
        print('metric:', m[0], '\n')
    else:
        print('no metric defined.\n')
        sys.exit() 

ray.init(num_cpus=num_cpu)

results_file = os.path.join(results_dir, f"{m[0]}.csv")

@ray.remote
def proccess_graph(g, m, seed=13):
    set_seed(seed)
    s = get_stamp(g)
    ug = nx.read_gml(g).to_undirected()
    r = m[1](ug)
    if r is None:
        print('..trying the first component..')
        components = [ug.subgraph(c) 
                            for c in sorted(nx.connected_components(ug), 
                                    key=len, reverse=True) if len(c) > 10]
        if len(components):
            r = m[1](components[0])
    return {
        'timestamp': s,
        m[0]: r,
    }


graphs = pd.read_csv(os.path.join(results_dir, 'shapes_fix.csv'), parse_dates=True, index_col=0)

if os.path.exists(results_file):
    results = pd.read_csv(results_file, dtype=str)
else:
    results = pd.DataFrame()

timestamps = set(results.timestamp) if 'timestamp' in results else set()
for batch in tqdm(batched(graphs.fname, batch_size), total=math.ceil(len(graphs.fname)/batch_size)):
    batch = [proccess_graph.remote(os.path.join(data_dir, g), m) 
                for g in batch if get_stamp(g) not in timestamps]       
    if len(batch):
        results = pd.concat([results, pd.DataFrame(ray.get(batch))])
        results.to_csv(results_file, index=False)
