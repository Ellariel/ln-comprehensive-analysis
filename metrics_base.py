import os, sys
import ray
import math
import argparse
import numpy as np
import networkx as nx
import nx_parallel as nxp
import pandas as pd
from tqdm import tqdm
from itertools import batched

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_cpu', default=35, type=int)
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--data_dir', default=None, type=str)
    parser.add_argument('--results_dir', default=None, type=str)
    parser.add_argument('--shapes_file', default="shapes_fix.csv", type=str)
    args = parser.parse_args()
else:
     sys.exit()

from utils import *

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

results_file = os.path.join(results_dir, "base_metrics.csv")


print(f"networkx: {nx.__version__}")
print(f"nx_parallel: {nxp.__version__}")
os.environ['NETWORKX_AUTOMATIC_BACKENDS'] = "parallel"
os.environ['RAY_memory_monitor_refresh_ms'] = '500'
# nx.config.backends.parallel.active = True
# nx.config.backends.parallel.n_jobs = num_cpu
ray.init(num_cpus=num_cpu)


@ray.remote
def proccess_graph(g, seed=13):
    set_seed(seed)
    s = get_stamp(g)
    ug = nx.read_gml(g).to_undirected()
    components = [ug.subgraph(c) 
                    for c in sorted(nx.connected_components(ug), 
                                    key=len, reverse=True) if len(c) > 10]
    ug_ref = ray.put(ug)
    first_component_ref = ray.put(components[0])
    metrics = {
        'timestamp': s,
        'nodes' : len(ug.nodes),
        'edges' : len(ug.edges),
        'components' : len(components),
        'density' : ray.remote(density).remote(ug_ref),
        'bridges' : ray.remote(bridges).remote(ug_ref),
        'diameter' : np.max([ray.get(ray.remote(diameter_approximation).remote(ray.put(c), seed=seed)) 
                                        for c in components]),
        'transitivity' : ray.remote(transitivity).remote(ug_ref),
        'average_clustering' : ray.remote(average_clustering).remote(ug_ref),
        'degree_assortativity' : ray.remote(degree_assortativity).remote(ug_ref),

        #'burt_effective_size' : burt_effective_size.remote(ug_ref),
        #'effective_size' : effective_size.remote(ug_ref),
        #'min_edge_cover' : min_edge_cover.remote(ug_ref),
        #'global_efficiency' : global_efficiency.remote(ug_ref),
        #'mean_degree' : mean_degree.remote(ug_ref),
        #'constraint' : constraint.remote(ug_ref),
        #'average_node_connectivity' : average_node_connectivity.remote(ug_ref),
        #'mean_betweenness_centrality' : mean_betweenness_centrality.remote(ug_ref),
        #'gini_betweenness_centrality' : gini_betweenness_centrality.remote(ug_ref),
        #'resource_allocation_index' : resource_allocation_index.remote(ug_ref),
        #'jaccard_coefficient' : jaccard_coefficient.remote(ug_ref),
        #'preferential_attachment' : preferential_attachment.remote(ug_ref),
        #'common_neighbor_centrality' : common_neighbor_centrality.remote(ug_ref),
        #'shortest_path_length' : shortest_path_length_approximation.remote(ug_ref),
        #'closeness_vitality' : closeness_vitality.remote(first_component_ref),
        #'information_centrality' : information_centrality.remote(first_component_ref),
        #'communicability_betweenness_centrality' : communicability_betweenness_centrality.remote(first_component_ref),
        
        'label_communities' : ray.remote(label_communities).remote(first_component_ref, seed=seed),
        'lpa_communities' : ray.remote(lpa_communities).remote(first_component_ref, seed=seed),
    }

    for k, v in metrics.items():
        if isinstance(v, ray.ObjectRef):
            metrics[k] = ray.get(v)

    metrics.update({
        'label_non_randomness' : ray.remote(non_randomness).remote(first_component_ref, 
                                            k=metrics['label_communities']),
    })

    if metrics['label_communities'] != metrics['lpa_communities']: 
        metrics.update({
            'lpa_non_randomness' : ray.remote(non_randomness).remote(first_component_ref, 
                                            k=metrics['lpa_communities']),
        })

    for k, v in metrics.items():
        if isinstance(v, ray.ObjectRef):
            metrics[k] = ray.get(v)

    if metrics['label_communities'] == metrics['lpa_communities']: 
        metrics['lpa_non_randomness'] = metrics['label_non_randomness']

    metrics.update({
        'label_relative_non_randomness' : metrics['label_non_randomness'][1],
        'label_non_randomness' : metrics['label_non_randomness'][0],
        'lpa_relative_non_randomness' : metrics['lpa_non_randomness'][1],
        'lpa_non_randomness' : metrics['lpa_non_randomness'][0],
    })

    return metrics


graphs = pd.read_csv(os.path.join(results_dir, args.shapes_file), parse_dates=True, index_col=0)

if os.path.exists(results_file):
    results = pd.read_csv(results_file, dtype=str)
else:
    results = pd.DataFrame()

timestamps = set(results.timestamp) if 'timestamp' in results else set()
for batch in tqdm(batched(graphs.fname, batch_size), total=math.ceil(len(graphs.fname)/batch_size)):
    batch = [proccess_graph.remote(os.path.join(data_dir, g)) 
                for g in batch if get_stamp(g) not in timestamps]       
    if len(batch):
        results = pd.concat([results, pd.DataFrame(ray.get(batch))])
        results.to_csv(results_file, index=False)
