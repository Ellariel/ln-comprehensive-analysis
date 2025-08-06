import os, sys
import argparse
import networkx as nx
import pandas as pd
import numpy as np
from littleballoffur import ForestFireSampler


import warnings
warnings.filterwarnings("ignore")

from utils import get_stamp, set_seed, intersection, nodes_intersection_rate, edges_intersection_rate, sample_graph

if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--data_dir', default=None, type=str)
        parser.add_argument('--results_dir', default=None, type=str)
        parser.add_argument('--shapes_file', default="shapes_fix.csv", type=str)
        args = parser.parse_args()
else:
    pass
    #sys.exit()

base_dir = os.path.dirname(__file__)
if 'app' in base_dir:
    base_dir = './'
data_dir = os.path.join(base_dir, "data") if args.data_dir is None else args.data_dir
results_dir = os.path.join(base_dir, "results") if args.results_dir is None else args.results_dir
os.makedirs(results_dir, exist_ok=True)

print('data_dir:', data_dir)
print('results_dir:', results_dir)


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


graphs = pd.read_csv(os.path.join(results_dir, args.shapes_file), parse_dates=True, index_col=0)


g1 = graphs.iloc[0]['fname']
g2 = graphs.iloc[-1]['fname']
print(f"{g1} -> {g2}\n")
r = proccess_graphs(os.path.join(data_dir, g1), os.path.join(data_dir, g2)) 
r = dict(pd.DataFrame([r]).iloc[0])
print(r)

'''
20190120.gml.gz -> 20230716.gml.gz
{'timestamp': '20230716', 
'nodes_intersect_rate': np.float64(0.3087071240105541), 
'edges_intersect_rate': np.float64(0.7328449328449328), 
'sampled_nodes_intersect_rate_mean': np.float64(0.4260000000000001), 
'sampled_edges_intersect_rate_mean': np.float64(0.7194638771143249), 
'sampled_nodes_intersect_rate_std': np.float64(0.049678969393496884), 
'sampled_edges_intersect_rate_std': np.float64(0.05263554837477928)}
'''
pass