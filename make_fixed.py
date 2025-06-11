import os, sys
import argparse
import networkx as nx
import pandas as pd
import numpy as np
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

from utils import get_stamp, set_seed

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



def proccess_graph(g, seed=13):
    set_seed(seed)
    f = os.path.join(data_dir, g)
    s = get_stamp(f)
    g = nx.read_gml(f).to_undirected()

    zero_capacity_edges = [e for e in g.edges 
                                if int(g.edges[e].get('htlc_maximum_msat', 0)) < 1]
    g.remove_edges_from(zero_capacity_edges)
    zero_degree_nodes = [n[0] for n in g.degree 
                                if int(n[1]) < 1]
    g.remove_nodes_from(zero_degree_nodes)

    nx.write_gml(g, os.path.join(data_dir, f"{s}.gml.gz.fix"))



graphs = pd.read_csv(os.path.join(results_dir, 'shapes.csv'), parse_dates=True, index_col=0)
graphs.drop(['keep'], axis=1, inplace=True)

for g in tqdm(graphs.fname):
    results_file = os.path.join(data_dir, f"{get_stamp(g)}.gml.gz.fix")
    if not os.path.exists(results_file):
        proccess_graph(g)
    
