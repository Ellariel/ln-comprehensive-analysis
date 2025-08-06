import os, sys
import argparse
import pickle
import networkx as nx
import pandas as pd
from tqdm import tqdm

from proto import get_shortest_path

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from utils import get_stamp, set_seed

def cum(paths):
    s = pd.Series(paths).value_counts().reset_index(drop=True) * 100 / len(paths)
    return s.cumsum()

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
data_dir = os.path.join(base_dir, "data") if args.data_dir is None else args.data_dir
results_dir = os.path.join(base_dir, "results") if args.results_dir is None else args.results_dir
os.makedirs(results_dir, exist_ok=True)
results_file = os.path.join(results_dir, f"tx_metrics.pkl")

print('data_dir:', data_dir)
print('results_dir:', results_dir)


graphs = pd.read_csv(os.path.join(results_dir, args.shapes_file), parse_dates=True, index_col=0)


if os.path.exists(results_file):
    with open(os.path.join(results_dir, results_file), 'rb') as handle:
        results = pickle.load(handle)
else:
    results = {}

for g in tqdm(graphs.fname):
    set_seed()
    timestamp = get_stamp(g)
    if timestamp not in results:
        with open(os.path.join(data_dir, f'{timestamp}.txs.pkl'), 'rb') as handle:
            g, txset = pickle.load(handle)
        assert len(g.nodes) > 500
        assert len(txset) == 10000
        paths = []
        for tx in tqdm(txset[:5000], leave=False):
            paths += get_shortest_path(g, tx[0], tx[1], tx[2], 'LND')
            paths += get_shortest_path(g, tx[0], tx[1], tx[2], 'CLN')
            paths += get_shortest_path(g, tx[0], tx[1], tx[2], 'ECL')
        assert len(paths) >= 5000 * 2 * 3
        results[timestamp] = cum(paths)

        with open(os.path.join(results_dir, results_file), 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)




