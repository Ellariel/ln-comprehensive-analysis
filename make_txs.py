import os, sys
import argparse
import pickle
import random
import networkx as nx
import pandas as pd
from tqdm import tqdm
from glob import glob

from proto import gen_txset, random_amount, MIN_AGE, MAX_AGE

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

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

graphs = pd.read_csv(os.path.join(results_dir, 'shapes_fix.csv'), parse_dates=True, index_col=0)

results = [get_stamp(i) for i in glob(data_dir + "/*.txs.pkl")]

for g in tqdm(graphs.fname):
    set_seed()
    timestamp = get_stamp(g)
    if timestamp not in results:
        g = nx.read_gml(os.path.join(data_dir, g)).to_undirected()
        txset = gen_txset(g, 10000)
        assert len(txset) == 10000

        for e in g.edges:
            e = g.edges[e]
            e['fee_base_sat'] = float(e['fee_base_msat']) / 1000
            e['fee_rate_sat'] = float(e['fee_proportional_millionths']) / 10**6
            if 'htlc_maximum_msat' not in e:
                e['htlc_maximum_msat'] = random_amount() * 1000 + float(e['htlc_minimim_msat'])
            else:
                e['htlc_maximum_msat'] = float(e['htlc_maximum_msat'])
            e['capacity_sat'] = e['htlc_maximum_msat'] / 1000
            e['delay'] = float(e['cltv_expiry_delta'])
            e['age'] = random.randrange(MIN_AGE, MAX_AGE)

        with open(os.path.join(data_dir, f'{timestamp}.txs.pkl'), 'wb') as handle:
            pickle.dump((g, txset), handle, protocol=pickle.HIGHEST_PROTOCOL)
        results.append(timestamp)


