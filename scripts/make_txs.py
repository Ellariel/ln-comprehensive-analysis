import os, sys
import argparse
import pickle
import random
import networkx as nx
import pandas as pd
from tqdm import tqdm
from glob import glob

from proto import gen_txset

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from utils import get_stamp, set_seed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None, type=str)
    parser.add_argument("--results_dir", default=None, type=str)
    parser.add_argument("--directed", default=0, type=int)
    parser.add_argument("--shapes_file", default="shapes_fix.csv", type=str)
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
graphs = pd.read_csv(
    os.path.join(results_dir, args.shapes_file), parse_dates=True, index_col=0
)

results = [get_stamp(i) for i in glob(data_dir + f"/*.txs.{directed}.pkl")]

for g in tqdm(graphs.fname):
    set_seed()
    timestamp = get_stamp(g)
    if timestamp not in results:
        g = nx.read_gml(os.path.join(data_dir, f"{get_stamp(g)}.{directed}.gml"))
        txset = gen_txset(g, 10000)
        assert len(txset) == 10000

        with open(
            os.path.join(data_dir, f"{timestamp}.txs.{directed}.pkl"), "wb"
        ) as handle:
            pickle.dump((g, txset), handle, protocol=pickle.HIGHEST_PROTOCOL)
        results.append(timestamp)
