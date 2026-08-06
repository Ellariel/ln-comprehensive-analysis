# STEP 4

import os
from glob import glob
import numpy as np
import networkx as nx
import pandas as pd
from tqdm import tqdm
from datetime import datetime

import warnings

warnings.filterwarnings("ignore")

from utils import get_stamp, set_seed
from proto import prepare_graph


def read_shape_and_degree(fname):
    g = nx.read_gml(fname)
    return len(g.nodes), len(g.edges), np.mean(list(dict(g.degree).values()))


base_dir = os.path.dirname(__file__)
data_dir = os.path.abspath(os.path.join(base_dir, "..", "data"))
results_dir = os.path.abspath(os.path.join(base_dir, "..", "results"))
os.makedirs(results_dir, exist_ok=True)


def proccess_graph(source_file, results_file, seed=13):
    set_seed(seed)
    g = nx.read_gml(source_file)
    zero_capacity_edges = [
        e for e in g.edges if int(g.edges[e].get("htlc_maximum_msat", 0)) < 1
    ]
    g.remove_edges_from(zero_capacity_edges)
    zero_degree_nodes = [n[0] for n in g.degree if int(n[1]) < 1]
    g.remove_nodes_from(zero_degree_nodes)
    g = prepare_graph(g, seed=seed)
    nx.write_gml(g, f"{results_file}.directed.gml")
    nx.write_gml(g.to_undirected(), f"{results_file}.undirected.gml")


graphs = pd.read_csv(
    os.path.join(results_dir, "shapes_fix.csv"), parse_dates=True, index_col=0
)

for f in tqdm(graphs.fname):
    source_file = os.path.join(data_dir, f.replace(".fix", ""))
    results_file = os.path.join(data_dir, f"{get_stamp(f)}")
    if not os.path.exists(results_file):
        proccess_graph(source_file, results_file)

# update shapes
filelist = [i for i in glob(data_dir + "/*.directed.gml")]
timestamps = [get_stamp(i) for i in filelist]
shapes = [read_shape_and_degree(i) for i in tqdm(filelist)]
df = pd.concat(
    [
        pd.Series(timestamps),
        pd.DataFrame(shapes),
        pd.Series(filelist),
    ],
    axis=1,
)
df.columns = [
    "datetime",
    "nodes",
    "edges",
    "degree",
    "fname",
]
df["datetime"] = df["datetime"].apply(lambda x: datetime.strptime(x, "%Y%m%d"))
df.index = pd.DatetimeIndex(pd.to_datetime(df["datetime"]))
df = df[df.index >= "20.01.2019"]
df["fname"] = df["fname"].apply(lambda x: os.path.split(x)[1])
df["datetime"] = pd.to_datetime(df.index)
df["timestamp"] = df["datetime"].apply(lambda x: int(x.timestamp()))
df = df[
    [
        "timestamp",
        "datetime",
        "nodes",
        "edges",
        "degree",
        "fname",
    ]
].rename(columns={"edges": "channels"})

df.to_csv(os.path.join(results_dir, "shapes_directed.csv"), index=False)
