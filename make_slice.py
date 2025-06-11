import os
import ray
import logging
import networkx as nx
from glob import glob
from itertools import batched
from datetime import datetime, timedelta

from utils import restore_graph, get_stamp



logging.basicConfig(level=logging.INFO, filename="processing.log")

def log(s):
    print(s)
    logging.info('%s', s)

num_cpu = 10
twoweeks = 130
base_dir = "./data"

ray.init(num_cpus=num_cpu)
#os.environ["RAY_DEDUP_LOGS"] = "0"

@ray.remote
def proccess_timestamp(file_path, timestamp):
    g = restore_graph(file_path, timestamp.timestamp(), verbose=False)
    #print(f'nodes: {len(g.nodes)}, channels: {len(g.edges)}')
    if len(g.nodes) >= 0 and len(g.edges) >= 0:
        nx.write_gml(g, os.path.join(base_dir, f"{timestamp.strftime('%Y%m%d')}.gml.gz"))
    return {timestamp.strftime('%Y%m%d'): (len(g.nodes), len(g.edges))}

for file_path in sorted(glob(base_dir + "/*.gsp.bz2"), reverse=True):
    log(f'start file: {file_path}')
    time_stamp = datetime.strptime(get_stamp(file_path).split('-')[1], '%Y%m%d')
    stop_date = [get_stamp(i) for i in glob(base_dir + "/*.gml.gz")]
    log(f'already done: {stop_date}')
    planned = []
    for i in range(0, twoweeks):
        ts = time_stamp - timedelta(weeks=2 * i)
        if not ts.strftime('%Y%m%d') in stop_date:
            planned.append(ts)

    log(f'planned: {[i.strftime('%d.%m.%Y') for i in planned]}')

    results = []
    for batch in batched(planned, num_cpu):
        log(f"started new batch: {[i.strftime('%d.%m.%Y') for i in batch]}")
        results += ray.get([proccess_timestamp.remote(file_path, i) for i in batch])
        log('done.')

    results = {k: v for i in results for k, v in i.items()}
    log(f'planned: {results}')

