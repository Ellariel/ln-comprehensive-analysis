import os, sys
import glob
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


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

print('data_dir:', data_dir)
print('results_dir:', results_dir)

results_file = os.path.join(results_dir, "metrics.csv")

metrics = ['nodes',
        'edges',
        'components',
        'density',
        'bridges',
        'diameter',
        'transitivity',
        'average_clustering',
        'degree_assortativity',
        'burt_effective_size',
        'effective_size',
        'min_edge_cover',
        'global_efficiency',
        'mean_degree',
        'mean_betweenness_centrality',
        'gini_betweenness_centrality',
        'resource_allocation_index',
        'jaccard_coefficient',
        'preferential_attachment',
        'common_neighbor_centrality',
        'shortest_path_length',
        'information_centrality',
        'label_communities',
        'label_non_randomness',
        'label_relative_non_randomness',
        'lpa_communities',
        'lpa_non_randomness',
        'lpa_relative_non_randomness',
        'average_node_connectivity',
        'constraint',
        'closeness_vitality',
        'communicability_betweenness_centrality',
        'wiener_index',
        'ks_stat',
        'ks_p',
        'wasserstein_distance',
]

exclude = ['messages.csv', 'shapes.csv', 'shapes_fix.csv', "metrics.csv", 'base_metrics.csv']
files = [i for i in glob.glob(os.path.join(results_dir, "*.csv")) if os.path.basename(i) not in exclude]
data = pd.read_csv(os.path.join(results_dir, 'base_metrics.csv'), index_col=0)

for f in files:
    df = pd.read_csv(f)
    if 'timestamp' in df:
        data = data.join(df.set_index('timestamp'))

data['datetime'] = pd.to_datetime(data.index, format='%Y%m%d')
data = data[['datetime'] + list(data.columns[:-1])].reset_index()
data['timestamp'] = data['datetime'].apply(lambda x:  int(x.timestamp()))

data.to_csv(results_file, index=False)

metrics_done = set(data.columns) - set(['timestamp', 'datetime'])
print(f'metrics done {len(metrics_done)}:')
for i in metrics_done:
    print('  ', i)

metrics_needed = set(metrics) - metrics_done
if len(metrics_needed):
    print(f'metrics needed {len(metrics_needed)}: {metrics_needed}\n')

done = sum([1 for i in data.columns if i in metrics])
print(f"\n{len(metrics_done)} items, {done}/{len(metrics)} metrics, {done*100/len(metrics):.1f}%")

print('less than 95% complete:')
for i in data.columns:
    if data[i].notna().sum() < len(data) * 0.95:
        print('  ', i)



