import os, sys
import argparse
import pandas as pd
from datetime import datetime

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from utils import msg_counter


if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--data_dir', default=None, type=str)
        parser.add_argument('--results_dir', default=None, type=str)
        args = parser.parse_args()
else:
     sys.exit()

snapshot = 'gossip-20230924.gsp.bz2'
base_dir = os.path.dirname(__file__)
data_dir = os.path.join(base_dir, "data") if args.data_dir is None else args.data_dir
results_dir = os.path.join(base_dir, "results") if args.results_dir is None else args.results_dir
os.makedirs(results_dir, exist_ok=True)
print('data_dir:', data_dir)
print('results_dir:', results_dir)

node_announcements, channel_updates, channel_announcements = msg_counter(os.path.join(data_dir, snapshot))
df = pd.concat([pd.Series(node_announcements), pd.Series(channel_updates), pd.Series(channel_announcements)], ignore_index=False, axis=1)
df.columns = ['node_announcements', 'channel_updates', 'channel_announcements']

df.reset_index(drop=False, inplace=True)
df.index = pd.DatetimeIndex(df['index'].apply(datetime.fromtimestamp))
df = df[df.index <= '24.09.2023']
df.drop('index', axis=1, inplace=True)
df.to_csv(os.path.join(results_dir, 'messages.csv'), index=True, compression='zip')