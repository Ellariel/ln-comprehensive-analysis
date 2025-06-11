import os, sys
import argparse
import pandas as pd
from glob import glob
from tqdm import tqdm
from datetime import datetime

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from utils import read_shape, get_stamp


if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--data_dir', default=None, type=str)
        parser.add_argument('--results_dir', default=None, type=str)
        args = parser.parse_args()
else:
     sys.exit()



base_dir = os.path.dirname(__file__)
data_dir = os.path.join(base_dir, "data") if args.data_dir is None else args.data_dir
results_dir = os.path.join(base_dir, "results") if args.results_dir is None else args.results_dir
os.makedirs(results_dir, exist_ok=True)
print('data_dir:', data_dir)
print('results_dir:', results_dir)

fix = pd.read_csv(os.path.join(data_dir, 'fixed.txt'), 
                  header=None, dtype=str)[0].to_list()
filelist = [i for i in glob(data_dir + "/*.gml.gz.fix")]
filelist = [i for i in filelist if get_stamp(i) not in fix]

timestamps = [get_stamp(i) for i in filelist]
shapes = [read_shape(i) for i in tqdm(filelist)]
df = pd.concat([pd.Series(timestamps), 
           pd.DataFrame(shapes), 
           pd.Series(filelist)], axis=1)
df.columns = ['date', 'nodes', 'edges', 'fname']
df['date'] = df['date'].apply(lambda x: datetime.strptime(x, '%Y%m%d'))


df['fname'] = df['fname'].apply(lambda x: os.path.split(x)[1])
df.to_csv(os.path.join(results_dir, 'shapes_fix.csv'), index=False)