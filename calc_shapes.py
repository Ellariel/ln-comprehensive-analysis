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


def filter_holes(df):
    flag = True
    df = df.copy()
    for idx, item in list(df.iterrows())[1:]:
        pitem = df.iloc[idx - 1]
        check_1 = (len(str(item['nodes'])) > len(str(pitem['nodes']))) or\
                (len(str(item['edges'])) > len(str(pitem['edges'])))
        check_2 = (len(str(item['nodes'])) < len(str(pitem['nodes']))) or\
                (len(str(item['edges'])) < len(str(pitem['edges'])))
        if check_1:
            flag = True
        if check_2:
            flag = False
        df.loc[idx, 'keep'] = flag
    return df


base_dir = os.path.dirname(__file__)
data_dir = os.path.join(base_dir, "data") if args.data_dir is None else args.data_dir
results_dir = os.path.join(base_dir, "results") if args.results_dir is None else args.results_dir
os.makedirs(results_dir, exist_ok=True)
print('data_dir:', data_dir)
print('results_dir:', results_dir)

filelist = [i for i in glob(data_dir + "/*.gml.gz")]
timestamps = [get_stamp(i) for i in filelist]
shapes = [read_shape(i) for i in tqdm(filelist)]
df = pd.concat([pd.Series(timestamps), 
           pd.DataFrame(shapes), 
           pd.Series(filelist)], axis=1)
df.columns = ['date', 'nodes', 'edges', 'fname']
df['date'] = df['date'].apply(lambda x: datetime.strptime(x, '%Y%m%d'))

while True:
    df = filter_holes(df)
    done = not (df['keep'] == False).any()
    df = df[df['keep'] == True].reset_index(drop=True)
    if done:
        break  

df.index = pd.DatetimeIndex(pd.to_datetime(df['date']))
df = df[df.index >= '20.01.2019']
df['fname'] = df['fname'].apply(lambda x: os.path.split(x)[1])
df.to_csv(os.path.join(results_dir, 'shapes.csv'), index=False)