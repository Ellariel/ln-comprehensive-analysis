import os
import requests
from tqdm import tqdm


base_dir = os.path.dirname(__file__)
data_dir = os.path.join(base_dir, "data")

dates = ['20201014', '20201102', '20201203',
        '20210104', '20210908', '20220823', '20230924']
files = [f'https://storage.googleapis.com/lnresearch/gossip-{f}.gsp.bz2' 
         for f in dates]
for f in tqdm(files):
    file_name = os.path.join(data_dir, os.path.basename(f))
    if not os.path.exists(file_name):
        r = requests.get(f)
        r.raise_for_status()
        with open(file_name, "wb") as file:
            file.write(r.content)
