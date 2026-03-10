# NOTE: run from vfb_new, or at least some env with a newer numpy version

import pickle
import h5py
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('pkl_path', type=str)
args = parser.parse_args()

with open(args.pkl_path, 'rb') as f:
    data = pickle.load(f)

out_path = os.path.splitext(args.pkl_path)[0] + '.h5'
with h5py.File(out_path, 'w') as f:
    for k, v in data.items():
        f.create_dataset(k, data=np.array(v))

print("Keys:", list(data.keys()))
print("Shapes:", {k: np.array(v).shape for k, v in data.items()})
print("Saved to:", out_path)
