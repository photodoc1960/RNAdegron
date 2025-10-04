import os
import numpy as np
from glob import glob

bad_files = []

for fpath in glob('./data/*_bpp.npy'):
    try:
        bpp = np.load(fpath, allow_pickle=True)
        if bpp.shape[-1] != bpp.shape[-2] or bpp.shape[-1] > 130:
            print(f"Deleting: {fpath} | shape: {bpp.shape}")
            os.remove(fpath)
            bad_files.append(fpath)
    except Exception as e:
        print(f"Error loading {fpath}: {e} -- Deleting.")
        os.remove(fpath)
        bad_files.append(fpath)

print(f"Deleted {len(bad_files)} corrupt BPP files.")