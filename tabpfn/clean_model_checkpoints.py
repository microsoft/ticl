import glob
import re
import os
from collections import defaultdict
import torch

file_dict = defaultdict(list)
for name in glob.glob("models_diff/*.cpkt"):
    shortname = name.split("/")[1][len("prior_diff_real_checkpoint_"):-5]
    canonical_name = shortname.split("epoch")[0][:-5]
    file_dict[canonical_name].append(name)


canonical_file = dict()
for canonical_name, files in file_dict.items():
    last_epoch = -1
    for f in files:
        if "on_exit" in f:
            canonical_file[canonical_name] = f
            break
        epoch = int(re.findall("(\d+)", f.split("epoch")[1])[0])
        if epoch > last_epoch:
            last_epoch = epoch
            canonical_file[canonical_name] = f
            
for canonical_name, files in file_dict.items():
    print(f"Loading {canonical_file[canonical_name]}")
    try:
        torch.load(canonical_file[canonical_name])
    except Exception as e:
        print(f"Error loading {canonical_file[canonical_name]}")
        print(e)
        continue
    for file in files:
        if file != canonical_file[canonical_name]:
            os.remove(file)