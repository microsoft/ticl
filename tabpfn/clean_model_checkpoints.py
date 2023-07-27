import glob
import re
import os
from collections import defaultdict
import torch

file_dict = defaultdict(list)
for name in glob.glob("models_diff/*.cpkt"):
    shortname = name.split("/")[1]
    canonical_name = shortname.split("epoch")[0].strip("_")
    file_dict[canonical_name].append(name)


canonical_file = dict()
for canonical_name, files in file_dict.items():
    last_epoch = -1
    for f in files:
        if "on_exit" in f:
            # we don't count that one as the final one because it doesn't have the optimizer state
            continue
        epoch = int(re.findall("(\d+)", f.split("epoch")[1])[0])
        if epoch > last_epoch:
            last_epoch = epoch
            canonical_file[canonical_name] = f
            
for canonical_name, files in file_dict.items():
    if canonical_name not in canonical_file:
        # we only have "on exit" because we cleaned up before
        continue
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