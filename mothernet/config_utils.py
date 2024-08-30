from collections.abc import MutableMapping
from collections import defaultdict
import torch


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError("Boolean value expected.")


def flatten_dict(
    dictionary, 
    parent_key='', 
    separator='_', 
    only_last=False, 
    track_keys = False
):
    if "distribution" in dictionary:
        if track_keys:
            return {parent_key: dictionary}, {}
        else:
            return {parent_key: dictionary}
    if only_last:
        parent_key = ""
    items = []

    if track_keys: track_key_dict = {}
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if track_keys: track_key_dict[key] = new_key
        if isinstance(value, MutableMapping):
            output = flatten_dict(
                value, 
                new_key, 
                separator=separator, 
                only_last=only_last,
                track_keys=track_keys,
            )
            if track_keys:
                new_flat_dict, new_track_key_dict = output
                track_key_dict.update(new_track_key_dict)
            else:
                new_flat_dict = output
        
            items.extend(new_flat_dict.items())
        else:
            items.append((new_key, value))

    if track_keys:
        return dict(items), track_key_dict
    else:
        return dict(items)


def compare_dicts(left, right, prefix=None, skip=None, return_bool=False):
    skip = skip or {}

    prefix = prefix or ""
    for k in set(left).union(set(right)):
        if k in skip:
            continue
        if k not in left:
            if return_bool:
                return False
            print(f"{prefix}{k} missing in left")
            continue
        if k not in right:
            if return_bool:
                return False
            print(f"{prefix}{k} missing in right")
            continue
        if isinstance(left[k], dict):
            res = compare_dicts(left[k], right[k], prefix=f"{prefix}{k}->", skip=skip, return_bool=return_bool)
            if return_bool and not res:
                return False
        else:
            if (torch.is_tensor(left[k]) and (left[k] != right[k]).all()) or (not torch.is_tensor(left[k]) and left[k] != right[k]):
                if return_bool:
                    return False
                print(f"{prefix}{k}: left: {left[k]}, right: {right[k]}")
    if return_bool:
        return True


def merge_dicts(*dicts):
    keys = set([k for d in dicts for k in d])
    merged = {}
    for k in keys:
        values = [d[k] for d in dicts if k in d]
        if len(values) == 1:
            merged[k] = values[0]
        elif all([isinstance(v, dict) for v in values]):
            merged[k] = merge_dicts(*values)
        else:
            raise ValueError(f"Can't merge {values} for key {k}")
    return merged


def update_config(config, extra_config):
    for k, v in extra_config.items():
        if isinstance(v, dict):
            config[k] = update_config(config[k], v)
        else:
            config[k] = v
    return config


def nested_dict():
    return defaultdict(nested_dict)
