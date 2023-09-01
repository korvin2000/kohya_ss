from typing import List, Dict, Any
import csv
import glob
import toml
import os
import argparse

def flatten_toml(toml_dict):
    """
    Parse nested toml dict into a flat dict.
    Uses key.subkey.subsubkey notation for nested keys.
    For list values, enumerate and attach _i to key.
    """
    if not isinstance(toml_dict, dict):
        return toml_dict
    result = {}
    for key, value in toml_dict.items():
        if isinstance(value, dict):
            value = flatten_toml(value)
            for subkey, subvalue in value.items():
                result[key + "." + subkey] = flatten_toml(subvalue)
        elif isinstance(value, list):
            for i, item in enumerate(value):
                flattened_item = flatten_toml(item)
                if isinstance(flattened_item, dict):
                    for subkey, subvalue in flattened_item.items():
                        result[key + "_" + str(i) + "." + subkey] = flatten_toml(subvalue)
                else:
                    result[key + "_" + str(i)] = flatten_toml(item)
        else:
            result[key] = value
    return result

def gather_toml(path:str) -> List[Dict[str, Any]]:
    """
    Gather all toml files recursively under path.
    """
    keyset = set()
    for toml_path in glob.glob(path + "/**/*.toml", recursive=True):
        keyset.update(flatten_toml(toml.load(toml_path)).keys())
    return list(keyset)

def process_subfolder(path:str, writer:csv.DictWriter):
    """
    Process all toml files recursively under path.
    """
    for toml_path in glob.glob(path + "/**.toml"):
        writer.writerow(flatten_toml(toml.load(toml_path)))
    
def process_toml_path(path:str, csv_path:str):
    """
    Process all toml files recursively under path.
    """
    keyset = gather_toml(path)
    writer = csv.DictWriter(csv_path, fieldnames=keyset)
    for subfolders in glob.glob(path + "/*"):
        if os.path.isdir(subfolders):
            process_subfolder(subfolders, writer)
    writer.writeheader()
    return keyset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='.')
    args = parser.parse_args()
    # finally merge to csv
    process_toml_path(args.path, os.path.join(args.path, 'results.csv'))

