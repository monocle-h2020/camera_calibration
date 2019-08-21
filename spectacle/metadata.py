import numpy as np
import json

def load_json(path):
    """
    Read a JSON file.
    """
    file = open(path)
    dump = json.load(file)
    return dump


def write_json(data, save_to):
    """
    Write a JSON file containing `data` to a path `save_to`.
    """
    with open(save_to, "w") as file:
        json.dump(data, file)


def load_metadata(root):
    """
    Read the metadata JSON located in the `root` folder.
    """
    metadata = load_json(root/"info.json")
    return metadata
