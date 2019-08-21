import numpy as np
import json

def load_json(path):
    """
    Read a JSON file.
    """
    file = open(path)
    dump = json.load(file)
    return dump


def load_metadata(root):
    """
    Read the metadata JSON located in the `root` folder.
    """
    metadata = load_json(root/"info.json")
    return metadata
