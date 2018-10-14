import json
from sys import argv
from phonecal import io

path = io.path_from_input(argv)

def read_json(path):
    file = open(path)
    dump = json.load(file)
    return dump

read_json(path)