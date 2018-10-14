from sys import argv
from phonecal import io

path = io.path_from_input(argv)

dump = io.read_json(path)

