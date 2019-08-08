import numpy as np
from sys import argv
from spectacle import io, linearity as lin

folders = io.path_from_input(argv)
roots = [io.folders(folder)[0] for folder in folders]

r_raw_paths  = [root/"products/linearity_pearson_r.npy" for root in roots]
r_jpeg_paths = [root/"products/linearity_pearson_r_jpeg.npy" for root in roots]

cameras = [io.read_json(root/"info.json")["device"]["name"] for root in roots]

def load_jpeg(path):
    try:
        jpeg = np.load(path)
    except FileNotFoundError:
        jpeg = None
    return jpeg

r_raw  = [  np.load(raw_path ) for raw_path  in r_raw_paths ]
r_jpeg = [load_jpeg(jpeg_path) for jpeg_path in r_jpeg_paths]

print("0.1% -- 99.9% range")

print(f"       Camera: |      RAW       |       J_R      |      J_G       |      J_B       | RAW < {lin.linearity_limit}")
for camera, raw_, jpeg_ in zip(cameras, r_raw, r_jpeg):
    print(f"{camera:>13}:", end="   ")
    raw = raw_[~np.isnan(raw_)].ravel()
    low, high = lin.percentile_r(raw)
    print(f"{low:.3f} -- {high:.3f}", end="   ")
    if jpeg_ is None:
        print("      --               --               --         ", end="")
    else:
        for j_ in jpeg_:
            jpeg = j_[~np.isnan(j_)].ravel()
            low, high = lin.percentile_r(jpeg)
            print(f"{low:.3f} -- {high:.3f}", end="   ")
    print(f"{np.where(raw < lin.linearity_limit)[0].shape[0]:>8}")
