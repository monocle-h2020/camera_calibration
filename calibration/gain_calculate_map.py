import numpy as np
from sys import argv
from spectacle import io, calibrate

folder = io.path_from_input(argv)
root, images, stacks, products, results = io.folders(folder)
phone = io.read_json(root/"info.json")

products_gain, results_gain = products/"gain", results/"gain"
ISO = io.split_iso(folder)
print("Loaded information")

names, means = io.load_means (folder  )
names, stds  = io.load_stds  (folder  )
colours      = io.load_colour(stacks  )
print("Loaded data")

means = calibrate.correct_bias(root, means)

variance = stds**2

fit_max = 0.95 * 2**phone["camera"]["bits"]

gains = np.tile(np.nan, means.shape[1:])
rons  = gains.copy()

for i in range(means.shape[1]):
    for j in range(means.shape[2]):
        m = means[:,i,j] ; v = variance[:,i,j]
        ind = np.where(m < fit_max)
        try:
            gains[i,j], rons[i,j] = np.polyfit(m[ind], v[ind], 1, w=1/m[ind])
        except:
            pass

    if i%15:
        print(f"{100 * i / means.shape[1]:.1f}%", end=" ", flush=True)

np.save(products_gain/f"iso{ISO}.npy", gains)
