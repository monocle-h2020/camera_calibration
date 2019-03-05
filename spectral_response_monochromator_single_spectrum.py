import numpy as np
from sys import argv
from phonecal import io, raw, plot
from matplotlib import pyplot as plt

folder, wvl1, wvl2 = io.path_from_input(argv)
wvl1 = float(wvl1.stem) ; wvl2 = float(wvl2.stem)
root, images, stacks, products, results = io.folders(folder)
phone = io.read_json(root/"info.json")

colours = io.load_colour(stacks)

mean_files = sorted(folder.glob("*_mean.npy"))
stds_files = sorted(folder.glob("*_stds.npy"))
assert len(mean_files) == len(stds_files)

wvls  = np.zeros((len(mean_files)))
means = np.zeros((len(mean_files), 4))
stds  = means.copy()

for j, (mean_file, stds_file) in enumerate(zip(mean_files, stds_files)):
    m = np.load(mean_file)
    mean_RGBG, _ = raw.pull_apart(m, colours)
    sub = mean_RGBG[:,756-50:756+51,1008-50:1008+51]
    wvls[j] = mean_file.stem.split("_")[0]
    means[j] = sub.mean(axis=(1,2))
    stds[j] = sub.std(axis=(1,2))
    print(wvls[j])

means -= phone["software"]["bias"]

SNR = means/stds

plt.figure(figsize=(10,5))
for j, c in enumerate("rgby"):
    plt.plot(wvls, means[:,j], c=c)
    plt.fill_between(wvls, means[:,j]-stds[:,j], means[:,j]+stds[:,j], color=c, alpha=0.3)
plt.xticks(np.arange(0,1000,50))
plt.xlim(wvl1,wvl2)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Spectral response (ADU)")
plt.ylim(ymin=0)
plt.savefig(results/f"spectral_response/{folder.stem}.pdf")
plt.show()
plt.close()
