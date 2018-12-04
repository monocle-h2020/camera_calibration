import numpy as np
from sys import argv
from phonecal import io, raw, plot
from matplotlib import pyplot as plt

folder = io.path_from_input(argv)
root, images, stacks, products, results = io.folders(folder)
phone = io.read_json(root/"info.json")

colours = io.load_colour(stacks)

wvls = np.arange(390, 682, 2)
means = np.zeros((len(wvls), 4))
stds = means.copy()

for j, wvl in enumerate(wvls):
    m = np.load(folder/f"{wvl}_mean.npy")
    mean_RGBG, _ = raw.pull_apart(m, colours)
    sub = mean_RGBG[:,756-50:756+51,1008-50:1008+51]
    means[j] = sub.mean(axis=(1,2))
    stds[j] = sub.std(axis=(1,2))
    print(wvl)

SNR = means/stds

plt.figure(figsize=(10,5))
for j, c in enumerate("rgb"):
    plt.plot(wvls, means[:,j], c=c)
    plt.fill_between(wvls, means[:,j]-stds[:,j], means[:,j]+stds[:,j], color=c, alpha=0.3)
plt.xticks(np.arange(350,850,50))
plt.xlim(340,760)
plt.ylim(ymin=0)
plt.show()
plt.close()
