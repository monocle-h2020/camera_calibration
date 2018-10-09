import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from phonecal import raw, plot, io
from phonecal.general import gaussMd

folder = argv[1]
isos, means = io.load_means (folder, retrieve_value=io.split_iso, file=True)
colours     = io.load_colour(folder)

low_iso = isos.argmin()
high_iso= isos.argmax()

saveto = folder.replace("stacks", "products").strip("/")
np.save(f"{saveto}.npy", means[low_iso])

for ind in (low_iso, high_iso):
    iso  = isos [ind]
    mean = means[ind]

    plt.figure(figsize=(10,7), tight_layout=True)
    plt.hist(mean.ravel(), bins=np.linspace(513, 543, 250), color='k')
    plt.xlabel("Mean bias (ADU)")
    plt.xlim(513, 543)
    plt.yscale("log")
    plt.ylabel("Frequency")
    plt.ylim(ymin=0.9)
    plt.grid(ls="--")
    plt.savefig(f"results/bias/Bias_mean_hist_iso{iso}.png")
    plt.close()

    plot.imshow_gauss(mean, sigma=10, colorbar_label="Mean bias (ADU)", saveto=f"results/bias/Bias_mean_gauss_iso{iso}.png")

    mean_RGBG, _ = raw.pull_apart(mean, colours)
    for j, c in enumerate("RGBG"):
        X = "2" if j == 3 else ""
        plot.imshow_gauss(mean_RGBG[j], sigma=5, colorbar_label="Mean bias (ADU)", saveto=f"results/bias/Bias_mean_gauss_iso{iso}_{c}{X}.png", colour=c)