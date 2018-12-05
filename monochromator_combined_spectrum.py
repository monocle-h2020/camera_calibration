import numpy as np
from sys import argv
from phonecal import io, raw, plot
from matplotlib import pyplot as plt

folder, wvl1, wvl2 = io.path_from_input(argv)
wvl1 = float(wvl1.stem) ; wvl2 = float(wvl2.stem)
root, images, stacks, products, results = io.folders(folder)
phone = io.read_json(root/"info.json")

colours = io.load_colour(stacks)

folders = folder.glob("*")

def load_spectrum(subfolder):
    mean_files = sorted(subfolder.glob("*_mean.npy"))
    stds_files = sorted(subfolder.glob("*_stds.npy"))
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
    print(subfolder)
    spectrum = np.stack([wvls, *means.T, *stds.T]).T
    return spectrum

spectra = [load_spectrum(subfolder) for subfolder in folders]

all_wvl = np.unique(np.concatenate([spec[:,0] for spec in spectra]))
all_means = np.tile(np.nan, (len(spectra), len(all_wvl), 4))
all_stds = all_means.copy()

for j, spec in enumerate(spectra):
    min_wvl, max_wvl = spec[:,0].min(), spec[:,0].max()
    min_in_all = np.where(all_wvl == min_wvl)[0][0]
    max_in_all = np.where(all_wvl == max_wvl)[0][0]
    all_means[j][min_in_all:max_in_all+1] = spec[:,1:5]
    all_stds[j][min_in_all:max_in_all+1] = spec[:,5:]

plt.figure(figsize=(10,5))
for mean, std in zip(all_means, all_stds):
    for j, c in enumerate("rgby"):
        plt.plot(all_wvl, mean[:,j], c=c)
        plt.fill_between(all_wvl, mean[:,j]-std[:,j], mean[:,j]+std[:,j], color=c, alpha=0.3)
    plt.xticks(np.arange(0,1000,50))
    plt.xlim(wvl1,wvl2)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Spectral response (ADU)")
    plt.ylim(ymin=0)
plt.show()
plt.close()
