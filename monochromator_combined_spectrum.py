import numpy as np
from sys import argv
from phonecal import io, raw, plot
from matplotlib import pyplot as plt

folder, wvl1, wvl2 = io.path_from_input(argv)
wvl1 = float(wvl1.stem) ; wvl2 = float(wvl2.stem)
root, images, stacks, products, results = io.folders(folder)
phone = io.read_json(root/"info.json")

colours = io.load_colour(stacks)

def load_cal_NERC(filename, norm=True):
    data = np.genfromtxt(filename, skip_header=1, skip_footer=10)
    if norm:
        data = data / data.max()  # normalise to 1
    with open(filename, "r") as file:
        info = file.readlines()[0].split(",")
    start, stop, step = [float(i) for i in info[3:6]]
    wavelengths = np.arange(start, stop+step, step)
    arr = np.stack([wavelengths, data])
    return arr

folders = sorted(folder.glob("*"))

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
    means -= phone["software"]["bias"]
    spectrum = np.stack([wvls, *means.T, *stds.T]).T
    return spectrum

spectra = [load_spectrum(subfolder) for subfolder in folders]
cal_files = [sorted(subfolder.glob("*.cal"))[0] for subfolder in folders]
cals = [load_cal_NERC(file) for file in cal_files]

all_wvl = np.unique(np.concatenate([spec[:,0] for spec in spectra]))
all_means = np.tile(np.nan, (len(spectra), len(all_wvl), 4))
all_stds = all_means.copy()
norms = np.zeros((len(spectra), 4)) ; norms[0] = 1

for i, spec in enumerate(spectra):
    min_wvl, max_wvl = spec[:,0].min(), spec[:,0].max()
    min_in_all = np.where(all_wvl == min_wvl)[0][0]
    max_in_all = np.where(all_wvl == max_wvl)[0][0]
    all_means[i][min_in_all:max_in_all+1] = spec[:,1:5]
    all_stds[i][min_in_all:max_in_all+1] = spec[:,5:]

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
plt.title(f"{phone['device']['name']}: Raw spectral curves")
plt.savefig(results/"spectral_response/raw_spectra.pdf")
plt.show()
plt.close()

all_calibrated = all_means.copy()
all_calibrated[:] = np.nan

for i, (mean, std, cal) in enumerate(zip(all_means, all_stds, cals)):
    calibrated = mean.copy() ; calibrated[:] = np.nan
    overlap, cal_indices, all_wvl_indices = np.intersect1d(cal[0], all_wvl, return_indices=True)
    calibrated[all_wvl_indices] = mean[all_wvl_indices] / cal[1, cal_indices, np.newaxis]
    all_calibrated[i] = calibrated

plt.figure(figsize=(10,5))
for mean, std in zip(all_calibrated, all_stds):
    for j, c in enumerate("rgby"):
        plt.plot(all_wvl, mean[:,j], c=c)
        plt.fill_between(all_wvl, mean[:,j]-std[:,j], mean[:,j]+std[:,j], color=c, alpha=0.3)
    plt.xticks(np.arange(0,1000,50))
    plt.xlim(wvl1,wvl2)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Spectral response (ADU)")
    plt.ylim(ymin=0)
plt.title(f"{phone['device']['name']}: Calibrated spectral curves")
plt.savefig(results/"spectral_response/calibrated_spectra.pdf")
plt.show()
plt.close()

for i, spec in enumerate(spectra):
    if i >= 1:
        max_this = np.nanargmax(all_calibrated[i], axis=0)
        for j in range(4):
            comparison = np.nanargmax(all_calibrated[:i, max_this[j], j])
            norms[i,j] = all_calibrated[i,max_this[j], j] / all_calibrated[comparison, max_this[j], j] * norms[comparison,j]

all_means_normalised = all_calibrated / norms[:, np.newaxis]

plt.figure(figsize=(10,5))
for mean, std, norm in zip(all_means_normalised, all_stds, norms):
    for j, c in enumerate("rgby"):
        plt.plot(all_wvl, mean[:,j], c=c)
        plt.fill_between(all_wvl, mean[:,j]-std[:,j], mean[:,j]+std[:,j], color=c, alpha=0.3)
    plt.xticks(np.arange(0,1000,50))
    plt.xlim(wvl1,wvl2)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Spectral response (ADU)")
    plt.ylim(ymin=0)
plt.title(f"{phone['device']['name']}: Normalised spectral curves")
plt.savefig(results/"spectral_response/normalised_spectra.pdf")
plt.show()
plt.close()

flat_means = np.nanmean(all_means_normalised, axis=0)
flat_stds  = np.nanstd(all_means_normalised, axis=0)

plt.figure(figsize=(10,5))
for j, c in enumerate("rgby"):
    plt.plot(all_wvl, flat_means[:,j], c=c)
    plt.fill_between(all_wvl, flat_means[:,j]-flat_stds[:,j], flat_means[:,j]+flat_stds[:,j], color=c, alpha=0.3)
plt.xticks(np.arange(0,1000,50))
plt.xlim(wvl1,wvl2)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Spectral response (ADU)")
plt.ylim(ymin=0)
plt.title(f"{phone['device']['name']}: Combined spectral curves")
plt.savefig(results/"spectral_response/combined_spectra.pdf")
plt.show()
plt.close()
