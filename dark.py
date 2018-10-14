import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from ispex import raw, plot, io
from ispex.general import Rsquare
from glob import glob

iso = argv[1]
folder_main = f"test_files/dark/iso{iso}/"
folders_t = glob(folder_main+"/*")
ts = np.zeros_like(folders_t, dtype=np.float32)
Ms = ts.copy()
Merrs = Ms.copy()
Vs = ts.copy()
Verrs = Vs.copy()
mean_arrs = []
var_arrs = []

for i,folder in enumerate(folders_t):
    t = 1/float(folder.split("\\")[-1])
    files = glob(folder+"/*.dng")

    arrs, colors = io.load_dng_many(f"{folder}/*.dng", return_colors=True)

    mean = arrs.mean(axis=0).astype(np.float32)  # mean per x,y
    mean_arrs.append(mean)
    var_arrs.append(arrs.var(axis=0))

    means = arrs.mean(axis=(1,2))
    mean_all = means.mean()
    mean_err = means.std() / np.sqrt(len(means) - 1)

    new_arrs = arrs - mean

    stds = new_arrs.std(axis=(1,2))
    std_mean = stds.mean()
    var = std_mean**2
    var_err = stds.std() / np.sqrt(len(stds) - 1) * 2 * std_mean
    ts[i], Vs[i], Verrs[i], Ms[i], Merrs[i] = t, var, var_err, mean_all, mean_err

    print(f"{(i+1)/len(folders_t)*100:.0f}%", end=" ")

mean_arrs = np.stack(mean_arrs)
var_arrs  = np.stack(var_arrs)
p, cov = np.polyfit(ts, Ms, 1, w=1/Merrs, cov=True)

trange = np.linspace(0, 0.35, 100)
trange_fit = np.polyval(p, trange)
trange_fit_err = np.sqrt(trange**2 * cov[0,0] + cov[1,1])
ts_fit = np.polyval(p, ts)
R2 = Rsquare(Ms, ts_fit)

plt.figure(figsize=(6,4), tight_layout=True)
plt.errorbar(ts, Ms, yerr=Merrs, fmt="ko", label="Data")
plt.plot(trange, trange_fit, c="k", label=f"Bias = {p[1]:.3f}\nDark = {p[0]:.3f} ADU/s")
plt.fill_between(trange, trange_fit-trange_fit_err, trange_fit+trange_fit_err, color="0.5", zorder=0,
                 label=f"$\sigma_B$ = {np.sqrt(cov[1,1]):.3f}\n$\sigma_D$ = {np.sqrt(cov[0,0]):.3f}")
plt.xlabel("$t$ (s)")
plt.ylabel("Mean (ADU)")
plt.xlim(1e-3, np.max(ts)*1.05)
plt.ticklabel_format(useOffset=False)
plt.title(f"10-image dark current ; $R^2 = {R2:.3f}$")
plt.legend(loc="lower right")
plt.savefig(f"results/dark/10image_{iso}.png")
plt.close()

m = mean_arrs.reshape((mean_arrs.shape[0], -1))  # as list
P = np.polyfit(ts, m, 1)  # linear fit to every pixel

plt.figure(figsize=(10,6), tight_layout=True)
plt.hist(P[0], bins=np.linspace(-75,100,250), color='k', density=True)
plt.yscale("log")
plt.xlabel("Dark current (ADU/s)")
plt.ylabel("Density")
plt.title(f"Dark current at ISO {iso} ; mean {P[0].mean():.2f} +- {P[0].std():.2f} ADU/s")
plt.savefig(f"results/dark/hist_{iso}.png")
plt.close()

LUT = np.load("results/gain_new/LUT.npy")
gain = LUT[1, int(iso)]
Pe = gain * P

plt.figure(figsize=(10,6), tight_layout=True)
plt.hist(Pe[0], bins=np.linspace(-35,50,250), color='k', density=True)
plt.yscale("log")
plt.xlabel("Dark current $e^-/s$")
plt.ylabel("Density")
plt.xlim(-35,50)
plt.title(f"Dark current at ISO {iso} ; mean {Pe[0].mean():.2f} +- {Pe[0].std():.2f} $e^-/s$")
plt.savefig(f"results/dark/hist_e_{iso}.png")
plt.close()

asimage = Pe[0].reshape(mean_arrs.shape[1:])
plot.bitmap(asimage, saveto=f"results/dark/map_{iso}.png")