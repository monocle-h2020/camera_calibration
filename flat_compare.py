import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from phonecal import raw, io, plot, flat
from phonecal.general import gaussMd, Rsquare, RMS

flat_files = io.path_from_input(argv)
root, images, stacks, products, results = io.folders(flat_files[0])
phone = io.read_json(root/"info.json")

colors = io.load_colour(stacks)
shape = colors.shape
flat_params = np.array([np.load(file) for file in flat_files])
flat1, flat2 = [flat.apply_vignette_radial(shape, params[0]) for params in flat_params]
flat1_name, flat2_name = [file.stem for file in flat_files]

diff = flat1 - flat2

flat_max = max([flat1.max(), flat2.max()])
diff_max = max([abs(diff.max()), abs(diff.min())])
vmins = [1, 1, -diff_max]
vmaxs = [flat_max, flat_max, diff_max]
clabels = [f"$g$ (Flat 1)", f"$g$ (Flat 2)", "Difference"]
fig, axs = plt.subplots(ncols=3, figsize=(6,2), sharex=True, sharey=True, squeeze=True, tight_layout=True, gridspec_kw={"wspace":0, "hspace":0})
for data, ax, vmin, vmax, clabel in zip([flat1, flat2, diff], axs, vmins, vmaxs, clabels):
    img = ax.imshow(data, vmin=vmin, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    colorbar_here = plot.colorbar(img)
    colorbar_here.set_label(clabel)
    colorbar_here.locator = plot.ticker.MaxNLocator(nbins=4)
    colorbar_here.update_ticks()
#fig.savefig(results/f"flat/{label}_correction_combined.pdf")
plt.show()
plt.close()

plt.hist(diff.ravel(), bins=250)
plt.xlabel(f"{flat1_name} - {flat2_name}")
plt.show()

rms = RMS(diff)
rms_rel = RMS(diff/flat1)
print(f"Difference: RMS {rms:.2f} or {100*rms_rel:.1f}%")

for (param1, err1, param2, err2), label in zip(flat_params.reshape((4, 7)).T, ["k0", "k1", "k2", "k3", "k4", "cx", "cy"]):
    diff = param1 - param2
    err_diff = np.sqrt(err1**2 + err2**2)
    print(f"delta {label:>3} = {diff:+.4f} +- {err_diff:.4f} ({abs(diff/err_diff):>5.0f} sigma)")
