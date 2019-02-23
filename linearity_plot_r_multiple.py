import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from phonecal import io, linearity as lin

folders = io.path_from_input(argv)
roots = [io.folders(folder)[0] for folder in folders]

r_raw_paths  = [root/"products/linearity_pearson_r.npy" for root in roots]
r_jpeg_paths = [root/"products/linearity_pearson_r_jpeg.npy" for root in roots]

cameras = [io.read_json(root/"info.json")["device"]["name"] for root in roots]

r_raw  = [np.load(raw_path ) for raw_path  in r_raw_paths ]
r_jpeg = [np.load(jpeg_path) for jpeg_path in r_jpeg_paths]

lower_limit = 0.9

bins = np.linspace(lower_limit, 1.0, 1000)

fig, axs = plt.subplots(nrows=len(r_raw), sharex=True, sharey=True, squeeze=True, tight_layout=True, figsize=(4,1.3*len(r_raw)), gridspec_kw={"wspace":0, "hspace":0})
for ax, raw_, jpeg_, camera in zip(axs, r_raw, r_jpeg, cameras):
    raw = raw_[~np.isnan(raw_)]
    print(camera)
    below_limit = np.where(raw < lower_limit)[0]
    print(f"RAW pixels with r < {lower_limit}: {len(below_limit)}")
    ax.hist(raw.ravel(), bins=bins, color='k')
    for j_, c in zip(jpeg_, "rgb"):
        jpeg_c = j_[~np.isnan(j_)]
        below_limit = np.where(jpeg_c < lower_limit)[0]
        print(f"JPEG {c} pixels with r < {lower_limit}: {len(below_limit)}")
        ax.hist(jpeg_c.ravel(), bins=bins, color=c, alpha=0.7)
    ax.set_ylabel(camera)
    ax.axvline(lin.linearity_limit, c='k', ls="--")
    bad_pixels = np.where(raw < lin.linearity_limit)[0]
    print(f"RAW pixels with r < {lin.linearity_limit}: {len(bad_pixels)}")
axs[0] .set_xlim(lower_limit, 1)
axs[0] .set_yscale("log")
axs[0] .set_ylim(ymin=0.9)
axs[-1].set_xlabel("Pearson $r$")
fig.savefig("results/linearity.pdf")
plt.show()
plt.close()
