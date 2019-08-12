import numpy as np
from sys import argv
from spectacle import io, plot, linearity as lin
from spectacle.general import RMS
from matplotlib import pyplot as plt

folders = io.path_from_input(argv)
roots = [io.folders(f)[0] for f in folders]
phones = [io.load_metadata(root) for root in roots]

intensities_es = []
intensities_er = []
dng_means      = []
jpeg_means     = []

for folder, phone in zip(folders, phones):
    print(phone["device"]["name"])
    root, images, stacks, products, results = io.folders(folder)
    colours = io.load_colour(stacks)

    array_size = np.array(colours.shape)
    mid1, mid2 = array_size // 2
    center = np.s_[mid1:mid1+2, mid2:mid2+2]

    colours_here = colours[center].ravel()
    colours_sort = np.argsort(colours_here)

    angles, means = io.load_means (folder, retrieve_value=io.split_pol_angle, selection=center)
    means = means.reshape((len(means), -1))
    means = means[:, colours_sort]
    print("Loaded DNG data")

    angles, jmeans = io.load_jmeans(folder, retrieve_value=io.split_pol_angle, selection=center)
    jmeans = jmeans.reshape((len(jmeans), -1, 3))
    jmeans = jmeans[:, colours_sort]
    print("Loaded JPEG data")

    offset_angle = io.load_angle(stacks)
    print("Read angles")
    intensities = lin.malus(angles, offset_angle)
    intensities_errors = lin.malus_error(angles, offset_angle, sigma_angle0=1, sigma_angle1=1)

    intensities_es.append(intensities)
    intensities_er.append(intensities_errors)
    dng_means.append(means)
    jpeg_means.append(jmeans)

for j, c in enumerate("rgbg"):
    fig, axs = plt.subplots(ncols=len(folders), nrows=2, figsize=(3.3*len(folders), 3.5), tight_layout=True, sharex=True, gridspec_kw={"hspace":0.1, "wspace":0.8})
    colour_index = colours_here[j]
    colour = "rgbg"[colour_index]
    if colour_index < 3:
        i = colour_index
        label = colour
    else:
        i = 1
        label = "g2"
    for phone, intensities, intensities_errors, means, jmeans, ax_col in zip(phones, intensities_es, intensities_er, dng_means, jpeg_means, axs.T):
        M =  means[:, j]
        J = jmeans[:, j, i]

        max_value = 2**phone["camera"]["bits"]

        r_dng  = lin.pearson_r_single(intensities, M, saturate=max_value*0.95)
        r_jpeg = lin.pearson_r_single(intensities, J, saturate=240)
        r_dng_str = "$r_{DNG}"
        r_jpg_str = "$r_{JPEG}"

        x = np.linspace(0, 1, 250)
        fit_linear   = np.polyfit(intensities[M < max_value*0.95], M[M < max_value*0.95], 1)
        fit_sRGB, pc = lin.curve_fit(lin.sRGB_generic, intensities[J < 240], J[J < 240], p0=[1, 2.2])
        y_linear = np.clip(np.polyval(fit_linear, x), 0, max_value)
        y_sRGB   = lin.sRGB_generic(x, *fit_sRGB)

        ax = ax_col[0]
        ax.errorbar(intensities, J, xerr=intensities_errors, fmt=f"{colour}o", ms=3)
        ax.plot(x, y_sRGB, c=colour)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(0, 255*1.02)
        ax.set_xticks(np.arange(0,1.2,0.2))
        ax.set_yticks(np.arange(0, 255, 50))
        ax2 = ax.twinx()
        ax2.errorbar(intensities, M, xerr=intensities_errors, fmt="ko", ms=3)
        ax2.plot(x, y_linear, c='k')
        ax2.set_ylim(0, max_value*1.02)
        ax2.locator_params(axis="y", nbins=5)
        ax.grid(True, axis="x")
        ax2.grid(True, axis="y")
        jpeglabel = ax.set_ylabel("JPEG value")
        jpeglabel.set_color(colour)
        ax.tick_params(axis="y", colors=colour)
        ax.tick_params(axis="x", bottom=False)
        ax2.set_ylabel("RAW value")
        ax.set_title(f"{phone['device']['name']}\n{r_jpg_str} = {r_jpeg:.3f}$ {r_dng_str} = {r_dng:.3f}$")

        J_res = 100*(J - lin.sRGB_generic(intensities, *fit_sRGB))/255
        ax_res = ax_col[1]
        ax_res.errorbar(intensities, J_res, xerr=intensities_errors, fmt=f"{colour}o", ms=3)
        ax_res.locator_params(axis="y", nbins=5)
        jpeglabel_res = ax_res.set_ylabel("Norm. res.\n(JPEG, %)")
        jpeglabel_res.set_color(colour)
        ax_res2 = ax_res.twinx()
        M_res = 100*(M - np.clip(np.polyval(fit_linear, intensities), 0, max_value))/max_value
        ax_res2.errorbar(intensities, M_res, xerr=intensities_errors, fmt="ko", ms=3)
        ax_res2.set_ylabel(f"Norm. res.\n(RAW, %)")
        ax_res2.locator_params(axis="y", nbins=5)
        ax_res.grid(True, axis="x")
        ax_res.grid(True, axis="y")
        ax_res.set_xlabel("Relative incident intensity")
        ax_res.tick_params(axis="y", colors=colour)
        ylim = [min([J_res.min(), M_res.min()])-0.3, max([J_res.max(), M_res.max()])+0.3]
        ax_res.set_ylim(*ylim)
        ax_res2.set_ylim(*ylim)

        print(f"RMS residual (RAW):  {RMS(M_res[M < max_value*0.95]):.1f}%")
        print(f"RMS residual (JPEG): {RMS(J_res[J < 240]):.1f}%")
    plt.savefig(io.results_folder/f"linearity_DNG_JPEG_{label}.pdf")
    plt.show()
    plt.close()
    print(f"Plotted pixel {j} ({label})")