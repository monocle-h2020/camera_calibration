"""
Plot the camera response at various incoming intensities for the central pixels
in a stack of images for multiple cameras. This script assumes JPEG data are
available for each camera.

The RAW and JPEG data should be taken at the same intensities so a 1-to-1
comparison can be made.

Command line arguments:
    * `folder`: the folder containing linearity data stacks. These should be
    NPY stacks taken at different exposure conditions, with the same ISO speed.
    (multiple arguments possible)
"""


import numpy as np
from sys import argv
from spectacle import io, plot, linearity as lin
from spectacle.general import RMS
from matplotlib import pyplot as plt

# Get the data folder from the command line
folders = io.path_from_input(argv)
roots = [io.find_root_folder(f) for f in folders]

# Load Camera objects
cameras = [io.load_camera(root) for root in roots]
print(f"Loaded Camera objects: {cameras}")

save_to = io.results_folder

# Lists to hold the data for each device
intensities_all = []
intensities_error_all = []
mean_raw_all = []
mean_jpeg_all = []

# Loop over the given folders
for folder, camera in zip(folders, cameras):
    # Load Camera object
    print("\n", camera)
    root = io.find_root_folder(folder)

    # Find the indices of the central pixels
    array_size = np.array(camera.image.shape)
    mid1, mid2 = array_size // 2
    center = np.s_[mid1:mid1+2, mid2:mid2+2]

    # Bayer channels of the central pixels
    # These will be used to pre-sort the data into the RGBG2 channels for every
    # camera
    colours_here = camera.bayer_map[center].ravel()
    colours_sort = np.argsort(colours_here)

    # Load the RAW data
    intensities_with_errors, means = io.load_means(folder, retrieve_value=lin.filename_to_intensity, selection=center)
    intensities, intensity_errors = intensities_with_errors.T
    means = means.reshape((len(means), -1))
    means = means[:, colours_sort]
    print("Loaded RAW data")

    # Load the JPEG data
    intensities_with_errors, jmeans = io.load_jmeans(folder, retrieve_value=lin.filename_to_intensity, selection=center)
    intensities, intensity_errors = intensities_with_errors.T
    jmeans = jmeans.reshape((len(jmeans), -1, 3))
    jmeans = jmeans[:, colours_sort]
    print("Loaded JPEG data")

    # Select only the appropriate JPEG channel (R-R, G-G, B-B, G2-G)
    jmeans = np.array([jmeans[:,0,0], jmeans[:,1,1], jmeans[:,2,2], jmeans[:,3,1]]).T

    # Add the data to the lists
    intensities_all.append(intensities)
    intensities_error_all.append(intensity_errors)
    mean_raw_all.append(means)
    mean_jpeg_all.append(jmeans)

# Loop over the Bayer RGBG2 channels and plot the response in each
for j, c in enumerate(plot.rgbg2):
    # Create a figure to hold the scatter plots of response (top row) and
    # residuals (bottom row) for each camera (columns)
    fig, axs = plt.subplots(ncols=len(folders), nrows=2, figsize=(3.3*len(folders), 3.5), tight_layout=True, sharex=True, gridspec_kw={"hspace":0.1, "wspace":0.8})

    # Loop over the cameras and their associated data
    for camera, ax_column, intensities, intensity_errors, means_raw, means_jpeg in zip(cameras, axs.T, intensities_all, intensities_error_all, mean_raw_all, mean_jpeg_all):
        # Select the mean data from the relevant channel only
        # This is very clunky (double for-loop and still using an index anyway)
        # It might be better to change the way data are loaded (e.g. per
        # channel instead of per camera) reverse the order of these for-loops
        mean_raw_c = means_raw[:, j]
        mean_jpeg_c = means_jpeg[:, j]

        # Calculate Pearson r values of the RAW and JPEG data
        # These are added to the plot titles
        r_raw = lin.pearson_r_single(intensities, mean_raw_c, saturate=0.95*camera.saturation)
        r_jpeg = lin.pearson_r_single(intensities, mean_jpeg_c, saturate=240)

        # Plot title, including device name and r values calculated above
        title = camera.device.name + "\n" + "$r_{JPEG} = " + f"{r_jpeg:.3f}" + "$   $r_{RAW} = " + f"{r_raw:.3f}" + "$"

        # Fit a linear function to the RAW response and an sRGB function to the
        # JPEG response, to plot as lines
        x = np.linspace(0, 1, 250)

        non_saturated_indices_raw = np.where(mean_raw_c < 0.95*camera.saturation)
        fit_raw = np.polyfit(intensities[non_saturated_indices_raw], mean_raw_c[non_saturated_indices_raw], 1)
        line_raw = np.clip(np.polyval(fit_raw, x), 0, camera.saturation)

        non_saturated_indices_jpeg = np.where(mean_jpeg_c < 240)
        fit_jpeg, pc = lin.curve_fit(lin.sRGB_generic, intensities[non_saturated_indices_jpeg], mean_jpeg_c[non_saturated_indices_jpeg], p0=[1, 2.2])
        line_jpeg = lin.sRGB_generic(x, *fit_jpeg)

        # Calculate the residuals of the RAW/JPEG response compared to the
        # linear/sRGB fit, and relative to the dynamic range
        y_raw = np.clip(np.polyval(fit_raw, intensities), 0, camera.saturation)
        residuals_raw = mean_raw_c - y_raw
        residuals_raw_percentage = 100 * (residuals_raw / camera.saturation)

        y_jpeg = lin.sRGB_generic(intensities, *fit_jpeg)
        residuals_jpeg = mean_jpeg_c - y_jpeg
        residuals_jpeg_percentage = 100 * (residuals_jpeg / 255)

        # Plot the JPEG response
        colour = c[0]  # "r" -> "r", "g" -> "g", "b" -> "b", "g2" -> "g"
        ax_jpeg = ax_column[0]
        ax_jpeg.errorbar(intensities, mean_jpeg_c, xerr=intensity_errors, fmt=f"{colour}o", ms=3)  # data
        ax_jpeg.plot(x, line_jpeg, c=colour)  # best-fitting model

        # JPEG plot parameters
        ax_jpeg.set_xlim(-0.02, 1.02)
        ax_jpeg.set_ylim(0, 260)
        ax_jpeg.set_xticks(np.arange(0, 1.2, 0.2))
        ax_jpeg.set_yticks(np.arange(0, 255, 50))
        ax_jpeg.grid(True, axis="x")
        label_jpeg = ax_jpeg.set_ylabel("JPEG value")
        label_jpeg.set_color(colour)
        ax_jpeg.tick_params(axis="y", colors=colour)
        ax_jpeg.tick_params(axis="x", bottom=False)
        ax_jpeg.set_title(title)

        # Plot the RAW response
        ax_raw = ax_jpeg.twinx()  # plot in the same window
        ax_raw.errorbar(intensities, mean_raw_c, xerr=intensity_errors, fmt=f"ko", ms=3)  # data
        ax_raw.plot(x, line_raw, c='k')  # best-fitting model

        # RAW plot parameters
        ax_raw.set_ylim(0, camera.saturation*1.02)
        ax_raw.locator_params(axis="y", nbins=5)  # automatic yticks
        ax_raw.grid(True, axis="y")
        ax_raw.set_ylabel("RAW value")

        # Plot the JPEG residuals
        ax_residual_jpeg = ax_column[1]
        ax_residual_jpeg.errorbar(intensities, residuals_jpeg_percentage, xerr=intensity_errors, fmt=f"{colour}o", ms=3)

        # JPEG residual plot parameters
        ax_residual_jpeg.locator_params(axis="y", nbins=5)
        label_jpeg_residual = ax_residual_jpeg.set_ylabel("Norm. res.\n(JPEG, %)")
        label_jpeg_residual.set_color(colour)
        ax_residual_jpeg.grid(True)
        ax_residual_jpeg.set_xlabel("Relative incident intensity")
        ax_residual_jpeg.tick_params(axis="y", colors=colour)

        # Plot the RAW residuals
        ax_residual_raw = ax_residual_jpeg.twinx()  # plot in the same window
        ax_residual_raw.errorbar(intensities, residuals_raw_percentage, xerr=intensity_errors, fmt=f"ko", ms=3)
        ax_residual_raw.locator_params(axis="y", nbins=5)
        ax_residual_raw.set_ylabel(f"Norm. res.\n(RAW, %)")

        # y limits of the residual, making sure all data fit
        ylim = [min([residuals_jpeg_percentage.min(), residuals_raw_percentage.min()])-0.3, max([residuals_jpeg_percentage.max(), residuals_raw_percentage.max()])+0.3]
        ax_residual_jpeg.set_ylim(*ylim)
        ax_residual_raw.set_ylim(*ylim)

        # Print the RMS residuals (extra information)
        print(f"RMS residual (RAW):  {RMS(residuals_raw_percentage[non_saturated_indices_raw]):.1f}%")
        print(f"RMS residual (JPEG): {RMS(residuals_jpeg_percentage[non_saturated_indices_jpeg]):.1f}%")

    # Save the figure for this channel
    save_to_c = save_to/f"linearity_response_multiple_{c}.pdf"
    plt.savefig(save_to_c)
    plt.close()
    print(f"Saved the {c} channel plot to '{save_to_c}'")
