import numpy as np
from sys import argv
from spectacle import raw, plot, io, wavelength
from datetime import datetime, timedelta

folder = io.path_from_input(argv)
root, images, stacks, products, results = io.folders(folder)
phone = io.read_json(root/"info.json")
bias = phone["software"]["bias"]

files = sorted(folder.glob("*.dng"))

trios = np.load(folder/"TriOS_calibration_interpolated.npy")
coefficients = wavelength.load_coefficients(results/"ispex/wavelength_solution.npy")

wavelengths = np.arange(390, 701, 1)

timestamps = []
Rs = []
Gs = []
Bs = []

for file in files:
    print(file)
    img  = io.load_raw_file(file)
    exif = io.load_exif(file)
    time = datetime.strptime(exif["EXIF DateTimeOriginal"].values, "%Y:%m:%d %H:%M:%S")
    time = (time - timedelta(hours=1)).timestamp()

    values = img.raw_image.astype(np.float32) - bias

    image_cut  = values        [760:1000, 2150:3900]
    colors_cut = img.raw_colors[760:1000, 2150:3900]
    x = np.arange(2150, 3900)
    y = np.arange(760 , 1000)

    RGBG, offsets = raw.pull_apart(image_cut, colors_cut)

    wavelengths_cut = wavelength.calculate_wavelengths(coefficients, x, y)
    wavelengths_split, offsets = raw.pull_apart(wavelengths_cut, colors_cut)

    lambdarange, all_interpolated = wavelength.interpolate_multi(wavelengths_split, RGBG)
    stacked = wavelength.stack(lambdarange, all_interpolated)
    plot.plot_spectrum(stacked[0], stacked[1:], title=time)

    timestamps.append(time)
    Rs.append(stacked[1])
    Gs.append(stacked[2])
    Bs.append(stacked[3])

timestamps = np.array(timestamps)
Rs = np.array(Rs)
Gs = np.array(Gs)
Bs = np.array(Bs)

remove_rows = []
for j, time in enumerate(timestamps):
    trios_ind = np.abs(trios[:,0] - timestamps[j]).argmin()
    time_diff = np.abs(trios[:,0][trios_ind] - timestamps[j])
    if time_diff > 10:
        remove_rows.append(j)
        continue

    trios_spectrum = trios[:,8:][trios_ind]
    Rs[j] = Rs[j]/trios_spectrum
    Gs[j] = Gs[j]/trios_spectrum
    Bs[j] = Bs[j]/trios_spectrum

timestamps = np.delete(timestamps, remove_rows)
Rs = np.delete(Rs, remove_rows, axis=0)
Gs = np.delete(Gs, remove_rows, axis=0)
Bs = np.delete(Bs, remove_rows, axis=0)

R = Rs.mean(axis=0)
G = Gs.mean(axis=0)
B = Bs.mean(axis=0)

Rerr = Rs.std(axis=0)
Gerr = Gs.std(axis=0)
Berr = Bs.std(axis=0)

RGB = np.stack([R,G,B])
plot.plot_spectrum(wavelengths, RGB, title="Averaged")

RGB -= RGB.min(axis=1)[:, np.newaxis]
normalization = RGB.max(axis=1)
RGB /= normalization[:, np.newaxis]
plot.plot_spectrum(wavelengths, RGB, title="Normalized")
R, G, B = RGB
Rerr /= normalization[0]
Gerr /= normalization[1]
Berr /= normalization[2]

# combine G and G2
spectral_response = np.stack([wavelengths, R, G, B, G, Rerr, Gerr, Berr, Gerr])

np.save(results/"ispex/spectral_response_ispex.npy", spectral_response)
np.save(results/"spectral_response/ispex_curve.npy", spectral_response)
print("Saved spectral response curves")
