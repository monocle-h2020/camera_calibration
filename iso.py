import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from phonecal import raw, io, plot
from phonecal.general import gaussMd, Rsquare
from scipy.optimize import curve_fit

folder = io.path_from_input(argv)
root, images, stacks, products, results = io.folders(folder)
phone = io.read_json(root/"info.json")
min_iso = phone["software"]["ISO min"]
max_iso = phone["software"]["ISO max"]

products_iso, results_iso = products/"iso", results/"iso"
print("Loaded information")

colours      = io.load_colour(stacks  )
bias         = io.load_bias  (products)
print("Loaded metadata")

isos, means = io.load_means (folder, retrieve_value=io.split_iso)
isos, stds  = io.load_stds  (folder, retrieve_value=io.split_iso)
print("Loaded data")

means -= bias

relative_errors = stds / means
median_relative_error = np.median(relative_errors)
print(f"Median relative error in photometry: {median_relative_error*100:.1f} %")

assert isos.min() == min_iso

ratios = means / means[isos.argmin()]
ratios_mean = ratios.mean(axis=(1,2))
ratios_errs = ratios.std (axis=(1,2))

ratios_RGBG,_ = raw.pull_apart(ratios, colours)

def generate_linear_model(slope, offset):
    model = lambda isos: np.polyval([slope, offset], isos)
    return model

def knee_model(isos, slope, offset, knee):
    linear_model = generate_linear_model(slope, offset)
    values = np.clip(linear_model(isos), a_min=None, a_max=linear_model(knee))
    return values

def generate_knee_model(slope, offset, knee):
    model = lambda isos: knee_model(isos, slope, offset, knee)
    return model

p_knee  , cov_knee   = curve_fit(knee_model, isos, ratios_mean, p0=[1/min_iso, 0, 200], bounds=([0, -np.inf, 1.05*min_iso], [1, np.inf, 0.95*max_iso]))

p_linear, cov_linear = np.polyfit(isos, ratios_mean, 1, cov=True)

model_linear = generate_linear_model(*p_linear)
model_knee   = generate_knee_model  (*p_knee  )

R2_linear = Rsquare(ratios_mean, model_linear(isos))
R2_knee   = Rsquare(ratios_mean, model_knee  (isos))

if R2_linear < 0.8 and R2_knee < 0.8:
    model = None
    R2 = None
    print("No adequate model found")
else:
    if R2_linear > 0.8:
        model = model_linear
        R2 = R2_linear
        print(f"Using linear model [y = ax + b] with a = {p_linear[0]:.3f} & b = {p_linear[1]:.3f}")
    elif R2_knee > 0.8:
        model = model_knee
        R2 = R2_knee
        print(f"Using knee model with with a = {p_knee[0]:.3f} & b = {p_knee[1]:.3f} & k = {p_knee[2]:.1f}")
    print(f"R^2 = {R2:.6f}")

iso_range = np.arange(0, max_iso+1, 1)
plt.figure(figsize=(3.3,3), tight_layout=True)
plt.errorbar(isos, ratios_mean, yerr=ratios_errs, fmt="ko", label="Data")
plt.plot(iso_range, model(iso_range), c='k', label=f"Fit")
plt.title(f"$R^2 = {R2:.6f}$")
plt.ylim(ymin=0)
plt.xlim(0, 1.01*phone["software"]["ISO max"])
plt.xlabel("ISO speed")
plt.ylabel("Normalization")
plt.legend(loc="lower right")
plt.savefig(results_iso/"normalization.pdf")
plt.show()
plt.close()

np.save(products_iso/"lookup_table.npy", np.stack([iso_range, model(iso_range)   ]))
np.save(products_iso/"data.npy"        , np.stack([isos, ratios_mean, ratios_errs]))
