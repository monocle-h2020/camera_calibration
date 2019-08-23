This folder contains raw *monochromator* stacks (NPY).

There will be subfolders for different grating/filter combinations. These in turn contain image stacks, corresponding to different wavelengths.

The format should be as follows:
.../stacks/monochromator/<grating_filter>/<wavelength>_<type>.npy
where <grating_filter> is a combination of grating and filter, <wavelength> is a wavelength in nm, and <type> is the stack type (mean/stds).
