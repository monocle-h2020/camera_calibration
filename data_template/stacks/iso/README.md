This folder will contain stacked images (NPY).

ISO values are based on app values, not on camera values.

The format should be as follows:
.../stacks/uso/iso<ISO>.npy
where <ISO> is the ISO value.

Two types of stacked images will be present:
 - ..._mean.npy: mean per pixel of a data set
 - ..._stds.npy: standard deviation per pixel of a data set
