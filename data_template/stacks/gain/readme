This folder will contain stacked *gain* images (NPY) directly from the camera.

There will be subfolders for different ISO values. Each ISO subfolder contains stacked images for different lighting conditions.

ISO values are based on app values, not on camera values.

The format should be as follows:
.../images/gain/iso<ISO>/<name>_<type>.npy
where <ISO> is the ISO value, <name> can be any string identifying a dataset (e.g. lighting conditions, exposure time, timestamp, ...) and <type> the image type (see below).

This folder may contain a softlink to the '../linearity' folder containing similar data for a certain ISO value. This softlink will follow the naming format described above.

Two types of stacked images will be present:
 - ..._mean.npy: mean per pixel of a data set
 - ..._stds.npy: standard deviation per pixel of a data set
