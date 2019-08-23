This folder contains stacked *linearity* images (NPY) directly from the camera.

There will be files for different lighting conditions. 
When using the linear polariser method, subfolders should be identified according to polariser angle, and a text file giving the polariser angle for full transmission should be included as "linearity/default_angle.dat"
When using the exposure time method, subfolders shoudl be identified according to exposure time. Note that reduction for this has NOT been implemented yet.

The format should be as follows:
.../images/linearity/<name>_<type>.npy
where <name> can be any string identifying a dataset.

When using the linear polariser method:
.../images/linearity/pol<pangle>_<type>.npy
where <pangle> is the indicated angle on the first polariser and <type> the image type (see below).

When using the exposure time method:
# NOT YET IMPLEMENTED

This folder may be included in the '../gain/' folder via softlink.

Four types of stacked images will be present:
 - ..._mean.npy: mean per pixel of a DNG data set
 - ..._stds.npy: standard deviation per pixel of a DNG data set
 - ..._jmean.npy: mean per pixel of a JPG data set
 - ..._jstds.npy: standard deviation per pixel of a JPG data set
