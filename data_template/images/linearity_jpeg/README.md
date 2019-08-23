This folder contains raw *linearity* images (DNG & JPG) directly from the camera.

There will be subfolders for different lighting conditions. 
When using the linear polariser method, subfolders should be identified according to polariser angle, and a text file giving the polariser angle for full transmission should be included as "linearity/default_angle.dat"
When using the exposure time method, subfolders shoudl be identified according to exposure time. Note that reduction for this has NOT been implemented yet.

The format should be as follows:
.../images/linearity/<name>/
where <name> can be any string identifying a dataset.

When using the linear polariser method:
.../images/linearity/pol<pangle>/
where <pangle> is the indicated angle on the first polariser

When using the exposure time method:
# NOT YET IMPLEMENTED

This folder may be included in the '../gain/' folder via softlink.
