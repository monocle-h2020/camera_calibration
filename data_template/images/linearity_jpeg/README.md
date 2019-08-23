This folder contains linearity images. Both linearity and JPEG characteristics are determined from these.

It is recommended to separate data by exposure. Multiple methods of obtaining different exposures are currently supported, namely:

* Linear polarisers: subfolders should be identified according to polariser angle, and a text file giving the polariser angle for full transmission should be included at "linearity/default_angle.dat". The following format is recommended: .../images/linearity/pol<pangle>/*, where <pangle> is the angle indicated on the rotating polariser, and * denotes the location of the image files.

* Exposure time: subfolders should be identified according to exposure time. The following format is recommended: .../images/linearity/t_<exposure_time>/*, where <exposure_time> is the exposure time (e.g. `t_1` for an exposure of 1 second, or `t_1_5.2` for an exposure of 1/5.2 seconds) and * denotes the location of the image files.
