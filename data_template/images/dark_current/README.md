This folder contains dark current images, taken at varying exposure times.

Since dark current is ISO speed dependent, it is recommended to separate data by ISO speed. Please note that the ISO speed in EXIF data may not be accurate, as shown in the SPECTACLE paper.

The following format is recommended:
.../images/dark_current/iso<ISO>/t_<exposure_time>/*
where <ISO> is the ISO value, <exposure_time> is the exposure time (e.g. `t_1` for an exposure of 1 second, or `t_1_5.2` for an exposure of 1/5.2 seconds), and * denotes the location of the image files.
