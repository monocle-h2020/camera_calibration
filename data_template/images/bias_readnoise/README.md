This folder contains bias images. Both bias and read noise characteristics are determined from these.

Since bias and read noise are ISO speed dependent, it is recommended to separate data by ISO speed. Please note that the ISO speed in EXIF data may not be accurate, as shown in the SPECTACLE paper.

The following format is recommended:
.../images/bias/iso<ISO>/*
where <ISO> is the ISO value, and * denotes the location of the image files.
