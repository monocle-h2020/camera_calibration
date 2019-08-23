This folder contains flat-field images.

Since many flat-fielding methods exist, it is recommended to split data from different methods into different folders.

The following format is recommended:
.../images/flat/<method>/*
where <method> is the method used in flatfielding (e.g. an integrating sphere, the sun, etc.), and * denotes the location of the image files.

The SPECTACLE code base currently does not account for differences in flat-field due to changes in aperture or focus.
