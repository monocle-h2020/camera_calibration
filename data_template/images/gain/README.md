This folder will contain raw *gain* images (DNG) directly from the camera.

There will be subfolders for different ISO values. Each ISO subfolder is further split into subfolders for different lighting conditions.

ISO values are based on app values, not on camera values.

The format should be as follows:
.../images/gain/iso<ISO>/<name>/<filename>.<raw>
where <ISO> is the ISO value, <name> can be any string identifying a dataset (e.g. lighting conditions, exposure time, timestamp, ...), <filename> are the individual filenames, and <raw> is the RAW file extension.

This folder may contain a softlink to the '../linearity' folder containing similar data for a certain ISO value. This softlink will follow the naming format described above.
