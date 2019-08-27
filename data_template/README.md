Template for storing SPECTACLE data:

- analysis: scientific results, such as plots and tables.
- calibration: calibration data, such as look-up tables, maps, and model parameters.
- images: images from the camera (RAW or JPEG).
- intermediaries: script outputs which are used in further calibration or analysis scripts, but are not themselves directly interesting to most users.
- stacks: pre-processed image stacks, with the mean and standard deviation per pixel.

Please note that this is only a suggested format, and it is not necessary to follow this. Currently many scripts make assumptions on file locations, such as the presence of `analysis` and `calibration` folders. Making this dynamic is a future goal.
