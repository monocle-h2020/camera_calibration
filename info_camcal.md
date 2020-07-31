This branch represents a major update where calibration methods are integrated into the `Camera` object.

- [x] Integrate all calibration methods into `Camera`
- [x] Update all scripts to use `Camera`-based calibrations
- [x] Add band information from RawPy to `Camera`
- [x] Extend `Camera` and function documentation
- [x] Update READMEs
- [x] Rename `metadata` to `camera` and update all code
- [x] Rename `load_metadata` to `load_camera` and update all code
- [x] Print camera info on loading instead of "Loaded metadata"
- [x] Split ISO/exposure information from main camera data file
- [x] Add basic file loading functions
- [x] Add basic functions from spectacle.general
- [x] Flatten Camera object by integrating Device and Image properties
- [x] Prevent bugs with new Settings interface
- [x] Add human-readable Camera names
- [x] Use human-readable Camera names in scripts
- [x] Include Camera name in camera.json and calibration files
