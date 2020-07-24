This branch represents a major update where calibration methods are integrated into the `Camera` object.

This is a sub-branch dedicated to renaming the `metadata` submodule to `camera`, including all related functions and documentation.

- [x] Integrate all calibration methods into `Camera`
- [x] Update all scripts to use `Camera`-based calibrations
- [ ] More intuitive `Camera` creation, for example using just a folder name (so the user never sees `root`)
- [x] Add band information from RawPy to `Camera`
- [ ] Extend `Camera` and function documentation
- [ ] Update READMEs
- [x] Rename `metadata` to `camera` and update all code
- [x] Rename `load_metadata` to `load_camera` and update all code
- [x] Print camera info on loading instead of "Loaded metadata"
