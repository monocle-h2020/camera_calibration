This is a to-do list for the `spectacle` module.

## Module

- [ ] Remove AstroPy dependencies for simplicity
- [ ] Fix silent deprecation warnings
- [ ] Fix silent error when trying to load calibration data if multiple files exist
- [ ] Make [load_monochromator_data](https://github.com/monocle-h2020/camera_calibration/blob/cb9a358e7b965816440831b77a4e49a5760d90cc/spectacle/spectral.py#L41) use a `Camera` object from args, rather than `root`

## Scripts

#### Changes to existing scripts
- [ ] Convert all command-line inputs to `optparse` format.
- [ ] Merge [stack_mean_std.py](tools/stack_mean_std.py) and [stack_heavy.py](tools/stack_heavy.py).
- [ ] Make error data optional in [flatfield_characterise_data.py](analysis/flatfield_characterise_data.py).
- [ ] Add varying apertures to [camera_settings.py](calibration/camera_settings.py).

#### New scripts
- [ ] Analyse the deviations between ISO normalisation look-up tables and the expected behaviour (normalisation of 1/ISO).
- [ ] More intuitive `Camera` creation, for example using just a folder name (so the user never sees `root`)

## Documentation

- [ ] Add metadata on all calibration files, including origins and quality of data.

### READMEs

### Module

- [ ] Add detailed information on inputs, outputs, and possible errors to all `spectacle` methods.

### Scripts
