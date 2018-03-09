import rawpy
import exifread

def load_dng(filename, gamma=(1,1), output_bps=8, **kwargs):
    img = rawpy.imread(filename)
    data = img.postprocess(gamma=gamma, output_bps=output_bps, **kwargs)
    return data

def load_exif(filename):
    with open(filename, "rb") as f:
        exif = exifread.process_file(f)
    return exif