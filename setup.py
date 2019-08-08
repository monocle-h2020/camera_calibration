from setuptools import setup

setup(
      name="spectacle",
      version="0.1",
      description="SPECTACLE camera calibration module",
      author="Olivier Burggraaff",
      author_email="burggraaff@strw.leidenuniv.nl",
      packages=["spectacle"],
      install_requires=["numpy", "scipy", "matplotlib", "rawpy", "exifread", "astropy"]
)