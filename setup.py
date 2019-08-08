from setuptools import setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
      name="spectacle",
      version="0.1",
      description="SPECTACLE camera calibration module",
      long_description = long_description,
      author="Olivier Burggraaff",
      author_email="burggraaff@strw.leidenuniv.nl",
      packages=["spectacle"],
      install_requires=["numpy", "scipy", "matplotlib", "rawpy", "exifread", "astropy"]
)