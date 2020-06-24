from setuptools import setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
      name="pyspectacle",
      version="1.1.3",
      description="SPECTACLE camera calibration module",
      long_description = long_description,
      url="https://github.com/monocle-h2020/camera_calibration",
      author="Olivier Burggraaff",
      author_email="burggraaff@strw.leidenuniv.nl",
      packages=["spectacle"],
      install_requires=["numpy", "scipy", "matplotlib", "rawpy", "exifread", "astropy"],
      classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
      ],
      python_requires='>=3.6'
)
