import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="jitterfitter", # Replace with your own username
    version="0.0.1",
    author="Collin Dabbieri",
    author_email="collin.m.dabbieri@vanderbilt.edu",
    description="A package for fitting radial velocity jitter to SDSS quasar spectra",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Collin-Dabbieri/jitterfitter",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
