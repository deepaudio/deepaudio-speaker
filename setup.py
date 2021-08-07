import os
import sys
from pathlib import Path

from pkg_resources import VersionConflict, require
from setuptools import find_packages, setup

with open("Readme.md") as f:
    long_description = f.read()

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

try:
    require("setuptools>=38.3")
except VersionConflict:
    print("Error: version of setuptools is too old (<38.3)!")
    sys.exit(1)


ROOT_DIR = Path(__file__).parent.resolve()
# Creating the version file

with open("version.txt") as f:
    version = f.read()

version = version.strip()
sha = "Unknown"

if os.getenv("BUILD_VERSION"):
    version = os.getenv("BUILD_VERSION")
elif sha != "Unknown":
    version += "+" + sha[:7]
print("-- Building version " + version)

version_path = ROOT_DIR / "deepaudio" / "speaker" / "version.py"

with open(version_path, "w") as f:
    f.write("__version__ = '{}'\n".format(version))

if __name__ == "__main__":
    setup(
        name="deepaudio.speaker",
        namespace_packages=["deepaudio"],
        version=version,
        packages=find_packages(),
        install_requires=requirements,
        description="Speaker embedding",
        long_description=long_description,
        long_description_content_type="text/markdown",
        author="Ruiqing Yin",
        # author_email="yinruiqing",
        url="https://github.com/deepaudio/deepaudio-speaker",
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: MIT License",
            "Natural Language :: English",
            "Programming Language :: Python :: 3.8",
            "Topic :: Scientific/Engineering",
        ],
    )