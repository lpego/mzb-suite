import os
from setuptools import setup


# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="mzbsuite",
    version="0.2.3",
    author="Michele Volpi, Luca Pegoraro",
    author_email="mivolpi@ethz.ch",
    description=(
        "A suite of tools and utilities for the analysis of macrozoobenthos images"
    ),
    license="BSD",
    keywords="macrozoobenthos skeletonization classificaiton insects benthos",
    url="",
    packages=[],
    long_description=read("README.md"),
    classifiers=[
        "Development Status :: 3 -- Alpha",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: BSD License",
    ],
)
