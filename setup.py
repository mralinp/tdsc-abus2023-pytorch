import re
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Single source of truth: __version__ in the package (CI bumps it before publishing)
with open("tdsc_abus2023_pytorch/__init__.py", encoding="utf-8") as fh:
    version = re.search(r'__version__ = "([^"]+)"', fh.read()).group(1)

setup(
    name="tdsc-abus2023-pytorch",
    version=version,
    author="Ali Naderi Parizi",
    author_email="me@alinaderiparizi.com",
    description="PyTorch dataset for TDSC ABUS 2023",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mralinp/tdsc-abus2023-pytorch",
    packages=find_packages(exclude=["tests", "tests.*"]),
    include_package_data=True,
    package_data={"tdsc_abus2023_pytorch": ["resources/*.json"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=[
        "torch",
        "numpy",
        "pandas",
        "pynrrd",
        "gdown",
    ],
)
