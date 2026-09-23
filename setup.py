from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="tdsc-abus2023-pytorch",
    version="0.1.11",
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
