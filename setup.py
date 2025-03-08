from setuptools import setup, find_packages

setup(
    name="tdsc-abus2023-pytorch",
    version="0.1.1",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "pynrrd",
        "torch",
        "requests",
        "gdown",
    ],
    author="Ali Naderi Parizi",
    author_email="me@alinaderiparizi.com",
    description="A PyTorch dataset for TDSC ABUS 2023",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/mralinp/tdsc-abus2023-pytorch",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
) 