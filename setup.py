from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="tdsc-abus2023-pytorch",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="PyTorch dataset for TDSC ABUS 2023",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/tdsc-abus2023-pytorch",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.7.0",
        "numpy>=1.19.0",
        "pandas>=1.2.0",
    ],
) 