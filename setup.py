from setuptools import setup, find_packages

setup(
    name="nhsnet",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.19.2",
        "scipy>=1.7.0",
        "torchvision>=0.10.0",
        "tqdm>=4.62.0",
    ],
    author="Raktim Mondol",
    author_email="raktimmondol@gmail.com",
    description="NHS-Net: A Biologically-Inspired Neural Architecture",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/raktim-mondol/nhsnet",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)