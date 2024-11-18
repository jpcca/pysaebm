from setuptools import setup, find_packages
import os

# Read the content of your README.md file
long_description = ""
if os.path.exists("README.md"):
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()

setup(
    name="alabEBM",  # Package name
    version="0.0.5", 
    packages=find_packages(),  # Automatically find subpackages
    description="Implementation of event-based models for degenerative diseases.",
    long_description=long_description,  # Use the content of README.md
    long_description_content_type="text/markdown",  # Specify Markdown format
    author="Hongtao Hao and Joseph Austerweil",
    author_email="hongtaoh@cs.wisc.edu",
    url="https://github.com/hongtaoh/alab-ebm",  # Link to your repository or project page
    install_requires=[
        "pandas",
        "numpy",
        "scipy",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",  # Specify Python version compatibility
    license="MIT",  # License type
    include_package_data=True,  # Include non-code files (like `data/`) in the package
)
