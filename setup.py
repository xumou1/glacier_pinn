#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
青藏高原冰川PINNs建模项目安装脚本
Tibetan Plateau Glacier PINNs Modeling Project Setup
"""

from setuptools import setup, find_packages
import os

# 读取README文件
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# 读取requirements文件
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="tibetan-glacier-pinns",
    version="0.1.0",
    author="Glacier Research Team",
    author_email="glacier.research@example.com",
    description="Physics-Informed Neural Networks for Tibetan Plateau Glacier Evolution Modeling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/glacier-research/tibetan-glacier-pinns",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: GIS",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "black>=21.6.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
            "jupyter>=1.0.0",
        ],
        "gpu": [
            "cupy>=9.0.0",
            "torch-geometric>=2.0.0",
        ],
        "viz": [
            "streamlit>=1.0.0",
            "dash>=2.0.0",
            "plotly>=5.0.0",
            "bokeh>=2.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "glacier-pinns=main_experiment:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yml", "*.yaml", "*.json", "*.txt"],
    },
    zip_safe=False,
)