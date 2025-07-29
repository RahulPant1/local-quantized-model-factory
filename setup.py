#!/usr/bin/env python3
"""
Setup script for Local Quantized Model Factory (LQMF)
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
with open(requirements_path, 'r') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Read long description from CLAUDE.md
readme_path = Path(__file__).parent / "CLAUDE.md"
with open(readme_path, 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="lqmf",
    version="1.0.0",
    description="Local Quantized Model Factory - Agent-powered model quantization system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="LQMF Development Team",
    python_requires=">=3.8",
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "lqmf=cli.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="quantization, model-compression, ai, machine-learning, transformers",
    project_urls={
        "Documentation": "https://github.com/your-username/lqmf",
        "Source": "https://github.com/your-username/lqmf",
        "Bug Reports": "https://github.com/your-username/lqmf/issues",
    },
)