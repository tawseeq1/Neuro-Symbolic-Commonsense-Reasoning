"""
Setup script for neuro-symbolic commonsense reasoning pipeline.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="neuro-symbolic-commonsense",
    version="0.1.0",
    author="NeuroSymbolic Team",
    description="A staged, modular pipeline for neuro-symbolic commonsense reasoning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/neuro-symbolic-commonsense",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "neuro-symbolic-train=scripts.train:main",
            "neuro-symbolic-eval=scripts.evaluate:main",
            "neuro-symbolic-download=scripts.download_datasets:main",
        ],
    },
) 