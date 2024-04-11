"""
Package setup
"""

from pathlib import Path
from setuptools import find_namespace_packages, setup


BASE_DIR = Path(__file__).parent
with open(Path(BASE_DIR, "requirements.txt"), "r", encoding='utf-8') as file:
    required_packages = file.read().splitlines()

setup(
    name="src",
    version="0.0.1",
    description="Fit Model - Titanic Data set",
    author="Cristian Najera",
    author_email="cristianenajera@gmail.com",
    python_requires=">=3.8",
    packages=find_namespace_packages(),
    install_requires=required_packages,
)
