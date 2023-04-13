"""Install dependencies and build with pip."""
from setuptools import find_packages, setup

setup(
    name="prompt2model",
    version="0.0.1",
    packages=find_packages(include=["prompt2model"]),
    install_requires=[
        "transformers==4.24.0",
        "datasets==2.11.0",
        "pandas==1.5.3",
        "gradio==3.24.1",
    ],
)
