from setuptools import find_packages
from setuptools import setup

setup(
    name="cifar10-net",
    version="0.1",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "lightning==2.1.2",
        "torchvision==0.14.0",
        "matplotlib==3.8.2",
        "wandb==0.24.2",
        "jsonargparse[signatures]==4.27.1",
        "rich==13.7.0",
    ],
    description="CIFAR10 classifier in PyTorch Lightning",
)
