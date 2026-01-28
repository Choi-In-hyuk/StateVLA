from setuptools import setup, find_packages

setup(
    name="statevla",
    version="0.1.0",
    description="StateVLA: State-based Vision-Language-Action Model with Predictive Feedback",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/StateVLA",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.21.0",
        "mamba-ssm>=1.2.0",
        "pyyaml>=6.0",
        "h5py>=3.7.0",
        "einops>=0.6.0",
        "opencv-python>=4.7.0",
        "ftfy>=6.1.0",
        "regex>=2022.1.18",
        "tqdm>=4.64.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
        ],
        "wandb": [
            "wandb>=0.15.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
