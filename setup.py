from setuptools import setup, find_packages

setup(
    name="parallelcore",
    version="0.1.0",
    description="High-performance parallel computing library for CPU with SIMD tensors",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "numba>=0.58.0",
        "psutil>=5.9.0",
    ],
    python_requires=">=3.8",
    author="ParallelCore Team",
    author_email="team@parallelcore.dev",
    url="https://github.com/parallelcore/parallelcore",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)