from setuptools import setup, find_packages

setup(
    name="glassbox-automl",
    version="1.0.0",
    description="Transparent, scratch-built AutoML library (NumPy-only core)",
    author="GlassBox Team",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=["numpy>=1.24"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
