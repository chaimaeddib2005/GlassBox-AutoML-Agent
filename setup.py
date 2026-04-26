from setuptools import setup, find_packages

setup(
    name="glassbox-automl",         
    version="1.0.0",
    description="Transparent, scratch-built AutoML library (NumPy-only core)",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="GlassBox Team",
    author_email="chaimaeddib@gmail.com", 
    url="https://github.com/chaimaeddib2005/GlassBox-AutoML-Agent",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=["numpy>=1.24"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)