from setuptools import setup, find_packages


VERSION = "0.0.1"
DESCRIPTION = "Postprocessing of Nek5000 ABL simulations"
LONG_DESCRIPTION = (
    "Tools for processing and plotting output from Nek5000 ABL simulations"
)

setup(
    name="abltools",
    version=VERSION,
    author="Linnea Huusko",
    author_email="linnea.huusko@gmail.com",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=["xarray"],
    keywords=["python", "Nek5000"],
    license="MIT Licence",
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT Licence",
        "Programming Language :: Python :: 3",
    ],
)
