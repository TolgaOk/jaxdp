from setuptools import setup, find_packages

setup(
    # Metadata
    name="jaxdp",
    version="0.2.0",
    author="Tolga Ok",
    author_email="T.Ok@tudelft.nl",
    url="",
    description="A Dynamic Programming package for discrete MDPs implemented in JAX",
    long_description=(""),
    license="MIT",

    # Package info
    packages=find_packages(include=["jaxdp", "jaxdp.*"]),
    install_requires=[
    ],
    zip_safe=False
)