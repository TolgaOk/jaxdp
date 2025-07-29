from setuptools import setup, find_packages

# Read requirements from requirements.txt
def read_requirements():
    with open('requirements.txt', 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    # Metadata
    name="jaxdp",
    version="0.3.0",
    author="Tolga Ok",
    author_email="T.Ok@tudelft.nl",
    url="",
    description="A Dynamic Programming package for discrete MDPs implemented in JAX",
    long_description=(""),
    license="MIT",
    python_requires=">=3.11",

    # Package info
    packages=find_packages(include=["jaxdp", "jaxdp.*"]),
    install_requires=read_requirements(),
    zip_safe=False
)