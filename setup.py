from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setup(
    name="jackknify", 
    version='0.1.0',
    author="Originally by Joshiwa van Marrewijk. Editied by Talon Myburgh",
    author_email="joshiwa01@gmail.com and myburgh.talon@gmail.com",
    description="Jackknifing interferometric datasets",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/talonmyburgh/jackknify", 
    packages=find_packages(),
    install_requires=[
        'numpy',
        'jax',
        'jaxlib',
        'python-casacore',
        'astropy',
        'click',
        'tqdm',
    ],
    entry_points='''
        [console_scripts]
        jackknify=jackknify.Cli:cli
    ''',
    )
