from setuptools import setup
from setuptools import find_packages

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='pandavro',
    version='1.7.2',
    description='The interface between Avro and pandas DataFrame',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/ynqa/pandavro',
    author='Makoto Ito',
    author_email='un.pensiero.vano@gmail.com',
    license='MIT',
    packages=find_packages(exclude=['example']),
    install_requires=[
        # fixed versions.
        'fastavro~=1.7.3',
        'pandas>=1.1',
        # https://pandas.pydata.org/pandas-docs/version/1.1/getting_started/install.html#dependencies
        'numpy>=1.15.4',
    ],
    extras_require={
        'tests': ['pytest==7.1.2'],
    },
    # https://pandas.pydata.org/pandas-docs/version/1.1/getting_started/install.html#python-version-support
    python_requires='>=3.6.1',
)
