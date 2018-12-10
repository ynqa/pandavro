from setuptools import setup
from setuptools import find_packages

setup(
    name='pandavro',
    version='1.3.0',
    description='The interface between Avro and pandas DataFrame',
    url='https://github.com/ynqa/pandavro',
    author='Makoto Ito',
    author_email='un.pensiero.vano@gmail.com',
    license='MIT',
    packages=find_packages(exclude=['example']),
    install_requires=[
        'fastavro>=0.14.7',
        'numpy',
        'pandas',
    ],
    extras_require={
        'tests': ['pytest'],
    },
)
