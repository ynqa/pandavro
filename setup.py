from setuptools import setup
from setuptools import find_packages

setup(
    name='pandavro',
    version='1.5.0',
    description='The interface between Avro and pandas DataFrame',
    url='https://github.com/ynqa/pandavro',
    author='Makoto Ito',
    author_email='un.pensiero.vano@gmail.com',
    license='MIT',
    packages=find_packages(exclude=['example']),
    install_requires=[
        'fastavro>=0.14.11',
        'numpy>=1.7.0',
        'pandas',
        'six>=1.9',
    ],
    extras_require={
        'tests': ['pytest'],
    },
    python_requires='>=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*',
)
