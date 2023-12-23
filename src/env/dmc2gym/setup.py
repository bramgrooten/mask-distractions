import os
from setuptools import setup
from setuptools import find_packages

# for gym I added ==0.21.0, avoiding conflicts with other parts of the repo [Bram]
setup(
    name='dmc2gym',
    version='1.0.0',
    author='Denis Yarats',
    description=('a gym like wrapper for dm_control'),
    license='',
    keywords='gym dm_control openai deepmind',
    packages=find_packages(),
    install_requires=[
        'gym==0.21.0',
        'dm_control',
    ],
)
