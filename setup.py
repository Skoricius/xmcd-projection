from setuptools import find_packages, setup
import os


# load README.md as long_description
long_description = ''
if os.path.exists('README.md'):
    with open('README.md', 'r') as f:
        long_description = f.read()

setup(
    name='XMCD Projection',
    version='0.1.0',
    packages=find_packages(include=['xmcd_projection']),
    description='Library for simulating XMCD projection signal',
    long_description=long_description,
    author='Luka Skoric',
    license='GNU GENERAL PUBLIC LICENSE',
    install_requires=[
        'trimesh>=3.9.12',
        'numpy>=1.20.2',
        'matplotlib>=3.4.1',
        'numba>=0.53.1',
        'joblib>=1.0.1',
        'PyQt5>=5.15.4',
        'pyqtgraph>=0.11.1',
        'scikit-image>=0.18.1',
        'scipy>=1.6.2',
        'PyOpenGL>=3.1.5',
        'cached-property>=1.5.2',
    ]
)
