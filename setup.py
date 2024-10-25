from setuptools import setup, find_packages

setup(
    name='hym',
    version='v0.1.2',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=[
        'openpyxl',
        'numpy',
        'matplotlib',
        'pandas'
    ],
)
