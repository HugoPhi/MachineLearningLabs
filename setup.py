from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='hym20220592',
    version='v0.1.5',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=[
        'openpyxl',
        'numpy',
        'matplotlib',
        'pandas'
    ],
    long_description=long_description,
    long_description_content_type='text/markdown',
)
