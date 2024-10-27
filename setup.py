from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='hym20220592',
    version='v0.1.6',
    author='Yunming Hu',
    author_email='hugonelsonm3@gmail.com',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=[
        'openpyxl',
        'numpy',
        'matplotlib',
        'pandas'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="machine learning, data science, artificial intelligence",
    description='Machine-Learning from scratch',
    long_description=long_description,
    long_description_content_type='text/markdown',
)
