## How to release package to PyPi:
## 1) cd in package directory ope/
## 2) bump version in setup.py
## 3) python setup.py sdist
## 4) twine upload dist/* --skip-existing
## 5) See new version at https://pypi.org/project/offline-evaluation/


import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="offline-evaluation",
    version="0.0.6",
    author="Edoardo Conti, Lionel Vital, Joseph Gilley",
    author_email="team@banditml.com",
    description="Implementations of common offline policy evaluation methods.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/banditml/offline-policy-evaluation",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "pandas~=1.1.5",
        "scikit-learn~=0.24.2",
    ],
)
