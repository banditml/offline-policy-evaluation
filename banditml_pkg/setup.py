## How to release package to PyPi:
## 1) cd in package directory banditml_pkg/
## 2) bump version in setup.py
## 3) pipenv run python setup.py sdist bdist_wheel
## 4) pipenv run twine upload dist/* --skip-existing
## 5) See new version at https://pypi.org/project/banditml/


import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="banditml",
    version="0.6.3",
    author="Edoardo Conti, Lionel Vital, Joseph Gilley",
    author_email="team@banditml.com",
    description="Portable Bandit ML code for training & serving consistency.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/banditml/banditml",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "google-cloud-bigquery~=1.24",
        "pandas>=1.0.3,<1.1",
        "scikit-learn>=0.22.2.post1,<0.23",
        "torch~=1.4",
        "typing_extensions>=3.7.4",
    ],
)
