## How to release package to PyPi:
## 1) cd in package directory banditml/banditml_pkg/
## 2) bump version in setup.py
## 3) python setup.py sdist bdist_wheel
## 4) twine upload dist/* --skip-existing
## 5) See new version at https://pypi.org/project/banditml/


import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="banditml",
    version="0.0.4",
    author="Edoardo Conti, Lionel Vital",
    author_email="edoardo.conti@gmail.com",
    description="Portable Bandit ML code for training & serving consistency.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/banditml/banditml",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=["pandas", "torch", "sklearn", "dill", "skorch"],
)
