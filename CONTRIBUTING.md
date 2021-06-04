# Contributing to Bandit ML

## Pull Requests
We actively welcome your pull requests.

1. Fork the repo and create your branch from `master`.
2. Add tests if necessary.
3. Update the documentation if necessary.
4. Run the test suite to ensure tests pass.
5. Make sure your code lints with `black`.

## Development installation
Clone the repo & install the development requirements:
```
git clone https://github.com/banditml/banditml.git
cd banditml
```

`banditml` uses [`pipenv`](https://github.com/pypa/pipenv) to manage its environment.
```
pipenv install --python 3.7
```

You're good to go!

## Running tests

To run all unit tests:
```
pipenv run python -m unittest
```

To run unit tests for one module:
```
pipenv run python -m unittest tests.banditml.serving.test_predictor
```

## Issues
We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

## License
By contributing to Bandit ML, you agree that your contributions will be licensed
under the COPYING file in the root directory of this source tree.
