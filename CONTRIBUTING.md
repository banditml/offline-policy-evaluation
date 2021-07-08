# Contributing to Bandit ML

## Pull Requests
We actively welcome your pull requests.

1. Fork the repo and create your branch from `master`.
2. Update the documentation if necessary.
3. Make sure your code lints with `black`.

## Development installation
Clone the repo & install the development requirements:
```
git clone https://github.com/banditml/offline-policy-evaluation.git
```

```
cd offline-policy-evaluation
virtualenv env
. env/bin/activate
pip install -r dev-requirements.txt
```

To test changes to the library, run the notebooks in `examples/` which use the local version of the library.

## Issues
We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

## License
By contributing to banditml/offline-policy-evaluation, you agree that your
contributions will be licensed under the COPYING file in the root directory of this source tree.
