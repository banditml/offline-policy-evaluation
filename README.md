
<p align="center">
  <a href="https://banditml.com" target="_blank">
    <img src="https://gradient-app-bucket-public.s3.amazonaws.com/static/images/logo.png" alt="Bandit ML" height="140">
  </a>
</p>

[![PyPI version](https://badge.fury.io/py/banditml.svg)](https://badge.fury.io/py/banditml) [![](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)


# What's banditml?

[banditml](https://github.com/banditml/banditml) is a lightweight contextual bandit & reinforcement learning library designed to be used in production Python services. This library is developed by [Bandit ML](https://www.banditml.com) and ex-authors of Facebook's applied reinforcement learning platform, [Reagent](https://github.com/facebookresearch/ReAgent).

Specifically, this repo contains:
- Feature engineering & preprocessing
- Model implementations
- Model training workflows
- Model serving code for Python services

## Supported models

Models supported:

- Contextual Bandits (small datasets)
  - [x] Linear bandit w/ ε-greedy exploration
  - [x] Random forest bandit w/ ε-greedy exploration
  - [x] Gradient boosted decision tree bandit w/ ε-greedy exploration
- Contextual Bandits (medium datasets)
  - [x] Neural bandit with ε-greedy exploration
  - [x] Neural bandit with UCB-based exploration [(via. dropout exploration)](https://arxiv.org/abs/1506.02142)
  - [x] Neural bandit with UCB-based exploration [(via. mixture density networks)](https://publications.aston.ac.uk/id/eprint/373/1/NCRG_94_004.pdf)
- Reinforcement Learning (large datasets)
  - [ ] [Deep Q-learning with ε-greedy exploration](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
  - [ ] [Quantile regression DQN with UCB-based exploration](https://arxiv.org/abs/1710.10044)
  - [ ] [Soft Actor-Critic](https://arxiv.org/abs/1801.01290)

<b>4</b> feature types supported:
* <b>Numeric:</b> standard floating point features
  * e.g. `{totalCartValue: 39.99}`
* <b>Categorical:</b> low-cardinality discrete features
  * e.g. `{currentlyViewingCategory: "men's jeans"}`
* <b>ID list:</b> high-cardinality discrete features
  * e.g. `{productsInCart: ["productId022", "productId109"...]}`
  * Handled via. learned embedding tables
* <b>"Dense" ID list:</b> high-cardinality discrete features, manually mapped to dense feature vectors
  * e.g `{productId022: [0.5, 1.3, ...], productId109: [1.9, 0.1, ...], ...}`

## Docs

```
pip install banditml
```

[Get started](DOCS.md)

## License

GNU General Public License v3.0 or later

See [COPYING](COPYING) to see the full text.
