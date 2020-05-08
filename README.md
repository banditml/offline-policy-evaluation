<p align="center">
  <a href="https://banditml.com" target="_blank">
    <img src="https://gradient-app-bucket-public.s3.amazonaws.com/static/images/bandit_full_logo.png" alt="Bandit ML" height="72">
  </a>
</p>
<p align="center">
  A lightweight & open source framework for personalization
</p>
</p>

[![PyPI version](https://badge.fury.io/py/banditml.svg)](https://badge.fury.io/py/banditml) [![](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)


# What's Bandit ML?

[Bandit ML](https://www.banditml.com) is a lightweight decision making & personalization framework built by experts from Facebook's applied reinforcement learning team, [Reagent](https://github.com/facebookresearch/ReAgent). Bandit ML gives users access to state of the art contextual bandit and reinforcement learning algorithms through a simple javascript widget.

This repo holds the open source machine learning code that powers [banditml.com](https://www.banditml.com). Specifically, this repo contains:
- Feature engineering & preprocessing
- Model implementations
- Model training workflows

## Supported models

Models supported:

- Contextual Bandits (small datasets)
  - [x] Linear bandit w/ ε-greedy exploration
  - [x] Random forest bandit w/ ε-greedy exploration
  - [x] Gradient boosted decision tree bandit w/ ε-greedy exploration
- Contextual Bandits (medium datasets)
  - [x] Neural bandit with ε-greedy exploration
  - [x] Neural bandit with UCB-based exploration [(via. dropout exploration)](https://arxiv.org/abs/1506.02142)
  - [ ] Neural bandit with UCB-based exploration [(via. mixture density networks)](https://publications.aston.ac.uk/id/eprint/373/1/NCRG_94_004.pdf)
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

If you just want to train a model for free and do everything else yourself these are the docs for you:

[Do it yourself quick start](DOCS.md)

Alternatively, the  [hosted solution](https://www.banditml.com)  offers an end-to-end service for integrating contextual bandit and reinforcement learning algorithms into your application. Specifically, the end-to-end service offers:
- A UI to create, manage, and view bandit experiments
- A lightweight javascript client library to log data, get decisions, and provide feedback about decisions
- A UI to monitor, train, and evaluate bandit models
- Model understanding & insight tools

The docs to the hosted solution can be found [here](https://www.banditml.com/docs/).

## License

GNU General Public License v3.0 or later

See [COPYING](COPYING) to see the full text.
