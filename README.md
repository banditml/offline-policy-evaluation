<p align="center">
  <p align="center">
    <a href="https://banditml.com" target="_blank">
      <img src="https://gradient-app-bucket-public.s3.amazonaws.com/static/images/bandit_full_logo.png" alt="Bandit ML" height="72">
    </a>
  </p>
  <p align="center">
    A lightweight & open source framework for personalization
  </p>
</p>

# What's Bandit ML?

[Bandit ML](https://banditml.come) is a lightweight decision making & personalization framework built by experts from Facebook's applied reinforcement learning team, [Reagent](https://github.com/facebookresearch/ReAgent). Bandit ML gives users access to state of the art contextual bandit and reinforcement learning algorithms through a simple javascript widget.

This repo holds the open source machine learning code that powers banditml.com. Specifically, this repo contains:
- Feature engineering & preprocessing
- Model implementations
- Model training workflows

## Supported models

Models supported:
- [x] Neural contextual bandit with ε-greedy exploration
- [ ] [Neural contextual bandit with UCB-based exploration](https://arxiv.org/abs/1911.04462)
- [ ] [Deep Q-learning with ε-greedy exploration](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
- [ ] [Soft Actor-Critic](https://arxiv.org/abs/1801.01290)

All models can handle <b>4</b> types of features:
* <b>Numeric:</b> (standard floating point features)
  * e.g. `[12.5, 1.3, ...]`
* <b>Categorical:</b> (low-cardinality discrete features)
  * e.g. `['t-shirt', 'jeans', ...]`
* <b>ID list:</b> (high-cardinality discrete features)
  * e.g. `['productId1', ..., 'productId1001']`
  * Handled via. learned embedding tables
* <b>"Dense" ID list:</b> (high-cardinality discrete features, manually mapped to dense feature vectors)
  * e.g `[productId1': [0.5, 1.3..], ..., 'productId1001': [1.9, 0.1..]]`)

## Docs
Official docs coming soon!

Simple docs below:

Run all tests:
```
python -m unittest
```

Run tests for one module:
```
python -m unittest tests.ml.serving.test_predictor
```
