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
- Feauture engineering & preprocessing
- Model implementations
- Model training workflows

## Supported models
All models can handle 3 types of features:
* Numeric (standard floating point features)
  * (e.g. `[12.5, 1.3, ...]`)
* Categorical (low-cardinality discrete features)
  * (e.g. `['t-shirt', 'jeans', ...]`)
* ID list (high-cardinality discrete features)
  * (e.g. `['productId1', 'productId2', ..., 'productId1001']`)

Models supported
- [x] Neural contextual bandit with ε-greedy exploration
- [ ] [Neural contextual bandit with UCB-based exploration](https://arxiv.org/abs/1911.04462)
- [ ] [Deep Q-learning with ε-greedy exploration](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
- [ ] [Soft Actor-Critic](https://arxiv.org/abs/1801.01290)


## Docs
Coming soon!
