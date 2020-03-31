# Do it yourself quick start

This guide walks through training and serving a bandit model.

## Installation
Clone the repo & install the requirements:
```
git clone https://github.com/banditml/banditml.git
cd banditml
virtualenv env
. env/bin/activate
pip install -r requirements.txt
```
You're good to go!
## Data format
Bandit ML expects data to be stored in [Google BigQuery](https://cloud.google.com/bigquery). Specifically, two tables are needed (`decisions` and `rewards`)

`decisions` table:
- <b>experiment_id (STRING)</b>: An ID for the overall experiment this observation is part of.  Models are trained on samples within the same experiment.
- <b>context (STRING)</b>: A JSON string holding a map of the context used to make a decision (i.e. the features that are input to the model). Map values are either `int` or `float` (no `str` values).
- <b>decision (INT)</b>: The decision made mapped to its corresponding integer representation.
- <b>decision_id (STRING)</b>: The ID of this decision. Used to join subsequent rewards to this decision.
- <b>mdp_id (STRING)</b>: The ID of the markov decision process this sample is a part of. Typically in recommendation problems each individual recommendation is part of an overall session (in this case these samples would be part of the same session ID which would serve as the `mdp_id`). Reinforcement learning algorithms learn over MDPs.
- <b>variation_id (INT)</b>: An ID indicating which variant of the experiment this decision is a part of. Used to compute A/B test results across variants.
- <b>ts (INT)</b>: timestamp in seconds of when this decision was made.

Sample records:
```
 row | decision_id |            context           | decision |   experiment_id   | mdp_id | variant_id |     ts     |
-----|-------------|------------------------------|----------|-------------------|--------|------------|------------|
  1  |   c2aa520f  | {"country": 2, "year": 1796} |     1    | height-prediction |    a   |      2     | 1585612647 |
  2  |   f3e637a5  | {"country": 4, "year": 2017} |     2    | height-prediction |    b   |      1     | 1585834128 |
```

`rewards` table:
- <b>decision_id (STRING)</b>: The ID of the decision to join this reward to. Sometimes decisions are lists (e.g. ranking problems). In this case all items in the ranked list should be logged as seperate rows and have the same `decision_id` with a numeric `position` feature in `context` to de-bias the position effect in the decision.
- <b>decision (INT)</b>: The decision this reward joins to. Rewards are joined using both `decision_id` and `decision` since `decision_id` is not always unique (when decisions are ranked lists of several items).
- <b>metrics (STRING)</b>: A JSON string holding a map of the reward metrics. Used to construct a reward. Map values are either `int` or `float` (no `str` values).
- <b>experiment_id (STRING)</b>: See description in `decisions` table above.
- <b>mdp_id (STRING)</b>: See description in `decisions` table above.
- <b>ts (INT)</b>: See description in `decisions` table above.

Sample records:
```
 Row | decision_id | decision |       metrics       |   experiment_id   | mdp_id |     ts     |
-----|-------------|----------|---------------------|-------------------|--------|------------|
  1  |   c2aa520f  |     1    | {"height": 158.462} | height-prediction |    a   | 1585613418 |
  2  |   f3e637a5  |     2    | {"height": 172.331} | height-prediction |    b   | 1585934016 |
  3  |             |          |     {"1": 10.5}     | height-prediction |    a   | 1599999016 |
```

**IMPORTANT:**  <b>*Immediate*</b> rewards vs. <b>*delayed*</b> rewards:

Rewards can be `immediate` (e.g. a click on something recommeded) or `delayed` (e.g. a user purchases something you previously recommended at the end of a session after some browsing and after other decisions have been made). In the `rewards` table we handle both of these types of reward and use the following convention to distinguish them. `immedate` rewards should be logged with both a `decision_id` and a `decision` (e.g. rows `1` and `2` above). These reward rows are simply joined to their corresponding rows in the `decisions table`. `delayed` rewards should be logged without a `decision_id` and `decision` (e.g. row `3` above). `delayed` rewards should have keys in `metrics` that correspond to decisions previously made throughout the MDP. banditml joins these delayed rewards to the correct corresponding rows in the `decisions` table based on the `mdp_id` and where those specific decisions were chosen throughout the MDP.

In the example above, `decision_id` `c2aa520f` would get an additional `10.5` reward at training time as we have a matching delayed reward in row `3` based on `mdp_id` and key `1` in `metrics`.

## Training a model

Once data is in BigQuery in the proper format, we can train a model. To train a model, create a config file that specifies information about the model, features, and reward function. For a sample config, see [example_experiment_config.json](configs/example_experiment_config.json).

Then, to train a model simply run:

```
 python -m workflows.train_bandit \
     --params_path configs/bandit.json \
     --experiment_config_path configs/example_experiment_config.json \
     --model_path trained_models/model.pkl
```

This saves a model `model.pkl` in `trained_models`.

## Serving a model

To serve the model in a Python service see the sample workflow `workflows/predict.py`. This module outlines how to use the saved model to serve predictions. You can test your model's predictions by running this workflow:

```
python -m workflows.predict \
	--model_path trained_models/model.pkl
```

To make model portability easier, you can use the [banditml pip package](https://pypi.org/project/banditml/) to
serve your model in a Python service like this:

```
import sys

import banditml
from banditml import model_io

sys.modules["banditml_pkg"] = banditml
sys.modules["banditml_pkg.banditml"] = banditml

path = "/some/model/path/model.pkl"

predictor = model_io.read_predictor_from_disk(path)
print(predictor.predict({"country": 5, "year": 1990}))
```

## Running tests

To run all unit tests:
```
python -m unittest
```

To run unit tests for one module:
```
python -m unittest tests.banditml_pkg.banditml.serving.test_predictor
```
