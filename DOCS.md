
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
- <b>decision_id (STRING)</b>: The unique ID of this decision. Used to join subsequent rewards to this decision.

Sample records:
```
 Row | decision_id |            context           | decision |   experiment_id   |
-----|-------------|------------------------------|----------|-------------------|
  1  |   c2aa520f  | {"country": 2, "year": 1796} |     1    | height-prediction |
  2  |   f3e637a5  | {"country": 4, "year": 2017} |     2    | height-prediction |  
```

`rewards` table:
- <b>decision_id (STRING)</b>: The unique ID of the decision this reward joins to.
- <b>metrics (STRING)</b>: A JSON string holding a map of the reward metrics. Used to construct a reward. Map values are either `int` or `float` (no `str` values).

Sample records:
```
 Row | decision_id |        metrics      |
-----|-------------|---------------------|
  1  |   c2aa520f  | {"height": 158.462} |
  2  |   f3e637a5  | {"height": 172.331} |
```

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

## Running tests

To run all unit tests:
```
python -m unittest
```

To run unit tests for one module:
```
python -m unittest tests.ml.serving.test_predictor
```
