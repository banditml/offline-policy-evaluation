
# Quick start

This guide walks through how to train and serve models.

## Installation
```
pip install banditml
```

## Data format
`banditml` expects data to be stored in [Google BigQuery](https://cloud.google.com/bigquery). Specifically, a BigQuery query string should be passed into `banditml.get_dataset(query_str)` that returns a dataset with the following schema:

- <b>mdp_id (STRING)</b>: The ID of the markov decision process this sample is a part of. Typically in recommendation problems each individual recommendation is part of an overall session (in this case these samples would be part of the same session ID which would serve as the `mdp_id`). Reinforcement learning algorithms learn over MDPs.
- <b>sequence_number (INT)</b>: The ordinal position of a decision within an `mdp`.
- <b>context (STRING)</b>: A JSON string holding a map of the context used to make a decision (i.e. the features that are input to the model). Map values are either `numeric` (for numeric features) or `str` (for categorical or ID list features).
- <b>decision (STRING)</b>: The decision made.
- <b>reward (FLOAT)</b>: A score indicating how good this decision was.

Sample records:
```
| row | mdp_id | sequence_number | context                      | decision | reward |
|-----|--------|-----------------|------------------------------|----------|--------|
|  1  |  u123  |        1        | {"country": 2, "year": 1796} |  female  |  154.2 |
|  2  |  u456  |        1        | {"country": 4, "year": 2017} |   male   |  170.9 |
```

## Training a model

Once a dataset matching the schema above is fetched, we can train a model. Follow the steps below to get training:


Import `banditml` library:
```
from banditml.training import trainer
from banditml.data_reader.reader import BigQueryReader
```

Fetch training data:
```
bq_reader = BigQueryReader("/path/to/bq_creds.json")
query_str = "select * from table"

df = bq_reader.get_dataset(query_str)
```

Define ml config and features config:
```
ml_config = {
  "model_name": "model_v1",
  "features": {
    "features_to_use": ["*"],
    "dense_features_to_use": ["*"]
  },
  "feature_importance": {
    "calc_feature_importance": True,
    "keep_only_top_n": False,
    "n": 10
  },
  "model_type": "neural_bandit",
  "reward_type": "binary",
  "model_params": {
    "neural_bandit": {
      "max_epochs": 50,
      "batch_size": 256,
      "layers": [-1, 64, 32, -1],
      "activations": ["relu", "relu", "linear"],
      "dropout_ratio": 0.2,
      "learning_rate": 0.001,
      "l2_decay": 0.001
    }
  },
  "train_percent": 0.85
}
```

```
features_config = {
  "choices": ["a", "b", "c"],
  "features": {
      "total_orders": {"type": "N"},
      "days_since_last_order": {"type": "N"},
      "median_days_between_orders": {"type": "N"},
      "avg_order_size": {"type": "N"},
      "company_p50_total_orders": {"type": "N"},
      "company_p50_avg_order_size": {"type": "N"},
      "company_p50_days_between_orders": {"type": "N"},
      "decision": {"type": "C"}
  },
  "product_sets": {}
}
```

Train the model:
```
predictor = trainer.train(df, ml_config, features_config)
```

Or train & save the model artifacts to a local directory:
```
# model_v1.json and model_v1.pt saved to /trained_models
dir = "/trained_models/"
trainer.train(df, ml_config, features_config, dir)
```

## Serving a model

To serve the model in a Python service use the `BanditPredictor` object:
```
from banditml.serving.predictor import BanditPredictor
```
Pass in the paths to the model artifacts:
```
config_path = "/some/model/path/model.json"
model_path = "/some/model/path/model.pt"

predictor = BanditPredictor.predictor_from_file(config_path, model_path)
print(predictor.predict({"country": "usa", "year": 1990}))

# Out:
# {'scores': [[170.55], [146.37]], 'ids': ["male", "female"]}
```
