import json
import logging
import math

import requests

BANDIT_APP_CREDS_PATH = "credentials/bandit_app_creds.json"
BANDIT_APP_EXP_CONFIG_URL = "https://www.banditml.com/api/get_bandit_exp_config"


def read_config(config_path):
    with open(config_path) as json_file:
        config = json.load(json_file)
    return config


def get_logger(name: str):
    logging.basicConfig(
        format="[%(asctime)s %(levelname)-3s] %(message)s",
        level=logging.INFO,
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(name)


def fancy_print(text: str, color="blue", size=60):
    ansi_color = "\033[94m"
    if color == "green":
        ansi_color = "\033[95m"
    elif color == "blue":
        ansi_color = "\033[94m"
    else:
        raise Exception(f"Color {color} not supported")

    end_color = "\033[0m"
    str_len = len(text)
    padding = math.ceil((size - str_len) / 2)
    header_len = padding * 2 + str_len + 2
    border = "#" * header_len
    message = "#" * padding + " " + text + " " + "#" * padding
    print(f"{ansi_color}\n{border}\n{message}\n{border}\n{end_color}")


def get_experiment_config_from_bandit_app(experiment_id):
    with open(BANDIT_APP_CREDS_PATH) as json_file:
        creds = json.load(json_file)

    api_key = creds["api_key"]
    params = {"experiment_id": experiment_id}
    response = requests.get(
        url=BANDIT_APP_EXP_CONFIG_URL,
        params=params,
        headers={"Authorization": f"ApiKey {api_key}"},
    )

    if response.status_code != 200:
        raise Exception(
            "Get experiment config from bandit app failed with code: "
            f"{response.status_code}\n"
        )

    return response.json()
