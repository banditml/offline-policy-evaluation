"""
Script to fill bigquery table with data simulated for
the following ecommerce problem:

A user clicks on their shopping cart with the intention of checking out.
Given what products are in their cart, what product(s) should you
recommend to them?

The goal is to optimize the incremental $'s generated
during checkout from recommended items.

Usage:
    python scripts/write_test_dataset_to_bq.py
"""

from collections import defaultdict
import json
import random
import sys
import uuid

from google.cloud import bigquery
from google.oauth2 import service_account


# Bigquery configs
PROJECT = "gradient-decision"
DECISION_TABLE_NAME = "gradient_app_staging.decisions"
REWARDS_TABLE_NAME = "gradient_app_staging.rewards"
CREDS_PATH = "credentials/bq_creds.json"

EXPERIMENT_ID = "test-experiment-1"

# Define the products in the product catalog
PRODUCT_ID_MAPPING = {
    "womens-winter-jacket": {
        "id": 1,
        "price": 149.99,
        "descriptions": {"gender": "womens", "season": "winter"},
    },
    "mens-winter-jacket": {
        "id": 2,
        "price": 189.99,
        "descriptions": {"gender": "mens", "season": "winter"},
    },
    "womens-jeans": {
        "id": 3,
        "price": 59.99,
        "descriptions": {"gender": "womens", "season": "all-weather"},
    },
    "mens-jeans": {
        "id": 4,
        "price": 59.99,
        "descriptions": {"gender": "mens", "season": "all-weather"},
    },
    "womens-t-shirt": {
        "id": 5,
        "price": 14.99,
        "descriptions": {"gender": "womens", "season": "summer"},
    },
    "mens-t-shirt": {
        "id": 6,
        "price": 14.99,
        "descriptions": {"gender": "mens", "season": "summer"},
    },
    "womens-shorts": {
        "id": 7,
        "price": 29.99,
        "descriptions": {"gender": "womens", "season": "summer"},
    },
    "mens-shorts": {
        "id": 8,
        "price": 39.99,
        "descriptions": {"gender": "mens", "season": "summer"},
    },
    "scarf": {
        "id": 9,
        "price": 19.99,
        "descriptions": {"gender": "unisex", "season": "winter"},
    },
    "unisex-tank-top": {
        "id": 10,
        "price": 12.99,
        "descriptions": {"gender": "unisex", "season": "summer"},
    },
    "beanie": {
        "id": 11,
        "price": 19.99,
        "descriptions": {"gender": "unisex", "season": "winter"},
    },
    "sunglasses": {
        "id": 12,
        "price": 39.99,
        "descriptions": {"gender": "unisex", "season": "summer"},
    },
}
MIN_ITEMS_IN_CART = 0
MAX_ITEMS_IN_CART = 3
NUM_GET_DECISION_CALLS = 10000
IP_ADDRESSES = [
    "192.241.227.82",
    "227.243.212.120",
    "68.104.78.107",
    "145.203.95.11",
    "5.151.83.83",
    "97.56.129.9",
    "173.95.52.102",
    "55.27.28.242",
    "188.147.211.234",
]


def get_random_cart_contents():
    return random.sample(
        PRODUCT_ID_MAPPING.keys(), random.randint(MIN_ITEMS_IN_CART, MAX_ITEMS_IN_CART)
    )


def get_purchase_probabilities(cart_contents):
    same_gender_prob_boost = 0.06
    same_season_prob_boost = 0.03
    close_enough_category_prob_boost = 0.01
    no_info_prob = 0.005

    purchase_probs = defaultdict(int)
    for product in cart_contents:
        product_gender = PRODUCT_ID_MAPPING[product]["descriptions"]["gender"]
        product_season = PRODUCT_ID_MAPPING[product]["descriptions"]["season"]
        for potential_product_rec, meta in PRODUCT_ID_MAPPING.items():
            # add gender based boosts:
            if product_gender == meta["descriptions"]["gender"]:
                purchase_probs[potential_product_rec] += same_gender_prob_boost
            elif meta["descriptions"]["gender"] == "unisex":
                purchase_probs[
                    potential_product_rec
                ] += close_enough_category_prob_boost
            else:
                purchase_probs[potential_product_rec] += 0

            # add season based boosts:
            if product_season == meta["descriptions"]["season"]:
                purchase_probs[potential_product_rec] += same_season_prob_boost
            elif meta["descriptions"]["season"] == "all-weather":
                purchase_probs[
                    potential_product_rec
                ] += close_enough_category_prob_boost
            else:
                purchase_probs[potential_product_rec] += 0

    # if the user had nothing in their cart
    if len(purchase_probs) == 0:
        for potential_product_rec, meta in PRODUCT_ID_MAPPING.items():
            purchase_probs[potential_product_rec] = no_info_prob

    return purchase_probs


def boost_purchase_probs(purchase_probabilities, boost, keys=None):
    for k, v in purchase_probabilities.items():
        if keys is not None:
            if k in keys:
                purchase_probabilities[k] += boost
        else:
            purchase_probabilities[k] += boost


def main():
    # intialize bigquery client
    credentials = service_account.Credentials.from_service_account_file(CREDS_PATH)
    client = bigquery.Client(project=PROJECT, credentials=credentials)
    decision_table = client.get_table(DECISION_TABLE_NAME)
    reward_table = client.get_table(REWARDS_TABLE_NAME)

    # create decisions to insert into table
    decisions_to_insert, rewards_to_insert = [], []
    for i in range(NUM_GET_DECISION_CALLS):

        products_in_cart = get_random_cart_contents()
        purchase_probs = get_purchase_probabilities(products_in_cart)

        # update the purchase probabilities based on "trafficSource"
        # let's assume 1 = organic, 2 = facebook, 3 = google
        traffic_source = random.randint(1, 3)  # categorical feature

        # add a boost for organic traffic
        organic_traffic_boost = 0.01
        if traffic_source == 1:
            boost_purchase_probs(purchase_probs, organic_traffic_boost)

        # perhaps some traffic sources make certain items more likely
        # to be purchased. Boost sunglasses, beanies, & scarfs for facebook traffic
        facebook_traffic_boost = 0.02
        if traffic_source == 2:
            boost_purchase_probs(
                purchase_probs,
                facebook_traffic_boost,
                keys=["sunglasses", "beanie", "scarf"],
            )

        decision_id = str(uuid.uuid4())

        total_cart_value = 0
        for product in products_in_cart:
            total_cart_value += round(PRODUCT_ID_MAPPING[product]["price"], 2)

        context = {
            "ipAddress": random.choice(IP_ADDRESSES),  # auto-pulled feature
            "productsInCart": [
                PRODUCT_ID_MAPPING[i]["id"] for i in products_in_cart
            ],  # idlist feature
            "totalCartValue": total_cart_value,  # number feature
            "trafficSource": traffic_source,  # categorical feature
        }

        # decision - let's assume that 60% of the time the ecommerce company
        # picked the best product (based on purchase probability).
        # the other 40% of the time was random.
        for product_name in products_in_cart:
            del purchase_probs[product_name]

        coin_flip = random.random()
        if coin_flip < 0.60:
            # recommend the best item
            decision_str = max(purchase_probs, key=purchase_probs.get)
        else:
            decision_str = random.choice(list(purchase_probs.keys()))

        decision = PRODUCT_ID_MAPPING[decision_str]["id"]

        # remove totalCartValue field randomly in 1% of logs
        if random.random() < 0.01:
            del context["productsInCart"]

        # remove productsInCart field randomly in 1% of logs
        if random.random() < 0.01:
            del context["totalCartValue"]

        # remove trafficSource field randomly in 1% of logs
        if random.random() < 0.01:
            del context["trafficSource"]

        decisions_to_insert.append(
            {
                "decision_id": decision_id,
                "context": json.dumps(context),
                "decision": decision,
                "experiment_id": EXPERIMENT_ID,
            }
        )

        # now log reward if the user buys the item which we use the
        # computed "true" probabilities for
        draw = random.random()
        if draw <= purchase_probs[decision_str]:
            rewards_to_insert.append(
                {
                    "decision_id": decision_id,
                    "metrics": json.dumps(
                        {
                            "purchase": 1,
                            "addOn$Spent": PRODUCT_ID_MAPPING[decision_str]["price"],
                        }
                    ),
                }
            )

    # insert decisions into BQ table
    client.insert_rows(decision_table, decisions_to_insert)
    print(f"Successfully inserted {NUM_GET_DECISION_CALLS} decisions.")

    # insert rewards into BQ table
    client.insert_rows(reward_table, rewards_to_insert)
    print(f"Successfully inserted {len(rewards_to_insert)} rewards.")


if __name__ == "__main__":
    main()
