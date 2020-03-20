import logging
import math
from typing import Dict, List

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


logger = logging.getLogger(__name__)


def gaussian_fill_w_gain(tensor, activation, dim_in, min_std=0.0) -> None:
    """ Gaussian initialization with gain."""
    gain = math.sqrt(2) if activation == "relu" else 1
    init.normal_(tensor, mean=0, std=max(gain * math.sqrt(1 / dim_in), min_std))


class EmbedDnn(nn.Module):
    def __init__(
        self,
        layers,
        activations,
        use_batch_norm=False,
        min_std=0.0,
        dropout_ratio=0.0,
        use_layer_norm=False,
        feature_specs={},
        product_sets={},
        float_feature_order=[],
        id_feature_order=[],
    ) -> None:
        super().__init__()
        self.layers: nn.ModuleList = nn.ModuleList()
        self.batch_norm_ops: nn.ModuleList = nn.ModuleList()
        self.activations = activations
        self.use_batch_norm = use_batch_norm
        self.dropout_layers: nn.ModuleList = nn.ModuleList()
        self.use_dropout = dropout_ratio > 0.0
        self.layer_norm_ops: nn.ModuleList = nn.ModuleList()
        self.use_layer_norm = use_layer_norm
        self.feature_specs = feature_specs
        self.float_feature_order = float_feature_order
        self.id_feature_order = id_feature_order
        self.embeddings = nn.ModuleList()
        self.embeddings_idx_map = {}

        assert len(layers) >= 2, "Invalid layer schema {} for network".format(layers)

        # handle features that require embeddings
        for feature_name in id_feature_order:
            meta = feature_specs[feature_name]
            product_set_id = meta["product_set_id"]
            num_ids = len(product_sets[product_set_id]["ids"])
            # embedding size rule of thumb written by google:
            # https://developers.googleblog.com/2017/11/introducing-tensorflow-feature-columns.html
            embedding_dim = math.ceil(num_ids ** 0.25)
            layers[0] += embedding_dim

            if product_set_id not in self.embeddings_idx_map:
                # add the +1 to num_ids as we need a 0 vector for the padding index
                self.embeddings_idx_map[product_set_id] = len(self.embeddings)
                self.embeddings.append(
                    nn.Embedding(num_ids + 1, embedding_dim, padding_idx=0)
                )

        for i, layer in enumerate(layers[1:]):
            self.layers.append(nn.Linear(layers[i], layer))
            if self.use_batch_norm:
                self.batch_norm_ops.append(nn.BatchNorm1d(layers[i]))
            if self.use_layer_norm and i < len(layers) - 2:
                # LayerNorm is applied to the output of linear
                self.layer_norm_ops.append(nn.LayerNorm(layer))  # type: ignore
            if self.use_dropout and i < len(layers[1:]) - 1:
                # applying dropout to all layers except
                # the input and the last output layer
                self.dropout_layers.append(nn.Dropout(p=dropout_ratio))
            gaussian_fill_w_gain(
                self.layers[i].weight, self.activations[i], layers[i], min_std
            )
            init.constant_(self.layers[i].bias, 0)

    def forward(
        self,
        X_float: torch.FloatTensor,
        X_id_list: torch.LongTensor = None,
        X_id_list_idxs: torch.LongTensor = None,
    ) -> torch.FloatTensor:
        """ Forward pass for generic feed-forward DNNs. Assumes activation names
        are valid pytorch activation names.
        """

        id_feature_col_idx = 0
        # concat embedded feature to the float feature tensor
        for feature_name in self.id_feature_order:
            embedding_table_id = self.feature_specs[feature_name]["product_set_id"]
            embedding_table = self.embeddings[
                self.embeddings_idx_map[embedding_table_id]
            ]
            col_start = X_id_list_idxs[0][id_feature_col_idx]
            col_end = X_id_list_idxs[0][id_feature_col_idx + 1]
            embeddings = embedding_table(X_id_list[:, col_start:col_end])

            # https://datascience.stackexchange.com/questions/44635/does-sum-of-embeddings-make-sense
            # average the embeddings, but first drop the padded 0 embeddings
            # using a shape that maintains the tensor dimensions
            valid_row_mask = (embeddings.sum(dim=(2), keepdim=True) != 0).float()
            denom = valid_row_mask.sum(dim=(1, 2)).unsqueeze(dim=1)
            emedding_sum = embeddings.sum(dim=1)
            avg_embeddings = emedding_sum / denom
            # set Nan values to zero (case where there is an empty id list)
            avg_embeddings[avg_embeddings != avg_embeddings] = 0

            # concatenate the float tensor and embedded tensor
            X_float = torch.cat((X_float, avg_embeddings), dim=1)
            id_feature_col_idx += 2

        x = X_float
        for i, activation in enumerate(self.activations):
            if self.use_batch_norm:
                x = self.batch_norm_ops[i](x)
            x = self.layers[i](x)
            if self.use_layer_norm and i < len(self.layer_norm_ops):
                x = self.layer_norm_ops[i](x)
            if activation == "linear":
                pass
            elif activation == "tanh":
                x = torch.tanh(x)
            else:
                x = getattr(F, activation)(x)
            if self.use_dropout and i < len(self.dropout_layers):
                x = self.dropout_layers[i](x)
        return x
