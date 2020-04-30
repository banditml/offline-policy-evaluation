from skorch import NeuralNetClassifier, NeuralNetRegressor
import torch


def fit_custom_pytorch_module_w_skorch(reward_type, module, X, y, hyperparams):
    """Fit a custom PyTorch module using Skorch."""

    if reward_type == "regression":
        skorch_func = NeuralNetRegressor
    else:
        skorch_func = NeuralNetClassifier
        # torch's nll_loss wants 1-dim tensors & long type tensors.
        y = y.long().squeeze()

    skorch_net = skorch_func(
        module=module,
        optimizer=torch.optim.Adam,
        lr=hyperparams["learning_rate"],
        optimizer__weight_decay=hyperparams["l2_decay"],
        max_epochs=hyperparams["max_epochs"],
        batch_size=hyperparams["batch_size"],
        iterator_train__shuffle=True,
    )

    skorch_net.fit(X, y)
    return skorch_net
