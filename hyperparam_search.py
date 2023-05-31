import optuna
import torch

import numpy as np

from time import time
from torch_geometric.datasets import TUDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from graph_esn.gesn import GroupedDeepGESN
from auto_esn.esn.reservoir.activation import self_normalizing_default, tanh
from graph_esn.readout.aggregator import sum_vertex_features, mean_vertex_features
from auto_esn.esn.reservoir.initialization import WeightInitializer, default_hidden

# torch.set_num_threads(1)

dataset = TUDataset(root='data', name='Mutagenicity')
train_data, test_data, y_train, y_test = train_test_split(dataset, dataset.y, test_size=0.2, stratify=dataset.y,
                                                          random_state=42)

y_true = y_test.numpy()

input_size = dataset.num_features


def objective(trial):
    activation_params = {}

    hidden_size = trial.suggest_int("hidden_size", 5, 500)
    hidden_density = trial.suggest_float("hidden_density", 1e-3, 0.3)
    activation = trial.suggest_categorical("activation", ["tanh", "sna"])
    aggregator = trial.suggest_categorical("aggregator", ["mean", "sum"])
    num_layers = trial.suggest_int("num_layers", 1, 6)
    num_groups = trial.suggest_int("num_groups", 1, 6)
    spectral_radius = trial.suggest_float("spectral_radius", 0.0, 1.0)
    head = trial.suggest_categorical("head", ["linear", "rf"])
    activation_params['leaky_rate'] = trial.suggest_float("leaky_rate", 0.0, 1.0)


    if activation == 'sna':
        activation_params["spectral_radius"] = trial.suggest_float("sna_sprectral_radius", 1.0, 5000)

    act_fn = {
        "tanh": tanh,
        "sna": self_normalizing_default
    }[activation](**activation_params)

    agg_fn = {
        "mean": mean_vertex_features,
        "sum": sum_vertex_features
    }[aggregator]

    head_clf = {
        "linear": LogisticRegression(max_iter=10_000),
        "rf": RandomForestClassifier(n_estimators=500, n_jobs=-1),
    }[head]

    times = []
    scores = []

    # for _ in range(5):
    model = GroupedDeepGESN(
        input_size=input_size, hidden_size=hidden_size,
        activation=act_fn, aggregator=agg_fn,
        num_layers=num_layers, groups=num_groups,
        initializer=WeightInitializer(
            weight_hh_init=default_hidden(spectral_radius=spectral_radius, density=hidden_density)
        ),
        conv_th=1e-5,
        readout=head_clf
    )
    with torch.no_grad():
        start = time()
        model.fit(train_data, y_train)
        y_pred = model(test_data)
        end = time()
    times.append(end - start)
    scores.append(accuracy_score(y_true, y_pred))

    # trial.set_user_attr('Var-acc', np.var(scores))
    # trial.set_user_attr('Var-time', np.var(times))
    trial.set_user_attr('Mean-acc', np.mean(scores))
    trial.set_user_attr('Mean-time', np.mean(times))

    return [
        np.mean(scores),
        np.mean(times)
    ]


study = optuna.create_study(
    storage="sqlite:///db/runs.sqlite3",
    study_name="reservoir-gnn-mutagenicity-single-run",
    directions=["maximize", "minimize"],
    pruner=optuna.pruners.MedianPruner(
        n_startup_trials=10, n_warmup_steps=50, interval_steps=10
    )
)
# study.set_metric_names(["Mean accuracy", "Mean time"])
study.optimize(objective, n_trials=1_000, n_jobs=1)
