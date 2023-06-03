import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from numpy.random import default_rng
rng = default_rng(42)


def sample_from_M(delta, c, n=1000):
    # <answer>
    n0 = rng.binomial(n, 2*delta, 1).item()
    n1 = n - n0

    return np.concatenate(
        (
            rng.normal(size=n0),
            c + rng.exponential(size=n1)
        )
    )
    # </answer>


def sample_from_Xy(delta, c, n=1000):
    # <answer>
    pi0 = 0.5
    n0 = rng.binomial(n, pi0, 1).item()
    n1 = n - n0

    x0 = sample_from_M(delta, c, n=n0)
    y0 = np.zeros(n0, dtype=int)

    x1 = -sample_from_M(delta, c, n=n1)
    y1 = np.ones(n1, dtype=int)

    X = np.concatenate((x0, x1))[:, None]
    y = np.concatenate((y0, y1))

    return X, y
    # </answer>


def gen(deltas, cs):
    for delta, c in zip(deltas, cs):
        # Estimation de l'erreur du 1-NN pour un jeu de
        # données avec `delta` et `c`.

        # On entraine un 1-NN sur un jeu de donnée de taille 1000.
        # <answer>
        cls = KNeighborsClassifier(n_neighbors=1)
        X_train, y_train = sample_from_Xy(delta, c, n=1000)
        cls.fit(X_train, y_train)
        # </answer>

        # On estime son erreur avec un jeu de données de test de
        # taille 10000.
        # <answer>
        X_test, y_test = sample_from_Xy(delta, c, n=10000)
        pred = cls.predict(X_test)
        err = 1 - accuracy_score(y_test, pred)
        # </answer>

        # On génère `delta`, `c` et l'erreur du 1-NN
        yield delta, c, err
