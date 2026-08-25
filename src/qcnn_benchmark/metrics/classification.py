"""Métricas predictivas evaluadas post-hoc (fuera del ciclo de gradiente)
sobre un `predict_proba_fn(params, x)` ya entrenado."""

import numpy as np


def predict_labels(predict_proba_fn, params, X, threshold=0.5):
    return np.array([1 if float(predict_proba_fn(params, x)) >= threshold else 0 for x in X])


def batch_accuracy(predict_proba_fn, params, X, y):
    y_pred = predict_labels(predict_proba_fn, params, X)
    return float(np.mean(y_pred == np.asarray(y).astype(int)))
