"""Métricas predictivas evaluadas post-hoc (fuera del ciclo de gradiente)
sobre un `predict_proba_fn(params, x)` ya entrenado.

`confusion_counts` es la única que corre contra el modelo (recorre `X`);
las demás (`*_from_counts`) son puramente aritméticas sobre los 4 enteros
tp/fp/fn/tn -- eso alcanza para accuracy, balanced accuracy, precision,
recall y F1 (macro y por clase) según la sección "Predictive Evaluation
Metrics" de `qcnn_benchmark_article_draft.tex` (repo NGS, fuera de este
proyecto) sin tener que guardar predicciones por muestra: basta con
persistir 4 enteros por corrida en vez de un vector del tamaño del
conjunto de prueba."""

import numpy as np


def predict_labels(predict_proba_fn, params, X, threshold=0.5):
    return np.array([1 if float(predict_proba_fn(params, x)) >= threshold else 0 for x in X])


def batch_accuracy(predict_proba_fn, params, X, y):
    y_pred = predict_labels(predict_proba_fn, params, X)
    return float(np.mean(y_pred == np.asarray(y).astype(int)))


def confusion_counts(predict_proba_fn, params, X, y, threshold=0.5):
    """tp/fp/fn/tn de un `predict_proba_fn` entrenado sobre (X, y) -- clase
    positiva = etiqueta 1. Suficiente para derivar accuracy, balanced
    accuracy, precision, recall y F1 (macro y por clase) sin guardar
    predicciones por muestra, ver `*_from_counts` abajo."""
    y_pred = predict_labels(predict_proba_fn, params, X, threshold)
    y_true = np.asarray(y).astype(int)
    return {
        "tp": int(np.sum((y_pred == 1) & (y_true == 1))),
        "fp": int(np.sum((y_pred == 1) & (y_true == 0))),
        "fn": int(np.sum((y_pred == 0) & (y_true == 1))),
        "tn": int(np.sum((y_pred == 0) & (y_true == 0))),
    }


def precision_from_counts(tp, fp):
    return float(tp / (tp + fp)) if (tp + fp) > 0 else float("nan")


def recall_from_counts(tp, fn):
    return float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")


def f1_from_counts(tp, fp, fn):
    p, r = precision_from_counts(tp, fp), recall_from_counts(tp, fn)
    if np.isnan(p) or np.isnan(r) or (p + r) == 0:
        return float("nan")
    return float(2 * p * r / (p + r))


def balanced_accuracy_from_counts(tp, fp, fn, tn):
    """Media de recall por clase (sensibilidad de la clase 1, especificidad
    de la clase 0) -- mandatorio junto con macro-F1 en la especificación
    (ver docstring del módulo) para no sobreestimar desempeño con clases
    desbalanceadas."""
    recall_pos = recall_from_counts(tp, fn)
    recall_neg = recall_from_counts(tn, fp)
    return float((recall_pos + recall_neg) / 2)


def macro_f1_from_counts(tp, fp, fn, tn):
    """F1 promediado sobre ambas clases (clase 0 tratada como positiva
    intercambiando tp<->tn y fp<->fn)."""
    f1_pos = f1_from_counts(tp, fp, fn)
    f1_neg = f1_from_counts(tn, fn, fp)
    return float((f1_pos + f1_neg) / 2)
