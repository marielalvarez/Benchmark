"""Métricas de convergencia, calculadas retroactivamente sobre los CSVs de
resultados ya guardados (`n_updates_run`, `stopped_early_at`, `best_val_loss`
por corrida) -- sin reentrenar nada.

Limitación conocida (ver docstring de `qcnn_benchmark.metrics`):
`train_binary_classifier` sí calcula `train_loss_history`/`val_loss_history`
por actualización, pero esos arreglos nunca se persistieron a CSV, solo el
resumen escalar por corrida. Este módulo agrega esos tres escalares por
semilla (media +/- IC bootstrap BCa), no reconstruye la forma de la curva.
"""

import numpy as np
import pandas as pd

from qcnn_benchmark.stats import bootstrap_bca_ci


def _ci_or_degenerate(values, rng):
    """`bootstrap_bca_ci`, salvo cuando las 5 semillas dan el mismo valor
    exacto (frecuente en `n_updates_run` cuando ninguna corrida activa early
    stopping: las 5 llegan al tope de actualizaciones del protocolo) -- ahí
    el BCa de scipy es degenerado (devuelve NaN con un warning), así que se
    reporta el IC como el propio valor en vez de propagar el NaN."""
    if np.std(values, ddof=1) == 0:
        return float(values[0]), float(values[0])
    return bootstrap_bca_ci(values, rng=rng)


def early_stopped(df):
    """Máscara booleana: qué corridas de `df` activaron early stopping
    (`stopped_early_at` no nulo)."""
    return df["stopped_early_at"].notna()


def early_stopping_rate(df):
    """Fracción de corridas de `df` que activó early stopping."""
    return float(early_stopped(df).mean())


def convergence_summary(df, group_cols=("dataset", "model"), rng=None):
    """Por cada grupo de `df` (un CSV crudo de resultados ya cargado con
    pandas, agrupado por `group_cols` -- p.ej. dataset+model, o
    dataset+model+n_shots en E1): media +/- IC bootstrap BCa sobre las
    semillas de `n_updates_run` y `best_val_loss`, y la fracción de corridas
    que activó early stopping."""
    rng = np.random.default_rng(0) if rng is None else rng
    group_cols = list(group_cols)
    rows = []
    for keys, group in df.groupby(group_cols):
        keys = keys if isinstance(keys, tuple) else (keys,)
        n_upd = group["n_updates_run"].to_numpy(dtype=float)
        best_vl = group["best_val_loss"].to_numpy(dtype=float)
        n_upd_lo, n_upd_hi = _ci_or_degenerate(n_upd, rng)
        vl_lo, vl_hi = _ci_or_degenerate(best_vl, rng)
        rows.append(
            {
                **dict(zip(group_cols, keys)),
                "n_seeds": len(group),
                "n_updates_run_mean": float(np.mean(n_upd)),
                "n_updates_run_ci_lo": n_upd_lo,
                "n_updates_run_ci_hi": n_upd_hi,
                "early_stopping_rate": early_stopping_rate(group),
                "best_val_loss_mean": float(np.mean(best_vl)),
                "best_val_loss_ci_lo": vl_lo,
                "best_val_loss_ci_hi": vl_hi,
            }
        )
    return pd.DataFrame(rows)
