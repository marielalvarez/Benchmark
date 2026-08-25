"""Bootstrap BCa, permutación pareada, corrección de Holm, tamaños de
efecto (d_z, probabilidad de superioridad) -- el análisis estadístico que
E0A usa para comparar Hur/Wei/Gong entre sí sobre cada dataset.

Reutiliza `scipy.stats.bootstrap` (method="BCa") y `scipy.stats.
permutation_test` (permutation_type="samples", equivalente a una prueba de
permutación de signo sobre diferencias pareadas) en vez de reimplementarlos:
son las dos piezas de este módulo con más superficie para errores sutiles,
y scipy ya es una dependencia validada del proyecto. La corrección de Holm
sí se implementa a mano aquí -- es un algoritmo step-down de ~10 líneas,
bien definido, y no justifica una dependencia nueva (p.ej. statsmodels)
solo para esa función.
"""

import numpy as np
from scipy import stats as _scipy_stats


def bootstrap_bca_ci(data, statistic=np.mean, confidence_level=0.95, n_bootstrap=10_000, rng=None):
    """IC bootstrap BCa para `statistic(data)`. Devuelve (lo, hi)."""
    data = np.asarray(data)
    rng = np.random.default_rng() if rng is None else rng
    res = _scipy_stats.bootstrap(
        (data,),
        statistic,
        method="BCa",
        confidence_level=confidence_level,
        n_resamples=n_bootstrap,
        random_state=rng,
    )
    return float(res.confidence_interval.low), float(res.confidence_interval.high)


def paired_sign_permutation_test(x, y, n_permutations=10_000, rng=None):
    """Prueba de permutación pareada sobre x_i - y_i (H0: los pares son
    intercambiables, i.e. sin diferencia sistemática entre x e y).
    Devuelve (p_value, mean_diff_observado)."""
    x, y = np.asarray(x), np.asarray(y)
    rng = np.random.default_rng() if rng is None else rng

    def _mean_diff(a, b, axis=-1):
        return np.mean(a - b, axis=axis)

    res = _scipy_stats.permutation_test(
        (x, y),
        _mean_diff,
        permutation_type="samples",
        n_resamples=n_permutations,
        random_state=rng,
    )
    return float(res.pvalue), float(res.statistic)


def holm_correction(p_values, alpha=0.05):
    """Corrección de Holm (step-down) para comparaciones múltiples.

    Devuelve (p-valores ajustados, en el mismo orden que `p_values`;
    máscara booleana de rechazo a `alpha`).
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    sorted_p = p[order]

    adjusted_sorted = np.empty(m)
    running_max = 0.0
    for i in range(m):
        running_max = max(running_max, (m - i) * sorted_p[i])
        adjusted_sorted[i] = min(running_max, 1.0)

    adjusted = np.empty(m)
    adjusted[order] = adjusted_sorted
    return adjusted, adjusted <= alpha


def cohens_dz(x, y):
    """Tamaño de efecto pareado (Cohen's d_z) sobre las diferencias x - y."""
    diff = np.asarray(x) - np.asarray(y)
    return float(np.mean(diff) / np.std(diff, ddof=1))


def probability_of_superiority(x, y):
    """P(x_i > y_i) pareada, estimada empíricamente (los empates cuentan
    0.5, convención estándar del estadístico de Wilcoxon signed-rank)."""
    x, y = np.asarray(x), np.asarray(y)
    wins = np.mean(x > y) + 0.5 * np.mean(x == y)
    return float(wins)
