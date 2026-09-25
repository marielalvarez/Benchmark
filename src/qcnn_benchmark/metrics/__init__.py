"""Métricas del benchmark, organizadas en las 4 familias del diseño
experimental original (`context/diseño_experimentos (2).pdf`, diapositiva
"Familias de métricas"): predictivas, convergencia, de recursos, robustez.

Estado real de cada familia (no las 4 llegaron a implementarse a la vez,
a diferencia de `execution/` y `noise/` que sí se construyeron completas
desde el inicio):

- **Predictivas** (`classification.py`) -- DEFINIDA, backfill PENDIENTE.
  `batch_accuracy` (desde el inicio) más `confusion_counts` y las funciones
  `*_from_counts` (precision, recall, F1, balanced accuracy, macro-F1),
  agregadas ahora. Son mandatorias, no una preferencia de este framework:
  ver la sección "Predictive Evaluation Metrics" de
  `qcnn_benchmark_article_draft.tex` (repo NGS, fuera de este proyecto,
  `../qcnn_benchmark_article_draft.tex` relativo a la raíz de este repo) --
  "Accuracy, balanced accuracy, macro-averaged F1-score, per-class recall,
  confusion matrices, and predictive loss are mandatory."
  Aplicación **forward-only**: E0/E0A/E0_quantum_advantage/E0B/E1 (150
  corridas ya ejecutadas) NO se han backfilleado -- sus CSVs solo tienen
  `test_acc`, no tp/fp/fn/tn. Backfillear esas 150 corridas exige
  reentrenar (determinista dado el seed, pero cuesta cómputo de nuevo) y
  queda agendado para la Fase III del plan de trabajo (revisión de drafts,
  oct-nov 2026), no para el Experiment Freeze del 7-oct-2026. E2 y
  cualquier corrida nueva de una eventual expansión v1.1 (ver
  `resources.py` y la nota de costo de E1 bajo shots para Hur/Gong) sí
  deben guardar `confusion_counts` desde su primera ejecución.

- **Convergencia** (`convergence.py`) -- IMPLEMENTADA retroactivamente,
  sin reentrenar, a partir de tres escalares que sí se guardaron por corrida
  (`n_updates_run`, `stopped_early_at`, `best_val_loss`).
  Limitación conocida: `training.train_binary_classifier` sí calcula las
  historias completas de pérdida (`train_loss_history`/`val_loss_history`)
  por actualización, pero esos arreglos nunca se persistieron a CSV -- solo
  el resumen escalar por corrida. Por eso este módulo no puede reconstruir
  la forma de la curva de corridas ya terminadas, solo agregarla por semilla.

- **De recursos** (`resources.py`) -- IMPLEMENTADA. Qubits y parámetros
  (totales y efectivamente entrenables, que difieren para Wei bajo
  shots/ruido) leídos directamente de `models/qcnn_*.py` y
  `execution/registry.py`/`execution/shots.py`, no hardcodeados de nuevo
  aquí. Presupuesto de shots leído de los CSVs de E1.

- **Robustez** -- BLOQUEADA, no implementada. Mide degradación de
  desempeño y tasa de fallo bajo ruido, y depende de que exista la matriz
  de resultados de E2 (`E2_nisq_noise.ipynb`), que todavía no se ha
  ejecutado. No hay nada que calcular hasta entonces.
"""

from .classification import (
    balanced_accuracy_from_counts,
    batch_accuracy,
    confusion_counts,
    f1_from_counts,
    macro_f1_from_counts,
    precision_from_counts,
    predict_labels,
    recall_from_counts,
)
from .convergence import convergence_summary, early_stopping_rate
from .resources import n_wires, resource_summary, shots_budget_summary, total_params, trainable_params

__all__ = [
    "batch_accuracy",
    "predict_labels",
    "confusion_counts",
    "precision_from_counts",
    "recall_from_counts",
    "f1_from_counts",
    "balanced_accuracy_from_counts",
    "macro_f1_from_counts",
    "convergence_summary",
    "early_stopping_rate",
    "n_wires",
    "resource_summary",
    "shots_budget_summary",
    "total_params",
    "trainable_params",
]
