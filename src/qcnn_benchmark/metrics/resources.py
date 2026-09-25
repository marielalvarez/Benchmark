"""Métricas de recursos: qubits y parámetros entrenables de cada modelo, y
presupuesto de shots evaluado en E1/E2 -- leídos directamente de
`models/qcnn_*.py`, `models/cnn_*_analoga.py` y `execution/`, nunca
hardcodeados de nuevo aquí, para que no se desincronicen si esos módulos
cambian.

Aplicable a QCNNs y a sus análogas clásicas por igual, salvo `n_wires`
(las análogas no tienen circuito -- ver docstring de `noise/__init__.py`
sobre la misma distinción)."""

import pandas as pd

from qcnn_benchmark.execution.registry import get_circuit_spec
from qcnn_benchmark.execution.shots import n_trainable_params
from qcnn_benchmark.models import (
    cnn_gong_analoga,
    cnn_hur_analoga,
    cnn_wei_analoga,
    qcnn_cong,
    qcnn_gong,
    qcnn_hur,
    qcnn_wei,
)

_QUANTUM_MODULES = {"hur": qcnn_hur, "cong": qcnn_cong, "gong": qcnn_gong, "wei": qcnn_wei}
_CLASSICAL_MODULES = {
    "cnn_hur_analoga": cnn_hur_analoga,
    "cnn_gong_analoga": cnn_gong_analoga,
    "cnn_wei_analoga": cnn_wei_analoga,
}
_ALL_MODULES = {**_QUANTUM_MODULES, **_CLASSICAL_MODULES}


def _module(model_name):
    if model_name not in _ALL_MODULES:
        raise KeyError(f"'{model_name}' no está en el registro de recursos ({sorted(_ALL_MODULES)}).")
    return _ALL_MODULES[model_name]


def n_wires(model_name):
    """Qubits simulados por `model_name`, o `None` si es una análoga
    clásica (sin circuito). Wei no está en el registro genérico de
    circuitos (`execution/registry.py`, ver su docstring), se reporta
    aparte desde `qcnn_wei.N_WORK_QUBITS`."""
    if model_name not in _QUANTUM_MODULES:
        return None
    if model_name == "wei":
        return qcnn_wei.N_WORK_QUBITS
    return get_circuit_spec(model_name).n_wires


def total_params(model_name):
    """Parámetros totales del modelo (`TOTAL_PARAMS` de su módulo)."""
    return _module(model_name).TOTAL_PARAMS


def trainable_params(model_name, regime="analytic"):
    """Parámetros efectivamente entrenados bajo `regime`.

    "analytic" (E0/E0A/E0B, sin shots): todos los `TOTAL_PARAMS`, incluido
    Wei completo (filtro LCU + Hamiltoniano de lectura).
    "shots" o "noise" (E1/E2): igual para todos salvo Wei, que congela el
    filtro LCU (`beta`) y entrena solo el Hamiltoniano de lectura -- ver
    `execution.shots.make_wei_predict_proba` sobre la limitación real de
    PennyLane que obliga a esto.
    """
    if regime == "analytic" or model_name not in _QUANTUM_MODULES:
        return total_params(model_name)
    if regime in ("shots", "noise"):
        return n_trainable_params(model_name)
    raise ValueError(f"regime desconocido: {regime!r} (usa 'analytic', 'shots' o 'noise')")


def resource_summary(model_names, regime="analytic"):
    """Tabla de recursos (qubits, parámetros totales, parámetros
    efectivamente entrenados bajo `regime`) para cada modelo en
    `model_names`."""
    rows = [
        {
            "model": name,
            "n_wires": n_wires(name),
            "total_params": total_params(name),
            "trainable_params": trainable_params(name, regime=regime),
        }
        for name in model_names
    ]
    return pd.DataFrame(rows)


def shots_budget_summary(df):
    """Presupuestos de shots (`n_shots`) evaluados en un CSV crudo de E1/E2
    ya cargado con pandas -- valores únicos, ordenados."""
    return sorted(df["n_shots"].unique().tolist())
