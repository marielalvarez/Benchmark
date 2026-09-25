"""Canales despolarizante, error de lectura, relajación térmica, y las 7
condiciones de ruido de E2 (ninguna + 3 aisladas + 3 compuestas), sobre
`qml.NoiseModel`/`qml.noise.add_noise` -- ver `channels.py`. Aplicable solo
a QCNNs (no a las CNN análogas, que no tienen circuito que ruidosear).

`make_noisy_predict_proba` es el punto de entrada para E2: reconstruye el
circuito de un modelo (hur/cong/gong vía `execution.registry`) sobre
`default.mixed` (el device analítico `default.qubit` de los modelos no
soporta canales de ruido -- ver `channels.py`), aplica opcionalmente
transpilación a un mapa de acoplamiento (`execution.transpile`) y la
condición de ruido pedida, y finalmente fija `n_shots` -- en ese orden,
porque el ruido debe insertarse en el circuito YA transpilado (con los
SWAPs de más) y los shots se miden sobre el circuito ya ruidoso.

Wei et al. se maneja aparte (`make_wei_noisy_predict_proba`): su lectura
es un `qml.expval(qml.Hamiltonian(...))`, que `qml.transforms.transpile`
no soporta (ver `execution/transpile.py`) -- así que para Wei, E2 aplica
shots + ruido pero no transpilación. Además, igual que en
`execution.shots.make_wei_predict_proba`, bajo shots/ruido solo se
entrena el Hamiltoniano de lectura (`h`, 37 parámetros) -- el filtro LCU
(`beta`) queda fijo (ver esa docstring para la limitación real de
PennyLane 0.45 que obliga a esto: ningún método de diferenciación
compatible con shots puede perturbar el `StatePrep` de un vector que
depende de `beta` sin romper su validación de norma).
"""

import pennylane as qml

from qcnn_benchmark.execution.registry import get_circuit_spec
from qcnn_benchmark.execution.shots import WEI_FROZEN_BETA_SEED, n_trainable_params
from qcnn_benchmark.execution.transpile import transpile_qnode

from .channels import (
    GATE_TIME_US,
    NOISE_CONDITIONS,
    SEVERITY_PARAMS,
    composite_noise_model,
    depolarizing_noise_model,
    get_noise_model,
    readout_noise_model,
    relaxation_noise_model,
)

__all__ = [
    "NOISE_CONDITIONS",
    "SEVERITY_PARAMS",
    "GATE_TIME_US",
    "depolarizing_noise_model",
    "readout_noise_model",
    "relaxation_noise_model",
    "composite_noise_model",
    "get_noise_model",
    "make_noisy_predict_proba",
    "make_wei_noisy_predict_proba",
    "n_trainable_params",
]


def make_noisy_predict_proba(model_name, n_shots, condition, coupling_map=None):
    """`predict_proba(params, x)` de un modelo del registro genérico
    (hur, cong, gong) bajo `n_shots` finitos + la condición de ruido
    `condition` (una de `NOISE_CONDITIONS`) + transpilación opcional a
    `coupling_map`."""
    spec = get_circuit_spec(model_name)
    mixed_dev = qml.device("default.mixed", wires=spec.n_wires)
    qnode = qml.QNode(spec.get_qnode().func, mixed_dev)

    if coupling_map is not None:
        qnode = transpile_qnode(qnode, coupling_map=coupling_map)

    noise_model = get_noise_model(condition)
    if noise_model is not None:
        qnode = qml.noise.add_noise(qnode, noise_model=noise_model)

    qnode = qml.set_shots(qnode, shots=n_shots)

    def predict_proba(params, x):
        return spec.extract(spec.call(qnode, params, x))

    return predict_proba


def make_wei_noisy_predict_proba(n_shots, condition):
    """Igual que `make_noisy_predict_proba` pero para Wei et al. -- sin
    parámetro `coupling_map` (ver docstring del módulo) y entrenando solo
    `h` (37 parámetros; `beta` fijo, ver docstring del módulo y de
    `execution.shots.make_wei_predict_proba`)."""
    import autograd.numpy as anp
    import numpy as np

    from qcnn_benchmark.models import qcnn_wei

    frozen_beta = np.random.default_rng(WEI_FROZEN_BETA_SEED).normal(0, 0.1, qcnn_wei.N_FILTER_PARAMS)
    mixed_dev = qml.device("default.mixed", wires=qcnn_wei.N_WORK_QUBITS)
    qnode = qml.QNode(qcnn_wei._readout_circuit.func, mixed_dev)

    noise_model = get_noise_model(condition)
    if noise_model is not None:
        qnode = qml.noise.add_noise(qnode, noise_model=noise_model)
    qnode = qml.set_shots(qnode, shots=n_shots)

    def predict_proba(h, x):
        g = qcnn_wei.conv_lcu_state(x, frozen_beta)
        logit = qnode(g, h)
        return 1.0 / (1.0 + anp.exp(-logit))

    return predict_proba
