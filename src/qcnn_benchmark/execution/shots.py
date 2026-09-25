"""Motor de shots finitos: reconstruye, para cada modelo QCNN de este
benchmark, una versión de su `predict_proba` que mide por conteo de shots
(p.ej. 256/1024/4096) en vez de expectation value analítico exacto -- sin
modificar los adaptadores de modelo (`models/qcnn_*.py`), que siguen siendo
la fuente de verdad del régimen E0 (analítico, sin ruido).

Primitiva: `qml.set_shots(qnode, shots=n)` devuelve un qnode NUEVO que
ejecuta el mismo circuito (misma función, sin decorar, guardada por
PennyLane) pero estimando cada medición por muestreo en vez de por álgebra
lineal exacta -- no muta el qnode original, así que `models/qcnn_*.py`
sigue sirviendo intacto al régimen E0 en paralelo. PennyLane detecta
automáticamente que un circuito con `shots` fijo no admite
`diff_method="backprop"` y usa la regla de desplazamiento de parámetro
(parameter-shift) en su lugar -- el mismo mecanismo con el que se
calcularía el gradiente en hardware real. Por eso el ruido de shots que
introduce este módulo también afecta al *gradiente* durante el
entrenamiento (`qcnn_benchmark.training.train_binary_classifier` no
necesita cambiar: solo se le pasa un `predict_proba_fn` distinto), no solo
a la exactitud de prueba evaluada al final.

Pendiente de fidelidad física completa: esto modela el ruido de MUESTREO
finito (varianza estadística de estimar una probabilidad/expectation con N
shots) -- no ruido de hardware (relajación, despolarización, error de
lectura); para eso está `qcnn_benchmark.noise`, que se combina con este
módulo en E2.
"""

import numpy as np
import autograd.numpy as anp
import pennylane as qml

from .registry import get_circuit_spec

# Semilla fija con la que se congela `beta` (filtro LCU) para Wei bajo
# shots/ruido -- ver docstring de `make_wei_predict_proba` sobre por qué.
# Se expone como constante para que noise.make_wei_noisy_predict_proba use
# EXACTAMENTE el mismo beta (si usaran semillas distintas, un mismo vector
# de parámetros `h` significaría un modelo distinto en E1 vs. E2).
WEI_FROZEN_BETA_SEED = 0


def make_generic_predict_proba(model_name, n_shots):
    """`predict_proba(params, x)` de cualquier modelo del registro
    genérico (`execution.registry.CIRCUIT_SPECS`: hur, cong, gong) bajo
    `n_shots` finitos."""
    spec = get_circuit_spec(model_name)
    shots_qnode = qml.set_shots(spec.get_qnode(), shots=n_shots)

    def predict_proba(params, x):
        return spec.extract(spec.call(shots_qnode, params, x))

    return predict_proba


def make_wei_predict_proba(n_shots):
    """QCNN de Wei et al. con `n_shots` -- **solo entrena el Hamiltoniano
    de lectura (`h`, 37 parámetros)**; el filtro LCU (`beta`, 9 parámetros)
    queda FIJO en un valor aleatorio de referencia
    (`WEI_FROZEN_BETA_SEED`), no entrenable, bajo shots.

    Limitación real de PennyLane 0.45, no de este proyecto: la lectura de
    Wei usa `qml.StatePrep(g, wires)` para embeber `g = conv_lcu_state(x,
    beta)` (que depende de `beta`, un parámetro entrenable). Bajo shots,
    PennyLane necesita un método de diferenciación compatible
    (parameter-shift o SPSA), y AMBOS perturban el vector de estado de
    entrada de `StatePrep` -- que `StatePrep.compute_decomposition` pasa a
    `MottonenStatePreparation(state, wires)` sin normalizar, y
    `MottonenStatePreparation.__init__` no acepta ningún flag para relajar
    su validación estricta de norma unitaria. Confirmado experimentalmente
    con ambos diff_methods: la perturbación rompe la norma (o produce NaN)
    y el circuito lanza `ValueError: state_vector has to be of norm 1.0`.
    No hay combinación de flags que lo evite en esta versión de PennyLane.

    Congelar `beta` (que nunca pasa por `StatePrep` como argumento
    diferenciable -- se sustituye por un valor de punto fijo antes de
    llamar al qnode) es la salida honesta: E1/E2 quedan funcionales para
    Wei a costa de entrenar 37 parámetros en vez de 46. Documentado
    también en los notebooks que usan esto (E1_finite_shots.ipynb,
    E2_nisq_noise.ipynb) -- no es una reducción silenciosa."""
    from qcnn_benchmark.models import qcnn_wei

    frozen_beta = np.random.default_rng(WEI_FROZEN_BETA_SEED).normal(0, 0.1, qcnn_wei.N_FILTER_PARAMS)
    shots_qnode = qml.set_shots(qcnn_wei._readout_circuit, shots=n_shots)

    def predict_proba(h, x):
        g = qcnn_wei.conv_lcu_state(x, frozen_beta)
        logit = shots_qnode(g, h)
        return 1.0 / (1.0 + anp.exp(-logit))

    return predict_proba


PREDICT_PROBA_FACTORIES = {
    "hur": lambda n_shots: make_generic_predict_proba("hur", n_shots),
    "cong": lambda n_shots: make_generic_predict_proba("cong", n_shots),
    "gong": lambda n_shots: make_generic_predict_proba("gong", n_shots),
    "wei": make_wei_predict_proba,
}

# Número de parámetros entrenables de `predict_proba_fn` bajo shots -- para
# hur/cong/gong es el mismo que en E0 (TOTAL_PARAMS de cada módulo); para
# wei son solo los 37 del Hamiltoniano (ver make_wei_predict_proba). Los
# notebooks de E1/E2 usan esto en vez de `modulo.TOTAL_PARAMS` directamente.


def n_trainable_params(model_name):
    """Número de parámetros que espera el `predict_proba` que devuelve
    `make_shots_predict_proba(model_name, ...)` -- distinto de
    `modulo.TOTAL_PARAMS` solo para "wei" (ver `make_wei_predict_proba`)."""
    if model_name == "wei":
        from qcnn_benchmark.models import qcnn_wei

        return qcnn_wei.N_HAMILTONIAN_PARAMS
    from qcnn_benchmark.models import qcnn_cong, qcnn_gong, qcnn_hur

    return {"hur": qcnn_hur.TOTAL_PARAMS, "cong": qcnn_cong.TOTAL_PARAMS, "gong": qcnn_gong.TOTAL_PARAMS}[model_name]


def make_shots_predict_proba(model_name, n_shots):
    """`predict_proba(params, x)` del modelo `model_name` bajo `n_shots`
    finitos -- misma firma que `qcnn_benchmark.models.qcnn_*.predict_proba`,
    para poder pasarla sin cambios a `train_binary_classifier` (usar
    `n_trainable_params(model_name)` como `n_params`, NO
    `modulo.TOTAL_PARAMS` directamente -- ver nota sobre "wei" arriba)."""
    return PREDICT_PROBA_FACTORIES[model_name](n_shots)
