"""Registro de circuitos por modelo QCNN: para cada uno, cómo obtener su
qnode "crudo" (device analítico por defecto, función sin decorar accesible
vía `.func`) y cómo llamarlo/leer su resultado -- la pieza que comparten
`execution.shots` (E1) y `noise` (E2) para reconstruir el circuito de cada
modelo sobre un device distinto (con shots, con ruido, transpilado) sin
duplicar esta lógica en cada uno ni tocar `models/qcnn_*.py`.

Wei et al. no encaja en este registro de la misma forma que los otros tres
(su parte cuántica es solo la lectura -- Hamiltoniano de 37 parámetros --
mientras el filtro LCU es preprocesamiento clásico fuera del qnode): se
maneja aparte en `execution.shots.make_wei_predict_proba` y
`noise.make_wei_noisy_predict_proba`, documentado en cada uno.
"""

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class CircuitSpec:
    """`get_qnode()` obtiene el qnode original (analítico, device propio del
    modelo) desde el que se puede leer `.func` (la función sin decorar) y
    `.device` (para saber cuántos wires tiene). `call(qnode, params, x)`
    invoca ese qnode (o uno reconstruido sobre otro device, mismo `.func`)
    con los kwargs propios del modelo. `extract(resultado_crudo)` saca la
    probabilidad de clase positiva de lo que devuelve el qnode."""

    get_qnode: Callable[[], object]
    n_wires: int
    call: Callable[[object, object, object], object]
    extract: Callable[[object], float]


def _hur_spec():
    from qcnn_benchmark.models import qcnn_hur

    return CircuitSpec(
        get_qnode=lambda: qcnn_hur._hur_circuit.QCNN,
        n_wires=8,
        call=lambda qnode, params, x: qnode(
            x, params, U="U_6", U_params=qcnn_hur.U_PARAMS, embedding_type="Angle-compact", cost_fn="cross_entropy"
        ),
        extract=lambda probs: probs[1],
    )


def _cong_spec():
    from qcnn_benchmark.models import qcnn_cong

    return CircuitSpec(
        get_qnode=lambda: qcnn_cong._hur_circuit.QCNN,
        n_wires=8,
        call=lambda qnode, params, x: qnode(
            x, params, U="U_SU4", U_params=qcnn_cong.U_PARAMS, embedding_type="Angle", cost_fn="cross_entropy"
        ),
        extract=lambda probs: probs[1],
    )


def _gong_spec():
    from qcnn_benchmark.models import qcnn_gong

    return CircuitSpec(
        get_qnode=lambda: qcnn_gong._circuit,
        n_wires=qcnn_gong.N_QUBITS,
        call=lambda qnode, params, x: qnode(x, params),
        extract=lambda probs: probs[1],
    )


CIRCUIT_SPECS = {
    "hur": _hur_spec,
    "cong": _cong_spec,
    "gong": _gong_spec,
}


def get_circuit_spec(model_name):
    """`CircuitSpec` del modelo `model_name` ("hur", "cong" o "gong" --
    Wei se maneja aparte, ver docstring del módulo)."""
    if model_name not in CIRCUIT_SPECS:
        raise KeyError(
            f"'{model_name}' no está en el registro de circuitos genérico "
            f"({sorted(CIRCUIT_SPECS)}); Wei et al. se maneja por separado "
            "en execution.shots.make_wei_predict_proba / "
            "noise.make_wei_noisy_predict_proba."
        )
    return CIRCUIT_SPECS[model_name]()
