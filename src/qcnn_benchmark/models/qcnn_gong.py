"""Adaptador de Gong, Pei, Zhang & Zhou (2024), "Quantum convolutional
neural network based on variational quantum circuits" (Optics
Communications 550, 129993).

Circuito de 8 qubits, sin capa fully-connected (el paper la omite
explícitamente para clasificación binaria, Sec. 3.3): tres capas
convolución+pooling que reducen 8 -> 4 -> 2 -> 1 qubits, codificación
tree-structured hybrid amplitude (la aportación central del paper).

Codificación tree-structured hybrid amplitude (Sec. 3.2, Fig. 2)
------------------------------------------------------------------
Los N=8 qubits se dividen en m bloques independientes de n qubits
(`BLOCK_SIZE`). Dentro de cada bloque, el k-ésimo qubit (k=0..n-1) recibe
2**k rotaciones Ry controladas por los k qubits anteriores del bloque
(una por cada patrón de control posible) -- la generalización "en árbol"
de una rotación uniformemente controlada (Möttönen et al.), pero usando
el propio valor de la feature (escalado a [0, π]) como ángulo en vez de
derivarlo de una amplitud objetivo. Cada bloque de n qubits codifica así
2**n - 1 features (Ec. 3 del paper), para una capacidad total de
`N_BLOCKS * (2**BLOCK_SIZE - 1)` features por los N qubits. A diferencia
de la codificación "hybrid amplitude" simple, el escalado a [0, π] es
GLOBAL (no una normalización independiente por bloque) -- ver el párrafo
tras la Ec. 3 -- lo que aquí se obtiene reutilizando sin cambios
`qcnn_benchmark.representations.pca.dense_pca_representation` (ya hace
ese escalado global con min/max de train) y rellenando con ceros
(Ry(0) = identidad) las features que faltan hasta la capacidad del
encoding cuando `n_components < capacidad` -- p.ej. PCA-8 con
BLOCK_SIZE=2 (capacidad 12).

El orden exacto de qué bit de control corresponde a qué feature dentro de
un bloque (k>=2, i.e. bloques de 4+ qubits) es una convención interna
arbitraria pero consistente entre train/val/test: no afecta la capacidad
expresiva del encoding, solo qué feature de PCA cae en qué rama del
árbol.

Circuito convolucional: "Circuit 6" (Fig. 4f del paper)
--------------------------------------------------------
El paper reporta a Circuit 6 con exactamente 10 parámetros por
aplicación, gates Rx/Rz de un qubit e interacción de dos qubits
Rx-controlada. Esto identifica a Circuit 6 con el ansatz ya vendorizado
`unitary.U_6` de external/hur_qcnn/ (Apache-2.0, https://github.com/
takh04/QCNN) -- el mismo circuito que `qcnn_hur.py` reutiliza para su
"Ansatz 8" (10 parámetros, Rx/Rz + CRX + Rx/Rz): incluso comparten origen
en la familia de circuitos de referencia de Sim, Johnson & Aspuru-Guzik
(2019), de donde ambos papers derivan su numeración de "Ansatz"/"Circuit".
Se reutiliza sin modificar en vez de re-derivarlo gate por gate desde la
figura rasterizada del paper, reduciendo el riesgo de una transcripción
incorrecta de un circuito ya validado numéricamente en este repositorio.

Pooling (Fig. 5 del paper)
---------------------------
CNOT(descartado -> conservado) seguido de Rz y Rx sobre el qubit
conservado (2 parámetros). El qubit descartado no vuelve a usarse
--correcto en un simulador de estado completo sin necesidad de traza
parcial explícita, igual que documenta `qcnn_wei.py` para su propio
pooling.

Topología de cableado (Fig. 3, esquemática): pares adyacentes en cada
capa, compartiendo los mismos parámetros dentro de la capa (invariancia
traslacional, Sec. 3.1) -- 4 aplicaciones de conv+pool en la capa 1
(pares (0,1)(2,3)(4,5)(6,7)), 2 en la capa 2 (pares (0,2)(4,6) sobre los
4 qubits sobrevivientes) y 1 en la capa 3 (par (0,4)), terminando en el
qubit 0.

Total de parámetros: 3*10 (conv) + 3*2 (pool) = 36 -- coincide, por la
reutilización de U_6, con el conteo de Ansatz 8 de Hur et al. (Tabla I:
36) y con "Circuit 6" de Gong et al. (Sec. 4.1: "Circuit 6, featuring 10
parameters").
"""

import pathlib
import sys

import autograd.numpy as anp
import pennylane as qml

_MODULE_DIR = pathlib.Path(__file__).resolve()
PROJECT_ROOT = _MODULE_DIR
while not (PROJECT_ROOT / "external").exists() and PROJECT_ROOT != PROJECT_ROOT.parent:
    PROJECT_ROOT = PROJECT_ROOT.parent
HUR_QCNN_DIR = PROJECT_ROOT / "external" / "hur_qcnn" / "QCNN"
assert HUR_QCNN_DIR.exists(), f"No se encontró {HUR_QCNN_DIR}"

if str(HUR_QCNN_DIR) not in sys.path:
    sys.path.insert(0, str(HUR_QCNN_DIR))

import unitary as _hur_unitary  # noqa: E402  (vendorizado, Apache-2.0, takh04/QCNN)

assert hasattr(_hur_unitary, "U_6")

N_QUBITS = 8
BLOCK_SIZE = 2  # "our encoding n=2" en la Tabla 1 del paper
N_BLOCKS = N_QUBITS // BLOCK_SIZE
ENCODING_CAPACITY = N_BLOCKS * (2**BLOCK_SIZE - 1)  # 12 features (m=4, 2**2-1=3)

CONV_U_PARAMS = 10  # unitary.U_6 == "Circuit 6" de Gong et al.
POOL_PARAMS = 2
TOTAL_PARAMS = 3 * CONV_U_PARAMS + 3 * POOL_PARAMS  # 36

_CONV_PAIRS = [
    [(0, 1), (2, 3), (4, 5), (6, 7)],  # capa 1: 8 qubits activos
    [(0, 2), (4, 6)],  # capa 2: 4 qubits activos (0, 2, 4, 6)
    [(0, 4)],  # capa 3: 2 qubits activos (0, 4)
]
_POOL_PAIRS = [
    [(1, 0), (3, 2), (5, 4), (7, 6)],  # (descartado, conservado)
    [(2, 0), (6, 4)],
    [(4, 0)],
]

_dev = qml.device("default.qubit", wires=N_QUBITS)


def _tree_block_encoding(x_block, wires):
    """Codificación tree-structured amplitude (Fig. 2) dentro de un único
    bloque de `len(wires)` qubits: 2**k rotaciones Ry controladas para el
    k-ésimo qubit del bloque, consumiendo 2**len(wires) - 1 features."""
    n = len(wires)
    idx = 0
    for level in range(n):
        n_branches = 2**level
        for branch in range(n_branches):
            angle = x_block[idx]
            idx += 1
            if level == 0:
                qml.RY(angle, wires=wires[0])
            else:
                control_values = [(branch >> b) & 1 for b in reversed(range(level))]
                qml.ctrl(qml.RY, control=wires[:level], control_values=control_values)(angle, wires=wires[level])


def _tree_hybrid_encoding(x_padded):
    for b in range(N_BLOCKS):
        wires = list(range(b * BLOCK_SIZE, (b + 1) * BLOCK_SIZE))
        slot = 2**BLOCK_SIZE - 1
        x_block = x_padded[b * slot : (b + 1) * slot]
        _tree_block_encoding(x_block, wires)


def _pooling_gong(params, wires):
    """Fig. 5 del paper: CNOT(descartado -> conservado), luego Rz, Rx sobre
    el qubit conservado (2 parámetros)."""
    discard, keep = wires
    qml.CNOT(wires=[discard, keep])
    qml.RZ(params[0], wires=keep)
    qml.RX(params[1], wires=keep)


@qml.qnode(_dev)
def _circuit(x, params):
    n_missing = ENCODING_CAPACITY - x.shape[0]
    assert n_missing >= 0, (
        f"x tiene {x.shape[0]} features, más que la capacidad del encoding " f"({ENCODING_CAPACITY})"
    )
    x_padded = anp.concatenate([x, anp.zeros(n_missing)]) if n_missing > 0 else x
    _tree_hybrid_encoding(x_padded)

    p1, p2, p3 = params[0:10], params[10:20], params[20:30]
    p4, p5, p6 = params[30:32], params[32:34], params[34:36]

    for pair in _CONV_PAIRS[0]:
        _hur_unitary.U_6(p1, wires=list(pair))
    for pair in _POOL_PAIRS[0]:
        _pooling_gong(p4, wires=pair)

    for pair in _CONV_PAIRS[1]:
        _hur_unitary.U_6(p2, wires=list(pair))
    for pair in _POOL_PAIRS[1]:
        _pooling_gong(p5, wires=pair)

    for pair in _CONV_PAIRS[2]:
        _hur_unitary.U_6(p3, wires=list(pair))
    for pair in _POOL_PAIRS[2]:
        _pooling_gong(p6, wires=pair)

    return qml.probs(wires=0)


def forward_probs(x, params):
    """qml.probs(wires=0) tras las 3 capas de convolución+pooling."""
    return _circuit(x, params)


def predict_proba(params, x):
    """Probabilidad de clase positiva (índice 1) -- convención propia de
    este adaptador (no hay convención previa que reproducir: Gong et al.
    no publican código de referencia)."""
    return forward_probs(x, params)[1]
