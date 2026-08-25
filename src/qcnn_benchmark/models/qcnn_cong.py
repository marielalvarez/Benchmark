"""Adaptador de la arquitectura QCNN genérica de Cong, Choi & Lukin (2019),
"Quantum convolutional neural networks" (Nat. Phys. 15, 1273-1278).

Cong et al. no proponen una tarea de clasificación de imágenes clásicas ni
un esquema de codificación de datos clásicos: su QCNN opera directamente
sobre estados cuánticos de entrada (reconocimiento de fases topológicas
protegidas por simetría, optimización de códigos de corrección de
errores). Lo que aportan -- y lo que se reproduce aquí como arquitectura
de referencia "genérica" para el sub-experimento E0B -- es el patrón
estructural (Fig. 1b del paper): una capa convolucional que aplica un
único unitario de 2 qubits U_i de forma invariante a traslación, seguida
de una capa de pooling que mide una fracción de los qubits y aplica una
rotación condicionada V_j a los restantes, repitiendo hasta reducir el
registro a un solo qubit de lectura.

E0B compara esta QCNN "sin el sesgo de diseño de ningún paper específico"
contra Hur/Wei/Gong para aislar si sus aportaciones respectivas
(codificación densa, filtro LCU + hamiltoniano, codificación
tree-hybrid) valen algo más allá de una QCNN jerárquica genérica con el
unitario de 2 qubits más expresivo posible y la codificación de datos más
neutra posible.

Reutiliza, sin modificar, el circuito vendorizado en external/hur_qcnn/
(Apache-2.0, https://github.com/takh04/QCNN) -- el mismo repositorio ya
usado por `qcnn_hur.py`:

- unitary.U_SU4 -- unitario general de 2 qubits (SU(4) completo vía la
  descomposición U3+CNOT de Vatan-Williams, 15 parámetros): la elección
  "genérica" más natural para U_i, sin sesgo hacia ninguna tarea o
  encoding particular (es, por construcción, capaz de representar
  cualquier operación de 2 qubits).
- unitary.Pooling_ansatz1 -- el mismo pooling parametrizado (CRZ + X +
  CRX, 2 parámetros) que ya usa `qcnn_hur.py`; una instancia concreta y
  simulable con gradiente de "medir una fracción de qubits y aplicar una
  rotación condicionada" (Fig. 1b de Cong et al.).
- embedding.data_embedding(..., embedding_type='Angle') -- AngleEmbedding
  estándar (1 feature por qubit, rotación Y): la codificación más neutra
  posible, a diferencia de la codificación densa de Hur o la tree-hybrid
  de Gong, que sí son parte de la aportación específica de esos papers.
- QCNN_circuit.QCNN_structure -- la topología jerárquica de cableado
  8->4->2->1 (conv_layer1/2/3 + pooling_layer1/2/3), reutilizada intacta
  vía QCNN_circuit.QCNN con U='U_SU4'.

Espera datos con la representación PCA densa de
qcnn_benchmark.representations.pca con n_components=8 (una componente por
qubit), escalada a [0, π].
"""

import pathlib
import sys

_MODULE_DIR = pathlib.Path(__file__).resolve()
PROJECT_ROOT = _MODULE_DIR
while not (PROJECT_ROOT / "external").exists() and PROJECT_ROOT != PROJECT_ROOT.parent:
    PROJECT_ROOT = PROJECT_ROOT.parent
HUR_QCNN_DIR = PROJECT_ROOT / "external" / "hur_qcnn" / "QCNN"
assert HUR_QCNN_DIR.exists(), f"No se encontró {HUR_QCNN_DIR}"

if str(HUR_QCNN_DIR) not in sys.path:
    sys.path.insert(0, str(HUR_QCNN_DIR))

import QCNN_circuit as _hur_circuit  # noqa: E402  (vendorizado, Apache-2.0, takh04/QCNN)
import unitary as _hur_unitary  # noqa: E402  (vendorizado, Apache-2.0, takh04/QCNN)

assert hasattr(_hur_unitary, "U_SU4")

U_PARAMS = 15  # parámetros de U_SU4 por aplicación (SU(4) completo)
N_POOL_PARAMS = 2  # parámetros de Pooling_ansatz1 por aplicación
TOTAL_PARAMS = 3 * U_PARAMS + 3 * N_POOL_PARAMS  # 51


def forward_probs(x, params):
    """qml.probs(wires=4) del qnode QCNN_circuit.QCNN, sin modificar,
    fijando U='U_SU4' (unitario de 2 qubits genérico, sin sesgo de ningún
    paper específico) y codificación AngleEmbedding estándar (1 feature
    por qubit)."""
    return _hur_circuit.QCNN(x, params, U="U_SU4", U_params=U_PARAMS, embedding_type="Angle", cost_fn="cross_entropy")


def predict_proba(params, x):
    """Probabilidad de clase positiva (índice 1 de qml.probs(wires=4)),
    misma convención que qcnn_hur.predict_proba (confirmada contra
    Benchmarking.accuracy_test del repositorio de referencia)."""
    return forward_probs(x, params)[1]
