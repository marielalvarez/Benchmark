"""CNN clásica análoga al QCNN de Wei et al. (`qcnn_wei`, filtro LCU +
lectura hamiltoniana, 46 parámetros entrenables).

Arquitectura compartida en `_cnn_common` (Conv1d -> ReLU -> MaxPool1d ->
Conv1d -> ReLU -> MaxPool1d -> Linear -> sigmoide), con la dimensión de
entrada y el número de canales elegidos para igualar el conteo de
parámetros -- el mismo criterio de Gong et al. (2024), Tabla 4 ("ajustar
la dimensión de PCA para que el conteo de parámetros de la CNN sea
cercano al de la QCNN bajo comparación").

Aquí: `N_FEATURE=3` canales convolucionales, PCA-20 -> 46 parámetros
(coincidencia exacta con `qcnn_wei.TOTAL_PARAMS`). Nótese que `qcnn_wei`
en sí usa codificación de amplitud directa (1024 = 32x32 px, sin PCA) --
el análogo clásico sí se reduce con PCA, siguiendo la práctica de Gong et
al., que también reduce con PCA el input de la CNN de referencia
independientemente de qué encoding use la QCNN comparada.

Espera datos con la representación PCA densa de
qcnn_benchmark.representations.pca con n_components=INPUT_SIZE.
"""

from qcnn_benchmark.models import _cnn_common

N_FEATURE = 3
INPUT_SIZE = 20
TOTAL_PARAMS = _cnn_common.total_params(INPUT_SIZE, N_FEATURE)
assert TOTAL_PARAMS == 46  # objetivo QCNN: qcnn_wei.TOTAL_PARAMS == 46 (coincidencia exacta)

# init_fn(rng, n_params) para pasar a train_binary_classifier -- ver
# _cnn_common.make_kaiming_init (no normal_init: N(0,0.1) isotrópico deja
# esta arquitectura con ReLU muerta con alta probabilidad).
init = _cnn_common.make_kaiming_init(INPUT_SIZE, N_FEATURE)


def predict_proba(params, x):
    return _cnn_common.forward_proba(params, x, n_feature=N_FEATURE)
