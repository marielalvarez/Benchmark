"""CNN clásica análoga al QCNN de Hur et al. (`qcnn_hur`, Ansatz 8 / U_6,
36 parámetros entrenables).

Arquitectura compartida en `_cnn_common` (Conv1d -> ReLU -> MaxPool1d ->
Conv1d -> ReLU -> MaxPool1d -> Linear -> sigmoide), con la dimensión de
entrada elegida para igualar aproximadamente ese conteo de parámetros --
el mismo criterio de Gong et al. (2024), Tabla 4: "ajustar la dimensión de
PCA para que el conteo de parámetros de la CNN sea cercano al de la QCNN
bajo comparación" (en su caso: PCA-8 -> CNN de 26 parámetros vs. QCNN de
24; PCA-16 -> 34 vs. 36).

Aquí: `N_FEATURE=2` canales convolucionales (igual que la CNN de
referencia en external/hur_qcnn/QCNN/CNN.py), PCA-36 -> 35 parámetros.
Objetivo (`qcnn_hur.TOTAL_PARAMS`): 36. Diferencia: 1 parámetro -- dentro
del margen de las comparaciones de la Tabla 4 de Gong (que difieren en 2).

Espera datos con la representación PCA densa de
qcnn_benchmark.representations.pca con n_components=INPUT_SIZE.
"""

from qcnn_benchmark.models import _cnn_common

N_FEATURE = 2
INPUT_SIZE = 36
TOTAL_PARAMS = _cnn_common.total_params(INPUT_SIZE, N_FEATURE)
assert TOTAL_PARAMS == 35  # objetivo QCNN: qcnn_hur.TOTAL_PARAMS == 36 (diferencia: 1 parámetro)

# init_fn(rng, n_params) para pasar a train_binary_classifier -- no
# normal_init (N(0,0.1) isotrópico): con ReLU y sin normalización de
# activaciones, un init sin escala por capa deja esta arquitectura con
# ReLU muerta con alta probabilidad (ver _cnn_common.make_kaiming_init).
init = _cnn_common.make_kaiming_init(INPUT_SIZE, N_FEATURE)


def predict_proba(params, x):
    return _cnn_common.forward_proba(params, x, n_feature=N_FEATURE)
