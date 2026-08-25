"""CNN clásica análoga al QCNN de Gong et al. (`qcnn_gong`, Circuit 6 +
codificación tree-hybrid, 36 parámetros entrenables).

Arquitectura compartida en `_cnn_common` (Conv1d -> ReLU -> MaxPool1d ->
Conv1d -> ReLU -> MaxPool1d -> Linear -> sigmoide), con la dimensión de
entrada elegida para igualar aproximadamente ese conteo de parámetros --
el mismo criterio de Gong et al. (2024) mismos, Tabla 4 (ellos comparan su
propia QCNN contra una CNN con conteo de parámetros cercano: PCA-8 -> CNN
de 26 vs. QCNN-Circuit5 de 24; PCA-16 -> 34 vs. QCNN-Circuit6 de 36).

Aquí: `N_FEATURE=2` canales convolucionales, PCA-36 -> 35 parámetros.
Objetivo (`qcnn_gong.TOTAL_PARAMS`): 36. Diferencia: 1 parámetro -- misma
configuración que `cnn_hur_analoga` porque ambas QCNN comparadas (Ansatz 8
de Hur == Circuit 6 de Gong, ambas construidas sobre `unitary.U_6`)
tienen el mismo conteo de parámetros (36); no es un error de copiado.

Espera datos con la representación PCA densa de
qcnn_benchmark.representations.pca con n_components=INPUT_SIZE (nótese
que esto es independiente de la capacidad del encoding tree-hybrid de la
QCNN de Gong: el análogo clásico no usa esa codificación cuántica).
"""

from qcnn_benchmark.models import _cnn_common

N_FEATURE = 2
INPUT_SIZE = 36
TOTAL_PARAMS = _cnn_common.total_params(INPUT_SIZE, N_FEATURE)
assert TOTAL_PARAMS == 35  # objetivo QCNN: qcnn_gong.TOTAL_PARAMS == 36 (diferencia: 1 parámetro)

# init_fn(rng, n_params) para pasar a train_binary_classifier -- ver
# _cnn_common.make_kaiming_init (no normal_init: N(0,0.1) isotrópico deja
# esta arquitectura con ReLU muerta con alta probabilidad).
init = _cnn_common.make_kaiming_init(INPUT_SIZE, N_FEATURE)


def predict_proba(params, x):
    return _cnn_common.forward_proba(params, x, n_feature=N_FEATURE)
