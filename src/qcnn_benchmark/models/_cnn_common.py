"""Arquitectura CNN 1D compartida por los tres análogos clásicos
(`cnn_hur_analoga`, `cnn_wei_analoga`, `cnn_gong_analoga`): Conv1d -> ReLU
-> MaxPool1d -> Conv1d -> ReLU -> MaxPool1d -> Flatten -> Linear ->
sigmoide.

Implementada en autograd.numpy (no PyTorch, que no es dependencia de este
proyecto -- ver requirements.txt) para poder reutilizar sin cambios el
mismo `qcnn_benchmark.training.train_binary_classifier` (BCE/Adam/
early-stopping) que ya comparten los adaptadores QCNN, en vez de escribir
un ciclo de entrenamiento nuevo solo para las CNN.

Replica la arquitectura de referencia ya vendorizada en
external/hur_qcnn/QCNN/CNN.py (in_channels=1, kernel_size=2, padding=1,
MaxPool1d(kernel_size=2), dos bloques conv+pool), con un único cambio: una
sola salida sigmoide en vez de las dos salidas softmax del original --
necesario para compartir train_binary_classifier (BCE sobre una
probabilidad escalar, la misma convención que predict_proba en todos los
modelos QCNN de este proyecto). Como ese cambio solo afecta a la capa
lineal final, el conteo de parámetros de ambas capas convolucionales es
idéntico al de CNN.py.

Este módulo es privado (antepuesto con `_`): no es un modelo por sí mismo,
solo el generador de circuitería reutilizado por los tres análogos.
"""

import autograd.numpy as anp
import numpy as np
from pennylane import numpy as pnp

KERNEL_SIZE = 2
PAD = 1
POOL_SIZE = 2


def _conv1d(x_channels, w, b):
    """x_channels: (C_in, L). w: (C_out, C_in, KERNEL_SIZE). b: (C_out,)."""
    c_in, length = x_channels.shape
    zeros = anp.zeros((c_in, PAD))
    x_padded = anp.concatenate([zeros, x_channels, zeros], axis=1)
    out_len = length + 2 * PAD - KERNEL_SIZE + 1
    outs = [anp.tensordot(w, x_padded[:, t : t + KERNEL_SIZE], axes=([1, 2], [0, 1])) + b for t in range(out_len)]
    return anp.stack(outs, axis=1)


def _maxpool1d(x_channels):
    """Pooling no solapado de tamaño POOL_SIZE; descarta la cola si la
    longitud no es múltiplo exacto (igual que nn.MaxPool1d de PyTorch)."""
    c, length = x_channels.shape
    trimmed_len = (length // POOL_SIZE) * POOL_SIZE
    x2 = anp.reshape(x_channels[:, :trimmed_len], (c, length // POOL_SIZE, POOL_SIZE))
    return anp.max(x2, axis=2)


def _relu(x):
    return anp.maximum(x, 0.0)


def conv_output_length(input_size):
    """Longitud final tras dos bloques Conv1d(k=2, pad=1) + MaxPool1d(k=2)."""
    l1 = input_size + 2 * PAD - KERNEL_SIZE + 1
    p1 = l1 // POOL_SIZE
    l2 = p1 + 2 * PAD - KERNEL_SIZE + 1
    p2 = l2 // POOL_SIZE
    return p2


def total_params(input_size, n_feature):
    """Conteo de parámetros: 2 bloques conv (in/out=n_feature, salvo el
    primero con in=1) + capa lineal final de una sola salida sigmoide."""
    conv1 = n_feature * 1 * KERNEL_SIZE + n_feature
    conv2 = n_feature * n_feature * KERNEL_SIZE + n_feature
    final_len = conv_output_length(input_size)
    linear = n_feature * final_len + 1
    return conv1 + conv2 + linear


def _unpack_params(params, input_size, n_feature):
    idx = 0
    w1 = anp.reshape(params[idx : idx + n_feature * KERNEL_SIZE], (n_feature, 1, KERNEL_SIZE))
    idx += n_feature * KERNEL_SIZE
    b1 = params[idx : idx + n_feature]
    idx += n_feature

    w2 = anp.reshape(params[idx : idx + n_feature * n_feature * KERNEL_SIZE], (n_feature, n_feature, KERNEL_SIZE))
    idx += n_feature * n_feature * KERNEL_SIZE
    b2 = params[idx : idx + n_feature]
    idx += n_feature

    final_len = conv_output_length(input_size)
    w_lin = params[idx : idx + n_feature * final_len]
    idx += n_feature * final_len
    b_lin = params[idx]
    return w1, b1, w2, b2, w_lin, b_lin


def make_kaiming_init(input_size, n_feature):
    """Fábrica de `init_fn(rng, n_params)` (la interfaz que espera
    `train_binary_classifier`) con escala por capa apropiada para ReLU --
    `qcnn_benchmark.training.normal_init` (N(0, 0.1) isotrópico e
    independiente de la estructura de capas) deja esta arquitectura con
    ReLU muerta con alta probabilidad: como las entradas a la segunda capa
    conv son siempre >= 0 (salida de la ReLU anterior), un sesgo ~N(0,0.1)
    moderadamente negativo puede dominar por sí solo una contribución
    convolucional pequeña y apagar el canal completo para *todas* las
    posiciones a la vez, matando el gradiente de todo lo que está antes de
    la capa lineal final. Init de Kaiming (std = sqrt(2/fan_in) por capa,
    sesgos en 0) evita ese colapso."""

    def init_fn(rng, n_params):
        assert n_params == total_params(input_size, n_feature)

        fan_in1 = 1 * KERNEL_SIZE
        w1 = rng.normal(0.0, (2.0 / fan_in1) ** 0.5, n_feature * 1 * KERNEL_SIZE)
        b1 = np.zeros(n_feature)

        fan_in2 = n_feature * KERNEL_SIZE
        w2 = rng.normal(0.0, (2.0 / fan_in2) ** 0.5, n_feature * n_feature * KERNEL_SIZE)
        b2 = np.zeros(n_feature)

        final_len = conv_output_length(input_size)
        fan_in_lin = n_feature * final_len
        w_lin = rng.normal(0.0, (1.0 / fan_in_lin) ** 0.5, fan_in_lin)
        b_lin = np.zeros(1)

        flat = np.concatenate([w1, b1, w2, b2, w_lin, b_lin])
        return pnp.array(flat, requires_grad=True)

    return init_fn


def forward_proba(params, x, n_feature):
    # qml.AdamOptimizer.apply_grad iterates `params` as if it were a tuple of
    # separate trainable arguments (its docstring's `args`), not a single
    # flat array: after the first update it returns a plain Python list of
    # 0-d tensors instead of a 1-D array. Eso no rompe a los adaptadores
    # QCNN de este proyecto (solo indexan params[i] escalar por escalar
    # dentro del qnode), pero sí rompe operaciones de arreglo (reshape,
    # slices usados como pesos de matmul/conv) -- de ahí la
    # rematerialización aquí, de forma autograd-diferenciable, en cada
    # llamada.
    params = anp.array(params)
    input_size = x.shape[0]
    w1, b1, w2, b2, w_lin, b_lin = _unpack_params(params, input_size, n_feature)

    x0 = anp.reshape(x, (1, input_size))
    h1 = _relu(_conv1d(x0, w1, b1))
    p1 = _maxpool1d(h1)
    h2 = _relu(_conv1d(p1, w2, b2))
    p2 = _maxpool1d(h2)
    flat = anp.reshape(p2, (-1,))

    logit = anp.dot(w_lin, flat) + b_lin
    return 1.0 / (1.0 + anp.exp(-logit))
