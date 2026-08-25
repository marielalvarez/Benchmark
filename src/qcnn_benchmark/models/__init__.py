"""Adaptadores de modelo (QCNN y CNN análogas).

Los submódulos se importan explícitamente (p.ej. `from qcnn_benchmark.models
import qcnn_hur`) en vez de reexportarse aquí, porque algunos (qcnn_hur)
tienen efectos secundarios de import (sys.path, código vendorizado externo)
que solo deben ejecutarse cuando ese modelo específico hace falta.
"""
