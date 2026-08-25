"""Esquema Pydantic, carga de YAML y hashing canónico de configuración.

Implementado para lo que necesita E0A (ver `schema.py`); el registro de
plugins de modelos/datasets/ruido más general descrito en el diseño
original sigue pendiente, para cuando un experimento concreto (E1, E2, ...)
lo necesite.
"""

from .loader import load_config
from .schema import DatasetSpec, ExperimentConfig, StatisticalCriterion, TrainingProtocol

__all__ = [
    "DatasetSpec",
    "TrainingProtocol",
    "StatisticalCriterion",
    "ExperimentConfig",
    "load_config",
]
