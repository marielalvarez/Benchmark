"""Esquema Pydantic para configuración declarativa de experimentos, con hash
canónico de reproducibilidad.

Alcance de esta primera versión: lo mínimo que necesita E0A (comparación
Hur/Wei/Gong sobre los tres datasets del rediseño de Semana 2, criterio de
selección con IC bootstrap BCa + test pareado de permutación de signo +
Holm). No es el sistema de registro de plugins de modelos/datasets/ruido
descrito como completo en el diseño original -- eso sigue pendiente y se
amplía aquí según lo vayan necesitando experimentos futuros (E1, E2, ...),
en vez de construirse de antemano sin un consumidor concreto.

Todos los modelos son inmutables (`model_config = ConfigDict(frozen=True)`):
una `ExperimentConfig` ya construida no se muta, se reemplaza. El hash
canónico (`ExperimentConfig.canonical_hash()`) se calcula sobre una
serialización JSON con claves ordenadas, así que es estable entre máquinas
y entre-corridas siempre que la configuración declarada sea la misma.
"""

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class DatasetSpec(BaseModel):
    """Una tarea de clasificación binaria: dos clases de un dataset fuente.

    Convención de polaridad de este proyecto (antes no explícita en el
    código -- Hur et al. "0 vs 1" usa class_pos=1, Wei et al. "1 vs 8" usa
    class_pos=1, es decir, sin un patrón consistente previo): en las
    `DatasetSpec` declaradas aquí, `class_pos` es siempre la primera clase
    nombrada en `name` (p.ej. "coat_vs_shirt" -> class_pos=coat). La
    exactitud de un clasificador binario no depende de esta elección (ver
    la nota de equivalencia 0-vs-1 / 1-vs-0 en `notebooks/00_reproduce_hur.
    ipynb`), pero fijar una convención evita ambigüedad al reportar
    resultados.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    source: Literal["mnist", "fashion_mnist"]
    class_pos: int
    class_neg: int


class TrainingProtocol(BaseModel):
    """Hiperparámetros de `qcnn_benchmark.training.train_binary_classifier`.
    Los valores por default replican los defaults del módulo (BATCH_SIZE,
    N_UPDATES, etc. en `training/loop.py`) -- declararlos aquí también, en
    vez de solo referenciarlos, es lo que hace que la config sea autónoma
    y hasheable sin depender de leer el código para saber qué se corrió."""

    model_config = ConfigDict(frozen=True)

    batch_size: int = 25
    n_updates: int = 200
    learning_rate: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    clip_norm: float = 5.0
    val_check_every: int = 10
    patience_checks: int = 5
    min_delta: float = 1e-4


class StatisticalCriterion(BaseModel):
    """Criterio de selección del "mejor" candidato, confirmado para E0A:
    media +/- IC bootstrap BCa por dataset (no una cifra resumen única),
    con test pareado de permutación de signo + corrección de Holm para las
    comparaciones Hur-vs-Wei-vs-Gong. Aplicación directa del protocolo
    estadístico ya descrito en `context/diseño_experimentos (2).pdf`
    (intervalos bootstrap BCa, test pareado de permutación de signo,
    corrección de Holm, tamaños de efecto pareados), no un método nuevo."""

    model_config = ConfigDict(frozen=True)

    aggregation: Literal["mean_per_dataset"] = "mean_per_dataset"
    ci_method: Literal["bootstrap_bca"] = "bootstrap_bca"
    n_bootstrap: int = 10_000
    confidence_level: float = 0.95
    comparison_test: Literal["paired_sign_permutation"] = "paired_sign_permutation"
    multiple_comparison_correction: Literal["holm"] = "holm"
    alpha: float = 0.05


class ExperimentConfig(BaseModel):
    """Configuración completa e inmutable de un experimento. `canonical_hash()`
    identifica la configuración de forma reproducible entre máquinas."""

    model_config = ConfigDict(frozen=True)

    experiment_id: str
    description: str
    datasets: list[DatasetSpec] = Field(min_length=1)
    models: list[str] = Field(min_length=1)
    protocol: TrainingProtocol = TrainingProtocol()
    n_seeds: int = 5
    seed_root: int
    sample_seed: int
    statistical_criterion: StatisticalCriterion = StatisticalCriterion()

    def run_seeds(self) -> list[int]:
        """Las `n_seeds` semillas de ejecución, derivadas de `seed_root`
        (principio de "toda semilla deriva de una semilla raíz" del diseño
        original: ninguna parte del código debe leer estado aleatorio
        implícito)."""
        return [self.seed_root + i for i in range(self.n_seeds)]

    def canonical_json(self) -> str:
        """Serialización JSON canónica (claves ordenadas, sin espacios
        ambiguos) usada para el hash -- estable entre corridas y máquinas."""
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))

    def canonical_hash(self) -> str:
        """sha256 de `canonical_json()`, primeros 12 caracteres hex (estilo
        git-short-hash): suficiente para distinguir configuraciones en
        nombres de archivo/reportes sin acarrear un hash de 64 caracteres."""
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()[:12]
