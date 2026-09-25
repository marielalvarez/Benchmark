"""Pruebas del esquema de configuración declarativa (Pydantic + YAML + hash
canónico) usado por E0A."""

import pytest
from pydantic import ValidationError

from qcnn_benchmark.config import load_config


def test_e0a_config_loads_and_validates():
    cfg = load_config("configs/e0a.yaml")
    assert cfg.experiment_id == "E0A"
    assert {d.name for d in cfg.datasets} == {"coat_vs_shirt", "4_vs_9", "1_vs_0"}
    assert cfg.models == ["hur", "wei", "gong"]
    assert cfg.n_seeds == 5


def test_run_seeds_derive_from_seed_root():
    cfg = load_config("configs/e0a.yaml")
    assert cfg.run_seeds() == [cfg.seed_root + i for i in range(cfg.n_seeds)]


def test_canonical_hash_is_stable_across_independent_loads():
    cfg_a = load_config("configs/e0a.yaml")
    cfg_b = load_config("configs/e0a.yaml")
    assert cfg_a.canonical_hash() == cfg_b.canonical_hash()


def test_canonical_hash_changes_if_config_changes():
    cfg = load_config("configs/e0a.yaml")
    tweaked = cfg.model_copy(update={"n_seeds": cfg.n_seeds + 1})
    assert tweaked.canonical_hash() != cfg.canonical_hash()


def test_config_is_immutable():
    cfg = load_config("configs/e0a.yaml")
    with pytest.raises(ValidationError):
        cfg.experiment_id = "other"


def test_e0a_canonical_hash_matches_already_computed_results():
    """Regresión: results/e0a_raw_5a6b6cb49509.csv y el provenance ya
    calculados (E0A completo, 45 corridas) están indexados por este hash.
    Si algún cambio al esquema de ExperimentConfig (p.ej. agregar un campo
    nuevo) alguna vez cambia este valor, ese enlace se rompe en silencio --
    este test existe para que no pase desapercibido."""
    cfg = load_config("configs/e0a.yaml")
    assert cfg.canonical_hash() == "5a6b6cb49509"


def test_optional_e1_e2_fields_default_to_none_and_dont_affect_hash():
    """shots_budgets/noise_conditions son específicos de E1/E2 -- en
    E0A deben quedar en None y no aparecer en el hash (ver
    ExperimentConfig.canonical_json, exclude_none=True)."""
    cfg = load_config("configs/e0a.yaml")
    assert cfg.shots_budgets is None
    assert cfg.noise_conditions is None
    assert '"shots_budgets"' not in cfg.canonical_json()
    assert '"noise_conditions"' not in cfg.canonical_json()
