"""Ejecucion en segundo plano de la expansion de E1 (shots finitos, sin
ruido) a Hur, Wei y Gong -- decision confirmada 21-sep-2026, ver la celda
de INPUTS MANUALES de notebooks/E1_finite_shots.ipynb para el detalle
completo. Espeja esa celda de ejecucion exactamente (mismo MODEL_REGISTRY,
mismo protocolo, mismos seeds), agregando confusion_counts desde el inicio.

Estimado (cota superior, 0 early stopping, ver Sec. 2 del notebook):
~16.2 dias Hur + ~5.5 dias Gong + ~0.16 dias Wei = ~21.8 dias combinados.
Cong NO esta incluido (decision explicita, ver notebook).

Se lanza bajo `caffeinate -i` para que una suspension de la maquina no
contamine los timestamps acumulados como paso con la corrida original de
E1 (results/e1_raw_c2ee95d45e65.csv, ver la nota en la celda de costo del
notebook) -- eso no afecta la exactitud entrenada, pero conviene evitarlo
de todos modos dado lo largo de esta corrida.

Tambien llama a `os.setsid()` (ver incidente 21-sep-2026: un reinicio de
la sesion de Claude Code/VSCode mato tanto esta corrida como el backfill
de E0 a mitad de camino, porque ninguno de los dos estaba realmente
independizado del grupo de procesos del lanzador -- `nohup` por si solo
no alcanza, solo protege contra SIGHUP).

Este archivo vive en `scripts/` DENTRO del repo (no en /tmp ni en el
scratchpad de ninguna sesion) precisamente para sobrevivir limpiezas
periodicas de directorios temporales -- confirmado 22-sep-2026 tras un
incidente separado (ver notebooks/E1_finite_shots.ipynb) donde una
corrida equivalente en otra maquina (wscorelab01, sesion manual SSH,
sin relacion con este proceso) quedo muda por el mismo problema de
buffering que se corrige aqui.

Todos los `print()` usan `flush=True` (no basta con solo eso -- lanzar
con `python3 -u`, no `python3` a secas, sigue siendo necesario para que
la salida de librerias de terceros, p.ej. warnings de PennyLane, tampoco
se quede atrapada en el buffer). `train_binary_classifier` corre con
`verbose=True` para imprimir progreso cada `val_check_every` (10)
actualizaciones -- antes con `verbose=False` el log quedaba en silencio
hasta que una corrida ENTERA terminaba, hasta 8.6h en el peor caso para
Hur, sin ninguna senal intermedia de que seguia viva.
"""

import os
import pathlib
import time

try:
    os.setsid()
except OSError:
    pass

import pandas as pd

from qcnn_benchmark.config import load_config
from qcnn_benchmark.data import load_mnist_pool, load_fashion_mnist_pool
from qcnn_benchmark.representations import build_pca_dataset, build_amplitude_dataset
from qcnn_benchmark.models import qcnn_hur, qcnn_wei, qcnn_gong
from qcnn_benchmark.training import train_binary_classifier, uniform_pi_init, normal_init
from qcnn_benchmark.execution import make_shots_predict_proba
from qcnn_benchmark.execution.shots import n_trainable_params
from qcnn_benchmark.metrics import batch_accuracy, confusion_counts

MODELS_TO_RUN = ["hur", "wei", "gong"]

MODEL_REGISTRY = {
    "hur": {"module": qcnn_hur, "init": uniform_pi_init, "normalize": True,
            "build_rep": lambda x, y, pos, neg: build_pca_dataset(x, y, pos, neg, n_components=16, verbose=False)},
    "wei": {"module": qcnn_wei, "init": normal_init, "normalize": False,
            "build_rep": lambda x, y, pos, neg: build_amplitude_dataset(x, y, pos, neg, verbose=False)},
    "gong": {"module": qcnn_gong, "init": uniform_pi_init, "normalize": True,
             "build_rep": lambda x, y, pos, neg: build_pca_dataset(x, y, pos, neg, n_components=8, verbose=False)},
}

_POOL_CACHE = {}


def get_pool(source, normalize):
    key = (source, normalize)
    if key not in _POOL_CACHE:
        loader = load_mnist_pool if source == "mnist" else load_fashion_mnist_pool
        _POOL_CACHE[key] = loader(normalize=normalize)
    return _POOL_CACHE[key]


def main():
    _CONFIG_CANDIDATES = [pathlib.Path("configs/e1.yaml"), pathlib.Path("../configs/e1.yaml")]
    config_path = next(p for p in _CONFIG_CANDIDATES if p.exists())
    config = load_config(config_path)
    config = config.model_copy(update={"models": MODELS_TO_RUN})

    RESULTS_DIR = pathlib.Path("results") if pathlib.Path("results").exists() else pathlib.Path("../results")
    raw_path = RESULTS_DIR / f"e1_raw_{config.canonical_hash()}.csv"
    print("canonical_hash:", config.canonical_hash(), flush=True)
    print("models:", config.models, flush=True)
    print("raw_path:", raw_path, flush=True)

    # Reanudacion: si ya existe un CSV parcial de una corrida que murio a
    # mitad de camino (ver incidente 21-sep-2026), no se repiten las
    # combinaciones ya presentes.
    if raw_path.exists():
        existing_df = pd.read_csv(raw_path)
        rows = existing_df.to_dict("records")
        done_keys = {(r["dataset"], r["model"], r["n_shots"], r["seed"]) for r in rows}
        print(f"Reanudando: {len(rows)} filas ya hechas en {raw_path}, se saltan.", flush=True)
    else:
        rows = []
        done_keys = set()

    t_start = time.time()
    for dataset in config.datasets:
        for model_name in config.models:
            entry = MODEL_REGISTRY[model_name]
            x_all, y_all = get_pool(dataset.source, entry["normalize"])
            rep = entry["build_rep"](x_all, y_all, dataset.class_pos, dataset.class_neg)
            n_params = n_trainable_params(model_name)

            for n_shots in config.shots_budgets:
                predict_proba = make_shots_predict_proba(model_name, n_shots)

                for seed in config.run_seeds():
                    if (dataset.name, model_name, n_shots, seed) in done_keys:
                        continue
                    tag = f"{dataset.name}-{model_name}-shots{n_shots}-seed{seed}"
                    result = train_binary_classifier(
                        predict_proba, n_params, rep, entry["init"],
                        run_seed=seed,
                        batch_size=config.protocol.batch_size,
                        n_updates=config.protocol.n_updates,
                        learning_rate=config.protocol.learning_rate,
                        beta1=config.protocol.beta1,
                        beta2=config.protocol.beta2,
                        clip_norm=config.protocol.clip_norm,
                        val_check_every=config.protocol.val_check_every,
                        patience_checks=config.protocol.patience_checks,
                        min_delta=config.protocol.min_delta,
                        verbose=True,
                        tag=tag,
                    )
                    train_acc = batch_accuracy(predict_proba, result["params"], rep["X_train"], rep["y_train"])
                    test_acc = batch_accuracy(predict_proba, result["params"], rep["X_test"], rep["y_test"])
                    test_conf = confusion_counts(predict_proba, result["params"], rep["X_test"], rep["y_test"])
                    rows.append({
                        "dataset": dataset.name, "model": model_name, "n_shots": n_shots, "seed": seed,
                        "train_acc": train_acc, "test_acc": test_acc,
                        "test_tp": test_conf["tp"], "test_fp": test_conf["fp"],
                        "test_fn": test_conf["fn"], "test_tn": test_conf["tn"],
                        "n_updates_run": result["n_updates_run"], "stopped_early_at": result["stopped_early_at"],
                        "best_val_loss": result["best_val_loss"],
                    })
                    elapsed = time.time() - t_start
                    print(f"[{tag}] test_acc={test_acc:.4f}  (t={elapsed:.0f}s / {elapsed/3600:.1f}h acumulado)",
                          flush=True)
                    pd.DataFrame(rows).to_csv(raw_path, index=False)

    print("Total:", time.time() - t_start, "s")

    import json
    import numpy as np
    import pennylane, sklearn, scipy
    provenance = {
        "experiment_id": config.experiment_id,
        "winner_model": "wei",
        "include_cong": False,
        "scope_expansion": "hur+wei+gong, confirmado 21-sep-2026 (ver notebooks/E1_finite_shots.ipynb)",
        "canonical_hash": config.canonical_hash(),
        "config": json.loads(config.canonical_json()),
        "fast_smoke_test": False,
        "timestamp_utc": pd.Timestamp.now("UTC").isoformat(),
        "raw_results_path": str(raw_path),
        "package_versions": {
            "pennylane": pennylane.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__, "pandas": pd.__version__,
        },
    }
    provenance_path = RESULTS_DIR / f"e1_provenance_{config.canonical_hash()}.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False))
    print("Provenance guardado en:", provenance_path)


if __name__ == "__main__":
    main()
