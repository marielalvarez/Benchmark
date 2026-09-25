"""Transpilación a un mapa de acoplamiento fijo (E2): inserta SWAPs donde
haga falta para que el circuito respete la conectividad de un dispositivo
con solo `n_qubits` qubits físicos conectados según `coupling_map`,
reutilizando el transform ya incluido en PennyLane (`qml.transforms.
transpile`) en vez de reimplementar routing desde cero.

Nota honesta sobre fidelidad: el diseño original pide "SABRE, nivel de
optimización 2" -- el heurístico de routing SABRE (Sun et al.) es
específico de Qiskit, y replicarlo exactamente requeriría esa dependencia
(`qiskit` + `pennylane-qiskit`, no instaladas en este proyecto). Este
módulo resuelve el mismo problema (insertar SWAPs para satisfacer un mapa
de acoplamiento) con el router genérico que ya trae PennyLane, que no es
SABRE gate-por-gate pero cumple el mismo propósito de esta fase: exponer al
circuito a las restricciones de conectividad de un dispositivo NISQ real
antes de medir con ruido en E2. Migrar a un router Qiskit/SABRE real es un
follow-up natural si los resultados de E2 necesitan calzar con un
dispositivo físico específico.

Limitación conocida (de `qml.transforms.transpile`, no de este módulo):
no soporta medir expectation values de Hamiltonianos ni de productos
tensoriales de observables. Esto excluye a la lectura de Wei et al.
(`qml.expval(qml.Hamiltonian(...))`, 37 parámetros) -- para ese modelo, E2
aplica shots y ruido pero no este paso de transpilación (ver
`notebooks/E2_nisq_noise.ipynb`).

Segunda limitación descubierta al cronometrar E2 para Cong (18-sep-2026,
no se habia probado antes: `test_execution.py` solo cubre transpilación
sobre gong): `qml.transforms.transpile` revisa `len(op.wires) > 2` sobre
las operaciones de la cinta ANTES de expandir templates, no despues.
Un template multi-wire sin expandir (p.ej. `AngleEmbedding(wires=range(8))`,
que usan tanto Hur como Cong via el circuito vendorizado de
`external/hur_qcnn/`) cuenta como una sola operacion de 8 wires en ese
punto y dispara `NotImplementedError` aunque decomponga en puertas de 1
qubit. Gong no lo sufre porque su circuito ya esta escrito con puertas
elementales, sin templates multi-wire -- por eso el unico test existente
(`test_transpile_preserves_analytic_result_gong`) nunca lo detecto. La
correccion es expandir templates a puertas de 1-2 qubits ANTES de
`qml.transforms.transpile`, no despues -- ver `transpile_qnode`.
"""

import pennylane as qml


def grid_coupling_map(rows, cols):
    """Mapa de acoplamiento de una rejilla `rows` x `cols` (vecinos más
    cercanos, sin diagonales), como lista de aristas (i, j)."""
    edges = []
    for r in range(rows):
        for c in range(cols):
            node = r * cols + c
            if c + 1 < cols:
                edges.append((node, node + 1))
            if r + 1 < rows:
                edges.append((node, node + cols))
    return edges


# Mapa de acoplamiento fijo de 16 qubits del diseño original: una rejilla
# 4x4 -- elección genérica y documentada (no se afirma que corresponda a
# un dispositivo real específico), suficiente para el propósito de esta
# fase (ver nota de fidelidad arriba).
COUPLING_MAP_16Q = grid_coupling_map(4, 4)


def transpile_qnode(qnode, coupling_map=COUPLING_MAP_16Q):
    """Devuelve un qnode nuevo, equivalente a `qnode` pero con SWAPs
    insertados para respetar `coupling_map`. No muta `qnode`. Componible
    con `qcnn_benchmark.execution.shots.rebind`-style wrappers: aplicar
    primero `transpile_qnode` y luego `qml.set_shots(..., shots=n)` sobre
    el resultado.

    Expande templates multi-wire (p.ej. `AngleEmbedding`) a puertas de 1-2
    qubits ANTES de `qml.transforms.transpile` -- ver la nota en el
    docstring del módulo sobre por qué, si no, ese transform revienta con
    `NotImplementedError` en cualquier circuito que use un template sin
    expandir, aunque decomponga en puertas soportadas. No decompone las
    puertas que ya son de 1-2 qubits, así que no cambia el comportamiento
    ya validado para gong."""
    qnode = qml.transforms.decompose(qnode, stopping_condition=lambda op: len(op.wires) <= 2)
    return qml.transforms.transpile(qnode, coupling_map=coupling_map)
