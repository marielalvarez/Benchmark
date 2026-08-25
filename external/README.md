# external/

This directory holds checked-out or submodule-linked source
repositories for the concrete QCNN implementations this benchmark
wraps. Code inside `external/` is not edited directly by the benchmark
implementation — it is imported only through the corresponding bridge
module (`qcnn_bench.models.builtins.<name>.bridge`).

- `hur_qcnn/` — Hur, Kim & Park, "Quantum convolutional neural network
  for classical data classification" (Ansatz 8). Vendored as a Git
  submodule (Milestone 6), pinned to commit
  `9091189e198cb9d50c5e938b224a17bcfb2c14f9` of
  `https://github.com/takh04/QCNN` (Apache License 2.0). See
  `docs/adapters/hur_conformance_map.md` for the full paper-to-code
  conformance analysis, including the resolution of which upstream
  unitary "Ansatz 8" refers to.
- `wei_qcnn/` — Wei, Chen, Zhou & Long, "A quantum convolutional neural
  network on NISQ devices" (arXiv:2104.06918). Vendored as a Git
  submodule (Milestone 7), pinned to commit
  `95e5a07e8e9e75ba7e24e67fb32b030112a1309a` of
  `https://github.com/XanaduAI/qml-benchmarks` (Apache License 2.0).
  Its own `WeiNet` reference implementation (`src/qml_benchmarks/models/weinet.py`,
  byte-identical to the file formerly at the repository root as
  `wei.py`) replaces the paper's 4 physical ancilla qubits and
  measurement-postselection with a classical softmax mixture over a
  *fixed*, non-trainable filter — architecturally different from the
  paper, not merely a reduction — so it is retained only as an
  auxiliary analytic comparison and never determines the normative Wei
  adapter's architecture. See `docs/adapters/wei_source_audit.md` for
  the full classification and `docs/adapters/wei_conformance_map.md`
  for the paper-to-code conformance analysis the adapter is actually
  built against.

Both `hur_qcnn/` and `wei_qcnn/` are real Git submodules:
`git submodule update --init` (or a full `git clone
--recurse-submodules`) is required before the corresponding bridge
module can import them. Only `qcnn_bench.models.builtins.hur.bridge`
and `qcnn_bench.models.builtins.wei.bridge` may import from their
respective submodules directly — nothing else in this benchmark
reaches into `external/{hur_qcnn,wei_qcnn}/` source.
