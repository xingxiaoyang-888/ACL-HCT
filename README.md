# ACL-HCT: When Sampling Breaks Hierarchy

Initial research engineering for tangent-space correction of sampled Lorentz graph aggregation. **Real-network hierarchy collapse is a hypothesis, not a result.** The historical experiment matrix is not enabled.

## Install and reproduce

Python >=3.10; PyTorch >=2.0, NumPy and pytest. Install PyTorch separately using the [official version selector](https://pytorch.org/get-started/previous-versions/), then:

```bash
python -m pip install -e '.[test]'
python -m pytest -q
python -m acl_hct.smoke --device cpu --output logs/local-smoke.json
```

Tests are offline. The smoke runs eight optimization steps per method on a 40-node synthetic tree, plus exhaustive five-point frozen diagnostics. It has a hard 30-step ceiling. Training loss, synthetic parent-child radius order and a semantic branch distance gap are not held-out real-data results. Timings include Python loops and validation checks and do not establish production throughput or relative efficiency.

Without editable installation, use `PYTHONPATH=src python ...`. On PowerShell set `$env:PYTHONPATH="src"`.

## Mathematics and numerical scope

Coordinates are `(time, spatial...)`; Lorentz product is `-x0*y0 + sum(xi*yi)`; curvature is `-c`, c>0. Geometry supports FP32/FP64, broadcast batch dimensions and a final coordinate axis. Supported engineering domain: scaled radius from the coordinate origin <=3. Out-of-domain points raise errors. This origin-based numerical restriction is not a locality assumption or a global isometry-invariance claim. Validation synchronizes GPU execution; optimize only after correctness evidence.

`exp` projects ambient vectors to the tangent space. Series branches in `exp` and `log` have finite unused branches, including at zero. Exact distance is nonsmooth at coincidence; use squared tangent norms for differentiable zero-distance objectives. `normalize` rejects non-future-timelike inputs. Complete aggregation supports positive weights; correction is **equal-weight only**.

`correct(sample, population_size, c, method)` sees selected messages and actual visible count N only, never unsampled hidden states or hierarchy labels:

- `none`: normalized Lorentz mean.
- `third`: candidate local third-central-moment correction, with the finite-population coefficient in the research note. Requires fixed vectors, uniform sampling without replacement, equal weights and small local spread. No universal improvement is claimed.
- `jackknife`: cached-sum leave-one-out comparator with finite-population factor `(N-k)/N`; an approximate geometric comparator, not an exact unbiased estimator. Both corrections cost O(kd).

k=N returns the original output exactly; k=1/2 return it with fallback recorded. Step norm is capped at 0.1; clipping/fallback rates and step norms are recorded. Lower bias need not imply lower MSE. Batched metadata records whether any member clipped; the model calls one neighborhood at a time.

Sampling uses an explicit CPU `torch.Generator`; full fanout preserves input order. GNN neighbor IDs are deduplicated. An empty neighborhood explicitly uses self; other self loops must be explicit. Semantic truth stays separate from message adjacency. The synthetic training demo uses the known tree graph and all-node depth regression; it is not link prediction or a held-out task benchmark.

## Data

```bash
PYTHONPATH=src python scripts/acquire_data.py --download-only
# Parse and audit on a scheduled CPU allocation on HPC:
PYTHONPATH=src python scripts/acquire_data.py --audit-only
```

Pinned sources: [Princeton WordNet 3.0](https://wordnetcode.princeton.edu/3.0/) and [NLM MeSH 2026 XML](https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/). [WordNet license](https://wordnet.princeton.edu/license-and-commercial-use) and [MeSH terms](https://www.nlm.nih.gov/databases/download/terms_and_conditions_mesh.html) apply; raw data are not redistributed. Curl resumes interrupted transfers. Manifests record URL, version, retrieval/audit time, bytes, SHA256 and structure. Hashes are local integrity records, not upstream signatures. Amazon reviews are not downloaded.

WordNet parses noun semantic hypernym and instance-hypernym links, parent->child. MeSH parses immediate tree parents collapsed onto descriptor IDs. Multiple parents and isolates are preserved; the DAG audit detects cycles instead of fabricating depth. Depth is longest root path. `split_relations` is a conservative **small-graph** helper: held-out pairs and reverse pairs are excluded from training; held-out relations still inferable through training paths in either direction are quarantined. This repeated reachability implementation needs a scalable redesign before full-corpus use. No real-data benchmark split is claimed yet.

The downloaded official MeSH 2026 release assigns three tree positions to both D047991 and D048013. The parser retains both descriptors but quarantines links touching these ambiguous positions, and lists the collisions and excluded-link count in the manifest. It never silently chooses an owner. Reported MeSH edges therefore describe this explicitly filtered view.

## HPC isolation

Prepare dependencies and data before allocation. Supply authorized account, QOS, partition, CPUs, memory, output path and time explicitly to Slurm. Start with a CPU run of `scripts/smoke.slurm`; after audit passes, use `ACL_DEVICE=cuda`, one L40 and <=30 minutes. Private resource commands and JobIDs belong under `.local/`. Preserve Slurm GPU binding. The runner records versions, tests and smoke metrics, then exits; it contains no sweep or automatic retry campaign.

## Layout and evidence

- `src/acl_hct/`: geometry, aggregation, model, parsers, diagnostics and smoke CLI.
- `tests/`: FP64/reference/gradient/sampling/leakage/integration checks.
- `scripts/`: official data acquisition and bounded Slurm runner.
- `reports/STATUS.md`: milestone status and reproducible evidence.
- `research/legacy/`: inherited independent NumPy reference, not experimental proof.
- `docs/research/`: hypotheses and bounded initial acceptance contract.

Raw downloads, private host inventory, job identifiers, caches and detailed scheduler logs remain untracked. Read the acceptance contract before extending scope.
