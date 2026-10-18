# Sampling mechanisms: reproducibility artifact

This artifact studies why neighbor sampling changes prediction quality and computation cost across GNN operators. It includes the completed frozen-model study and the historical evidence used in the study.

| Evidence | Models / data | Scope |
|---|---|---|
| Final evaluation | GraphSAGE, GAT, HGCN, HyboNet on ogbn-arxiv; GraphSAGE on PubMed | 15 fixed checkpoints, seeds 11/23/37, Uniform and LABOR, three cost budgets, four paired sampling repeats |
| Mechanism interventions | The same 15 checkpoints | Middle-budget sampled / local-restored / upstream-restored / jointly-restored comparisons |
| Historical mechanisms | HGCN and HyboNet on Airport and Disease-NC | 24 checkpoints, full and matched training; development evidence |
| Auxiliary correction cases | Historical studies | Positive and negative results retained separately from the final evaluation |

## Start here

- [Final per-seed results](results/final/per-seed.csv), [per-repeat results](results/final/per-repeat.csv), [three-seed summaries](results/final/across-seeds.csv).
- [Mechanism results](results/mechanisms/), including margins, corrections, harms and interactions.
- [Historical 24-checkpoint results](results/historical-task-a/).
- [Frozen budgets](results/final/frozen-budgets.csv): retain infeasible cost matches; equal fanout is not equal cost.
- [Checkpoint downloads](https://github.com/xingxiaoyang-888/ACL-HCT/releases/tag/vldb2027-artifact-v1) and [inference instructions](docs/reproduce-inference.md).
- [Validation record](validation.json): public source bindings, exact scalar-table reproduction and figure reconstruction.

## Reproduce the figures without a GPU

From the repository root, use Python 3.10 or later:

```bash
python -m pip install -r artifacts/vldb2027/environment/requirements-figures.txt
python artifacts/vldb2027/scripts/verify_artifact.py
python artifacts/vldb2027/scripts/render_figures.py --output build/vldb-figures
```

The renderer checks every plotted scalar against the published figure record before drawing the four empirical figures. Use a fresh output directory for each rendering.

## Run the frozen models

Follow [the inference guide](docs/reproduce-inference.md) for the pinned environment, original data preparation and checkpoint hashes. The scientific entry point, sampler, model adapters and training recipes are in [src](src/); configuration and public provenance are in [config](config/). Third-party native models are obtained from their original repositories at pinned commits and reconstructed with hash checks; see [third-party notices](THIRD_PARTY_NOTICES.md).

Final quality, timing and memory measurements use NVIDIA A800-SXM4-80GB GPUs. Historical L40 measurements remain separate. Timings compare the same complete target workload and include sampling, transfer and forward execution; exact/RAM setup and cache reuse have explicitly recorded cost scopes. Different hardware can reproduce the procedure but need not reproduce the measured times.

## Data and reporting conventions

- Four sampling repeats are averaged **within** a training seed. Across-seed dispersion is the sample standard deviation of the three seed means; repeats are not additional training seeds.
- Accuracy is stored as a fraction in scalar CSVs. Figures show percentages or percentage-point differences. Native margins and estimator errors are compared within a model/operator.
- Joint intervention effects are not the sum of independent local and upstream contributions. Corrected and newly harmed predictions are both reported.
- `results/raw-final/` contains the per-cell result JSON and gzip-compressed per-batch cost records. Execution-account paths and machine identifiers are omitted; numerical measurements are retained.
- `results/export-provenance.json` records original and public file hashes. `MANIFEST.json` verifies the files shipped in this artifact. Checkpoint archives have their own manifests.
- Historical fit/selection results remain development evidence. Official-test results are the fixed final campaign; no method or checkpoint was reselected from those scores.

The [archived aggregation scripts](scripts/archived/) preserve the original aggregation definitions. Portable versions change input/output bindings while retaining the numerical reductions. The plotting entry points above work directly with the published scalar exports and do not require the original cluster.

### Recompute scalar tables

The detailed JSON/cost records needed for the final deployment tables are already in this repository:

```bash
python artifacts/vldb2027/scripts/aggregate_final_scalars.py --inputs artifacts/vldb2027/results/scalar-inputs.json --output build/final-scalars
```

The two large result ZIPs are distributed as small release parts for reliable downloads. This standard-library helper resumes verified parts and checks each reconstructed ZIP against its original SHA256:

```bash
python artifacts/vldb2027/scripts/download_results.py --output downloads
python -m zipfile -e downloads/final-mechanism-arrays.zip downloads
python -m zipfile -e downloads/historical-task-a-records.zip downloads
```

Retain the extracted `raw/` directory structure. These are previously collected results, not new model executions. Use the recorded NumPy 1.24.4 and PyTorch 2.0.1 for exact historical reductions:

```bash
python artifacts/vldb2027/scripts/aggregate_mechanisms.py --inputs artifacts/vldb2027/results/mechanism-inputs.json --raw-root downloads --output build/mechanism-scalars
python artifacts/vldb2027/scripts/aggregate_task_a.py --inputs artifacts/vldb2027/results/task-a-inputs.json --raw-root downloads --output build/task-a-scalars
```

These commands do not train models or generate new predictions. Historical Task A files contain the original development targets and predictions; the final mechanism files contain stored correctness, margins and estimator errors.
