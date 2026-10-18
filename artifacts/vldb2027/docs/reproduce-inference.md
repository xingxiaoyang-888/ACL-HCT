# Scientific source and reproduction routes

This source export separates **paper result inspection**, **new execution of
frozen models**, and **historical source audit**. It does not submit cluster
jobs, choose new checkpoints, search hyperparameters, resume a private
campaign, or run anything during download/import of a public entrypoint.

## What is included

| Location | Scope |
|---|---|
| `src/frozen/` | The accepted metadata-v4 17-file evaluation closure: 15 files byte-identical, one provenance-comment edit in `geometry.py`, and a portable replacement of the private `runner.py`. All scientific estimator, native operator, intervention, scoring and timing functions are unchanged. |
| `src/sampler/` | The accepted Uniform-WOR planner and its thread/RNG helpers. LABOR-0 calls installed DGL 1.1.3. |
| `src/training/arxiv_base/` | Original accepted 12-file source closure for the nine HGCN/HyboNet/GAT plain 500-epoch baselines; only the controller and one provenance comment are adapted. |
| `src/training/arxiv_sage/` | Original GraphSAGE architecture, training, planner, runtime and arxiv preparation. |
| `src/training/pubmed_sage/` | Original five-file fixed GraphSAGE/PubMed training and preparation. |
| `src/data/` | Accepted label-alignment and target-order source, called through the new explicit portable preparation interface. |
| `src/history/source_audit/` | All ten scientific files underlying the already collected 24-checkpoint history, with mapped storage names and an explicit nonexecution gate. |
| `config/` | Fixed plan, model settings, five original label-free budget locks, 15 checkpoint identities, data/native provenance, examples and source manifest. |

`config/source-manifest.json` maps every exported member to its original and
published SHA256 plus the reason for change. New portable files have no
original hash. Original source docstrings sometimes retain their historical
“source-only/unexecuted” wording; these are provenance snapshots, not a new
claim that the paper campaign was unexecuted. Public wrapper validation is
separate from the already accepted scientific checks.

The paper completed all 15 frozen final cells. This portable interface is a
new path/controller adaptation; consult the public validation record for its
actual validation status. It is not a claim that full data/model execution
was repeated during export. Exact performance reproduction also requires the
specified inputs, original checkpoints, native trees and recorded runtime.

## 1. Inspect the published source and results first

Commands below assume the working directory is `artifacts/vldb2027`.

```sh
python src/verify_package.py
python src/run_frozen.py --help
python src/run_training.py --help
python src/prepare_inputs.py --help
```

These commands require only the Python standard library. The result archive
and its scalar-only table/figure entrypoints are indexed by the artifact's
top-level README. They are the cheapest route to inspect the findings and
rebuild manuscript figures, without any checkpoint, task labels or GPU.

## 2. Provide the recorded environment and native dependencies

Follow `environment/README.md`. For HGCN/HyboNet, obtain the exact revisions
listed in `THIRD_PARTY_NOTICES.md`, then reconstruct outside this package:

```sh
python src/reconstruct_native.py --model hgcn --upstream /path/to/pinned-hgcn --output /path/to/native-hgcn
python src/reconstruct_native.py --model hybonet --upstream /path/to/pinned-fully-hyperbolic-nn --output /path/to/native-hybonet
```

The second command receives the repository root, not its `gcn/` subdirectory.
All 49 allowed source files are checked before export; incompatible upstream
bytes are rejected. The original trees are left unchanged. No native source
is downloaded, imported or licensed by these commands. Euclidean models do
not need either native tree.

## 3. Acquire and prepare official data

`config/data-provenance.json` records exact URLs, archive/member hashes and
the pinned Planetoid revision. Download separately through those official
routes; this package contains no data. Use the recorded raw files, not a new
library loader with a possibly different split or normalization.

```sh
python src/prepare_inputs.py arxiv --raw-root /path/to/extracted/arxiv --output /path/to/data/arxiv
python src/prepare_inputs.py pubmed --raw-root /path/to/planetoid/data --output /path/to/data/pubmed
```

arxiv uses 169,343 nodes, 128 FP32 features and the unique symmetric union of
the directed nonself edges. Features retain the original values. Its official
90,941 train and 29,799 validation IDs are unchanged. The validation set is
hash-ordered using `icml-ca-v1:valid-split:` and split into 14,899 development
and 14,900 historical-confirmation nodes; development is re-ordered with
`icml-ca-v1:dev-split:` into 9,932 fit and 4,967 gate nodes. Batch ordering
uses `icml-ca-v1:batch:`. Preparation traverses the compressed label stream
but parses only train/validation rows. It does not score test.

PubMed uses 19,717 nodes, 500 FP32 row-L1-normalized features, 88,648 canonical
nonself directed entries and the historical native-loader split: train IDs
0–59, **1,000 validation IDs 60–1059**, and the original 1,000 test IDs.
Validation is hash-split into 500 fit and 500 development nodes. This is not
the often-used 500-validation-node PyG split. Preparation deserializes the
`y` and `ally` containers but selects only the first 1,060 label rows; it
does not open `ind.pubmed.ty`.

The prepared public NPZ has exactly `x` (FP32), `indptr`/`indices` (int64),
`values` (FP32 ones), and `diagonal` (FP32 zeros). CSR is sorted, distinct and
nonself; native adapters own self-message semantics. Development label NPZs
contain only aligned `ids` and `y` int64 arrays. Metadata/preparation JSONs
bind file hashes. Different NPZ serialization or runtime versions can change
file bytes; disclose those differences rather than replacing recorded hashes
and calling them original artifacts.

## 4. Obtain the fixed checkpoint, or regenerate a baseline

`config/checkpoints.json` lists all original 12 arxiv and three PubMed CP
hashes, selected epochs, schemas and canonical metadata hashes. The separate
[checkpoint archive](https://github.com/xingxiaoyang-888/ACL-HCT/releases/download/vldb2027-artifact-v1/checkpoints.zip)
contains these 15 original byte-identical checkpoints plus 24 accepted
historical capture exports. Extract it outside the source tree and compare
each member against `results/checkpoints-manifest.json` before use. Historical
exports preserve the accepted model states but are not the original training
checkpoint serializations; that distinction and both hashes are recorded.
The example frozen manifest uses
`../checkpoints/frozen/ogbn-arxiv/sage/seed11/label-free-weights.pt`, relative
to its location in `config/`. Adjust this to the chosen extraction directory.

For an independent baseline regeneration, copy and fill
`config/training-inputs.example.json`. It accepts exactly graph, IDs,
preparation, train labels and fit labels—never gate/test labels. Paths are
relative to that JSON (absolute reader-local paths also work); every file
requires its actual SHA256. Use `native_root: null` for SAGE/GAT and the
reconstructed model-specific path for HGCN/HyboNet. Then:

```sh
CUDA_VISIBLE_DEVICES=0 python src/run_training.py --inputs /path/to/train-inputs.json --output /path/to/new-training-output
```

One invocation trains one plain model and seed. The allowed matrix is four
arxiv models plus SAGE/PubMed, seeds 11/23/37. All runs use 500 epochs, batch
size 1,024, selection every five epochs, four fixed fit sampling streams,
and the earliest maximum summed correct count. arxiv training fanouts are
15/15/15; PubMed uses 4/4/4. Architectures/native losses/optimizers and GAT's
50-epoch learning-rate warmup are exactly in the exported fixed config and
original training source. No correction training is exposed by this CLI.

Original SAGE and later GAT/hyperbolic training use their distinct original
initialization/RNG namespaces; the wrapper does not “unify” these and change
the experiment. It emits a new `checkpoint-metadata.json` alongside weights.
New training may differ numerically across systems. It is a regenerated
baseline, not proof of byte-identical reproduction of the paper checkpoint.

## 5. Freeze inputs before final evaluation

Copy `config/inference-inputs.example.json` and fill all file references.
The default `--checkpoint-mode paper` checks original CP bytes and canonical
metadata and allows `checkpoint.metadata: null` (metadata is verified after
loading). For independently trained weights, select `regenerated` and copy
the exact metadata from the new training output. This mode is marked as an
independent reproduction in output; it never replaces the paper scores.

The five supplied budget locks are the original A800 label-free selections:

| Dataset/model | Uniform low/middle/high | LABOR low/middle/high | Boundary |
|---|---|---|---|
| arxiv/HGCN | 5/15/30 | 8/12/36 | Discrete cost-grid lock, not exact final equal time. |
| arxiv/HyboNet | 5/15/30 | 5/15/36 | Same scope. |
| arxiv/SAGE | 5/15/30 | 5/15/36 | Low LABOR point over budget; retained as infeasible. |
| arxiv/GAT | 5/15/30 | 5/15/30 | Same scope. |
| PubMed/SAGE | 1/4/8 | 6/6/6 | Same chosen LABOR point at all three tiers; nonmonotone measured cost retained. |

To reproduce the original fixed comparison, use those locks. If calibrating
new hardware, first use seed11 with `targets`, `labels` and `budget_lock` all
null, and run `--stage calibrate`; the exact original eight-batch, 24-request
median/grid procedure runs without labels. Freeze its result before final
scoring. A new calibration is a new hardware study, not an update to the
paper's accepted budget map.

Only after this freeze should a reader prepare test labels:

```sh
python src/prepare_inputs.py test-labels --dataset ogbn-arxiv --raw /path/to/arxiv.zip --public-ids /path/to/data/arxiv/public/ids.json --output /path/to/test/arxiv --confirm-frozen-evaluation
python src/prepare_inputs.py test-labels --dataset PubMed --raw /path/to/planetoid/data --public-ids /path/to/data/pubmed/public/ids.json --output /path/to/test/pubmed --confirm-frozen-evaluation
```

These commands align original labels and construct the frozen hash-order
targets; they do not score predictions. arxiv has 48,603 targets in 380
batches (379×128+91), PubMed 1,000 in eight (7×128+104). The common prefix
is `vldb-sampling-mechanisms-v1:official-test-target:` followed by dataset
and node ID. Each repeated comparison processes that same complete workload.
PubMed test had historical exposure; it is not called a never-observed dataset.

Run one committed cell in a fresh process:

```sh
CUDA_VISIBLE_DEVICES=0 python src/run_frozen.py --stage final --inputs /path/to/frozen-inputs.json --output /path/to/new-final-output --checkpoint-mode paper --confirm-frozen-evaluation
```

The output directory must not exist. No overwrite, resume, automatic retry,
CP reselection or test-driven budget change is implemented. The public
interface records an input commitment; it cannot enforce a reader's global
once-only study policy across unrelated output directories. That prospective
policy remains the researcher's responsibility.

## Scientific and cost contracts

Uniform samples fixed-count nonself neighbors without replacement. LABOR-0
has random realized degree, no returned importance weights, and a zero
nonself contribution on empty samples. SAGE divides by the realized sample
count; HGCN/HyboNet use their fixed-self plus HT nonself sums; GAT retains
sampled-set softmax with one self edge. Local estimator properties must not
be promoted to unconditional unbiased model-output claims.

The middle-tier mechanism uses one whole-graph sampled or full prefix per
layer, then four final-layer corners: sampled, local restored, upstream
restored and jointly restored. This diagnostic is distinct from deployment
dependency cones and does not supply deployable oracle states. The original
four paired repeats and negative effects remain intact. Local L2 and native
margin scales are interpreted within a model, not across models as curvature
effects.

Deployment costs include planning, preparation, feature indexing/transfer,
forward and return of requested outputs. Full compares against the same
complete target workload. Format/model/public-graph setup is reported
separately. Full cache build includes forward plus D2H; target gather is
separate. Paid RAM cost is setup + build + q×RAM-gather; recomputation is
setup + q×full-workload. The unchanged evaluator emits q=1/10/100 and
reserved memory as process-history observations, not an isolated cold
minimum. The wrapper synchronizes at boundaries but does not establish
exclusive hardware or continuous absence of contention. Do not mix fresh
performance numbers with the original A800 stratum without labeling them.

## Historical 24-checkpoint evidence

The source-audit directory contains all ten accepted scientific files,
including the actual collector and its interventions, trace, sampler,
views, training utilities, inputs/checkpoint/runtime/contract modules. The
source manifest names their exact originals. Storage references have been
mapped to nonoperational `public-inputs/history/...` placeholders, and a
top-level guard intentionally prevents execution. The original functions
remain visible for review; this is **not a runnable historical collector**.

Use the published historical fit records and Task A scalar aggregation
route for reproducing those tables. The old private registration, protected
input indices and controller are not a public runtime interface. A future
historical collector reproduction would require public Airport/Disease
preprocessing/split manifests and guarded-input/trace contracts, plus
an independently validated portable path layer. Do not infer those inputs
from the new arxiv/PubMed pipeline or silently rebuild an old campaign.

History is HyboNet/HGCN × Airport/Disease-NC × full/central-matched training
× seeds 11/23/37. Task A uses 16 paired fit streams per CP. HyboNet has five
Airport and three Disease message layers; historical HGCN has one, so its
upstream-restored branch is structurally aliased to sampled. It must not be
confused with the new three-layer arxiv HGCN. The original native dependencies
are subsets of the same two pinned trees above. Fit summaries, degenerate
cells and adverse interventions remain development evidence in full.

## Remaining public-material boundaries

This source subtree excludes private scheduler/account/host identities,
ledgers, releases, data/label/CP binaries and personal fonts. The separately
published checkpoint archive is indexed above. Native upstream permission
and official dataset acquisition remain separate from having code. Source
review, portable CPU validation, scalar reproduction and actual GPU
re-execution are different validation levels; only completed levels may be
claimed in the top-level validation record. No new experiment was executed
to prepare this export.
