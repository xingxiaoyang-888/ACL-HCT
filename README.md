# ACL-HCT · GNN Sampling Mechanisms

Code and experimental results for studying how neighbor sampling changes GNN prediction quality and inference cost.

**[Reproduction guide](artifacts/vldb2027/README.md) · [Per-seed results](artifacts/vldb2027/results/final/per-seed.csv) · [Mechanism results](artifacts/vldb2027/results/mechanisms/) · [Checkpoints and raw results](https://github.com/xingxiaoyang-888/ACL-HCT/releases/tag/vldb2027-artifact-v1)**

## Experiments

- **Models:** GraphSAGE, GAT, HGCN and HyboNet on ogbn-arxiv; GraphSAGE on PubMed.
- **Sampling:** Uniform and LABOR, three fixed cost budgets, seeds 11/23/37 and four paired repeats.
- **Measurements:** accuracy, margins, corrected/introduced errors, latency, GPU memory and exact/cache reuse costs.
- **Historical evidence:** 24 checkpoints across training regimes and auxiliary correction results, kept separate from the final-test evaluation.

## Quick start

```bash
git clone https://github.com/xingxiaoyang-888/ACL-HCT.git
cd ACL-HCT
python -m pip install -r artifacts/vldb2027/environment/requirements-figures.txt
python artifacts/vldb2027/scripts/verify_artifact.py
python artifacts/vldb2027/scripts/render_figures.py --output build/vldb-figures
```

The plots use published results and require no GPU. See the [inference guide](artifacts/vldb2027/docs/reproduce-inference.md) for frozen model evaluation, data preparation and pinned dependencies.

## Contents

| Path | Contents |
|---|---|
| [`artifacts/vldb2027/src`](artifacts/vldb2027/src) | Frozen samplers, model adapters, evaluation and base-training recipes |
| [`artifacts/vldb2027/results`](artifacts/vldb2027/results) | Per-seed/per-repeat results, budgets, mechanism and cost records |
| [`artifacts/vldb2027/config`](artifacts/vldb2027/config) | Fixed scientific settings and provenance |
| [`artifacts/vldb2027/scripts`](artifacts/vldb2027/scripts) | Downloads, verification, aggregation and plotting |
| [`src/acl_hct`](src/acl_hct) | Historical geometry diagnostics and correction implementations |

Final measurements use NVIDIA A800-SXM4-80GB; historical L40 measurements remain separate. Positive and negative results are retained. Third-party models and data follow their original terms; see [notices](artifacts/vldb2027/THIRD_PARTY_NOTICES.md).
