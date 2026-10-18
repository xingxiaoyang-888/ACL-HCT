# ACL-HCT · GNN Sampling Mechanisms

Code and experimental results for **From Neighbor Samples to Predictions: An Operator-Aware Study of GNN Quality and Inference Cost**.

We study how local sampling error and upstream representation error affect GNN predictions, and when sampling, exact computation or output caching offers the better quality–cost trade-off.

**[Paper](artifacts/vldb2027/paper/main.pdf) · [Supplement](artifacts/vldb2027/paper/supplement.pdf) · [Reproduction guide](artifacts/vldb2027/README.md) · [Checkpoint downloads](https://github.com/xingxiaoyang-888/ACL-HCT/releases/tag/vldb2027-artifact-v1)**

## Experiments

- **Models:** GraphSAGE, GAT, HGCN and HyboNet on ogbn-arxiv; GraphSAGE on PubMed.
- **Sampling:** Uniform and LABOR, three frozen cost budgets, seeds **11/23/37**, four paired repeats.
- **Mechanisms:** sampled, local-restored, upstream-restored and jointly-restored predictions; accuracy, margins, corrections and harms.
- **Systems:** complete-target latency, GPU memory, exact computation and paid RAM reuse.
- **Historical evidence:** 24 checkpoints across training regimes, plus auxiliary correction results. Development and final-test evidence remain separate.

## Quick start

```bash
git clone https://github.com/xingxiaoyang-888/ACL-HCT.git
cd ACL-HCT
python -m pip install -r artifacts/vldb2027/environment/requirements-figures.txt
python artifacts/vldb2027/scripts/verify_artifact.py
python artifacts/vldb2027/scripts/render_figures.py --output build/vldb-figures
```

Figure reproduction uses published results and requires no GPU. For data preparation, pinned dependencies, native model reconstruction and frozen inference, see [the inference guide](artifacts/vldb2027/docs/reproduce-inference.md).

## Repository

| Path | Contents |
|---|---|
| [`artifacts/vldb2027/src`](artifacts/vldb2027/src) | Frozen evaluation, samplers, model adapters and base-training recipes |
| [`artifacts/vldb2027/results`](artifacts/vldb2027/results) | Complete per-seed/per-repeat results, budgets, mechanism and cost records |
| [`artifacts/vldb2027/paper`](artifacts/vldb2027/paper) | Manuscript, supplement, LaTeX, bibliography and figures |
| [`src/acl_hct`](src/acl_hct) | Earlier geometry diagnostics and correction implementations |
| [`tests`](tests) | Historical implementation tests |

The final campaign was measured on **NVIDIA A800-SXM4-80GB**; historical L40 results are reported separately. All positive and negative results are retained. Third-party models and data follow their original terms; see [notices](artifacts/vldb2027/THIRD_PARTY_NOTICES.md). Earlier prototype instructions are preserved in [README-legacy.md](README-legacy.md).
