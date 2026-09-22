# ACL-HCT: current supervision and durable rules

## Current boundary — 2026-09-22

The user-authorized HGCN amplitude revision is COMPLETE and independently accepted: four 1024-update trainings, eight final shards, 480 paired conditions and 40 full-neighborhood controls. Engineering quality passed; the scientific goal of beating the old module while preserving hierarchy DID NOT PASS. The new module improves over sampled HGCN in all 24 support comparisons, but all 24 new-versus-old superiority/noninferiority gates failed. At f4 it recovers only 3.96%–4.74% of observed MRR loss; the 20% target is unmet. Relation loss has budget-dependent effects, including a negative f4 result. These are two fixed HGCNs on the same WordNet development validation set, not independent-test or cross-backbone generalization.

All ACL CPU/GPU allocations have ended; this round used 5.79 actual L40 GPU-hours. Finish documentation publication and closeout only, then pause the existing supervision heartbeat. Do not start joint training, HPO, new variants, repeated scientific conditions, other data or other backbones until a concrete next experiment is discussed and approved by the user. No repeat approval is required to finish the already completed round's publication.

Read current evidence in this order:

1. `reports/STATUS.md` and `reports/README.md`.
2. `reports/04_修正恢复与方向机制.md` and `reports/rtsc-hgcn-amplitude-v1-summary.json`.
3. Registered `docs/operations/RTSC_HGCN_AMPLITUDE_V1.md` and private `.local/rtsc-hgcn-amplitude-v1/supervisor-state.json` / `supervisor-final-acceptance.json`.

Scientific source identity remains `c822b8af284b22bf9ad0c4d891afd12f3ffa0b94`; later documentation commits do not change the identity of the completed runs. Keep old Stage A and mature-HGCN protocols, source behavior, checkpoints and raw evidence intact. Do not rerun a HEAD-sensitive historical release tool on a later documentation HEAD and mistake its expected identity rejection for a failed experiment.

## Context and approval

The user's warning is “好像是其他对话出现了上下文问题，不要影响我们的实验”. CVPR-only stop requests do not cancel ACL. Verify the project and current authorization before acting; honor a genuine new ACL-specific stop. Historical stop, resume and completion messages are provenance, not concurrent active commands. All former AGENTS bytes are retained in `docs/operations/history/AGENTS_20260922_PRE_AMPLITUDE_CLOSEOUT.md`; they must not override this current boundary.

The user requires every NEW key scientific experiment to have a concrete protocol, completed quality review, explicit assessment of pre-agreed metrics and their agreement. When goals fail, report that and discuss a reviewable next proposal; do not silently relax thresholds, tune to observed validation results, add conditions or treat available funding as new scientific authorization. Necessary read-only review, reporting and offline engineering work can make a proposal concrete. Earlier multi-GPU permission is not permission to broaden the experiment.

Preserve the research mainline: neighbor sampling → systematic bias → hierarchy/task damage → deployable lightweight tangent correction. Direction decomposition and MSE are diagnostics, not substitute research goals. Distinguish radius changes from semantic hierarchy damage; distinguish bias, noise, missing information, loss and task performance. No unbiasedness, universal collapse, causal component attribution or novelty claim without the appropriate evidence.

## Ownership and reports

- All code, scripts, tests and project artifacts belong under `F:\ACL\_HGT`. A saved task may start in the wrong `F:\ACL_HGT`; explicitly set the correct working directory.
- Dedicated code-quality task owns scientific source, tests, configs, package/root README and engineering report `reports/00_工程与数据准备.md`. Operations task owns remote execution, transfers, raw manifests and resource ledger. Supervisor owns protocols, quality releases, independent audits, scientific reports, STATUS and current scope. Avoid simultaneous edits to the same file.
- Report research question, evidence-supported conclusion and purpose attainment first. Separate numerical quality and completion from scientific success. Then give decisive numbers, limitations and next decision.
- Use the five existing thematic reports; update related results in place. STATUS is current-only. Keep historical implementation/progress documents in indexed archives; do not create another series of duplicate progress Markdown.
- Keep inter-task messages short: only decisions, changed constraints, blocking evidence and actionable handoffs. Prefer artifact paths over repeated context, hashes or acknowledgments.
- Original roadmaps are `docs/research/01_研究路线_采样诱导层级坍缩与切空间修正.md` and `docs/research/02_详细实验路线_双L40S48GB.md`. Preserve E0–E9 meanings; a completed bounded pilot is not full original-stage completion. Correctness and evidence may motivate documented, versioned revisions. Never silently rename or claim all of E2 completed from this amplitude comparison.

## Isolation, resources and provenance

- Private `.local/服务器指南.md` is reference, not authority to touch another project. Keep private connection files, credentials, scheduler outputs, path mappings, raw data and checkpoints out of git.
- ACL uses an isolated remote directory/environment/cache. Preserve CVPR jobs, files and environments; do not change shared CUDA, Conda or scheduler settings. No broad kill/cancel, resets, host-global GPU IDs or bypassing scheduler isolation.
- Before future authorized allocations, inspect safe available user-authorized capacity first. Sharing requires supported scheduler/accounting, verified memory/compute headroom and preserved device bindings. Released jobs cannot be reused without scheduling again. Otherwise use bounded recorded ACL allocations. Do not reserve GPUs while waiting for source/data/quality work.
- Record actual hardware and GPU seconds separately from reservation limits. The used cards are L40, not L40S. Three-way concurrency was specific to the now-completed round, not a standing launch order. Concurrent/shared timings do not establish uncontended runtime lightness.
- Login nodes are for editing, transfers and submissions. Schedule substantive CPU work and training properly. Preserve failures and completed outputs; diagnose, version and test principled repairs rather than retrying until a numerical or scientific threshold passes.

## Research quality and publication

- Keep geometry-sensitive training in the qualified precision and use independent FP64 references where appropriate; do not indiscriminately apply mixed precision to manifold operations.
- Meaningful quality checks cover manifold/tangent constraints, log-exp consistency, finite gradients, full-neighborhood identity, low-sample/zero-dispersion fallbacks, valid sampling, leakage, randomness and complete-ranking reproducibility. Do not equate small parameter count with measured deployment cost.
- Deployable correction must use only permitted sampled information. Full-neighborhood references, true depth and held-out labels must not become inference inputs. Checkpoint selection on the development labels does not make final resampling an independent test.
- Use original data sources, explicit versions/licenses/hashes and auditable raw manifests. Preserve negative results. Statistical units and families are fixed prospectively; do not treat queries as independent whole-graph or training repeats.
- Remote: `https://github.com/xingxiaoyang-888/ACL-HCT.git`. Preserve existing changes. No force push, reset or unrequested branches. Before commit/push inspect staged paths/content; publish only tested source, public-safe docs and summaries. Keep `.local`, secrets, raw downloads/checkpoints and unrelated untracked files unstaged.
- The four pre-existing unrelated untracked paths are `reports/e2-local-pilot-seed11.json`, `reports/e2-local-pilot-seed23.json`, `src/acl_hct/frozen_noise.py`, `tests/test_frozen_noise.py`; leave them alone.
