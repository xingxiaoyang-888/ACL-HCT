# ACL-HCT: current supervision and durable rules

## Current authorization — 2026-09-23

The user approved the proposed next experiment with “可以，进行实验”: matched HGCN joint training using the OLD three-direction module, plus plain HGCN and a simple single-direction correction, each with task-only or task+relation supervision. This is SIX conditions from EACH of the two existing original HGCN best checkpoints, twelve 1024-update training runs; backbone and original task head train in every arm, modules start at zero output. Same data, sampling, batches and selection opportunity; f4 training, f4/f8/f16 paired evaluation. The original mainline remains sampling damage and deployable hierarchy/task recovery.

This authorization supersedes the previous completed-round stop ONLY for the discussed joint-training comparison and its necessary engineering, quality/probe, matched-cost measurement, final evaluation and independent reporting. The supervisor freezes `docs/operations/RTSC_HGCN_JOINT_V1.md` before scientific execution. Code task owns implementation/tests; operations owns server preparation/execution; supervisor owns protocol, quality releases and independent acceptance. No GPU allocation until quality-ready code and a precise release exist. First one GPU for quality/probe, then up to three ready independent L40 jobs concurrently within recorded phase and total bounds; inspect safe existing capacity first and preserve CVPR. No repeated user approval is needed for this same authorized experiment after its quality gates pass.

Current phase: protocol and engineering preparation; no new GPU jobs yet. Read `reports/STATUS.md`, the new joint protocol and `.local/rtsc-hgcn-joint-v1/supervisor-state.json` for current state. The code task may add a separately versioned joint implementation; do not change the old Stage A/amplitude algorithms, configs, raw evidence or checkpoints. Additional data/backbones, HPO, unregistered variants and a further scientific round still need a concrete discussion and user agreement.

The prior amplitude comparison is COMPLETE, published at aaff8b5, with scientific source c822b8a: relative-to-old improvement failed, relative-to-S joint gains persisted, f4 MRR recovery 3.96%–4.74%, resources closed. Its former stop/frozen-HEAD instructions and paused automation describe that completed round, not the newly authorized joint work. Old private roots stay read-only. Historical results remain in reports/04 and their separate machine summaries.

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
- Record actual hardware and GPU seconds separately from reservation limits. The used cards are L40, not L40S. The current joint round permits up to three ready independent jobs after exact quality release; eight cards is a group ceiling, not a request to occupy them. Concurrent/shared timings do not establish uncontended runtime lightness.
- Login nodes are for editing, transfers and submissions. Schedule substantive CPU work and training properly. Preserve failures and completed outputs; diagnose, version and test principled repairs rather than retrying until a numerical or scientific threshold passes.

## Research quality and publication

- Keep geometry-sensitive training in the qualified precision and use independent FP64 references where appropriate; do not indiscriminately apply mixed precision to manifold operations.
- Meaningful quality checks cover manifold/tangent constraints, log-exp consistency, finite gradients, full-neighborhood identity, low-sample/zero-dispersion fallbacks, valid sampling, leakage, randomness and complete-ranking reproducibility. Do not equate small parameter count with measured deployment cost.
- Deployable correction must use only permitted sampled information. Full-neighborhood references, true depth and held-out labels must not become inference inputs. Checkpoint selection on the development labels does not make final resampling an independent test.
- Use original data sources, explicit versions/licenses/hashes and auditable raw manifests. Preserve negative results. Statistical units and families are fixed prospectively; do not treat queries as independent whole-graph or training repeats.
- Remote: `https://github.com/xingxiaoyang-888/ACL-HCT.git`. Preserve existing changes. No force push, reset or unrequested branches. Before commit/push inspect staged paths/content; publish only tested source, public-safe docs and summaries. Keep `.local`, secrets, raw downloads/checkpoints and unrelated untracked files unstaged.
- The four pre-existing unrelated untracked paths are `reports/e2-local-pilot-seed11.json`, `reports/e2-local-pilot-seed23.json`, `src/acl_hct/frozen_noise.py`, `tests/test_frozen_noise.py`; leave them alone.
