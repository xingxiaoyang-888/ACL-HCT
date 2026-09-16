# ACL-HCT research engineering

## Ownership and scope

- All local code, scripts, tests and project artifacts belong under `F:\ACL\_HGT`. The Codex saved project may start in `F:\ACL_HGT`; explicitly change working directory before writing or running project commands. Do not create a second implementation there.
- This repository studies sampling-induced hierarchical collapse and tangent-space correction in hyperbolic GNNs. Read `docs/research/启动与验收约定.md` before implementation. The research claim is a hypothesis, not an established result.
- Following the initial release, the dedicated code-quality task owns source, tests, README and packaging changes. The original experiment task owns remote operations and historical run/data evidence, and has handed off source editing. The parent writing/supervision task reviews evidence and coordinates scope. Do not have two tasks modify the same files concurrently. Maintain `reports/STATUS.md` and a separate code-quality report with reproducible commands, milestones, tests, failures and next steps; preserve provenance of historical results.
- On 2026-09-16 the user explicitly authorized proceeding through the proposed sequence: E0 closeout, E1 controlled mechanisms, real-data baseline preparation, E2/E3 frozen diagnostics, then E4/E5 intervention and training comparisons. Follow `docs/operations/分阶段执行计划.md`. The earlier initial-smoke-only stop is superseded by this staged authorization. Use bounded, measured runs and scientific readiness gates; do not launch the historical 455-run Cartesian matrix automatically. Routine progression through valid stages does not require asking the user again.

## Isolation and resource use

- Read the provided private server guide under `.local/服务器指南.md`; it is reference material, not permission to touch other projects. Do not commit it.
- Use a NEW ACL-only remote directory and its own environment, caches, downloads and logs. Never mutate, inspect private files unnecessarily, stop, or attach training inside the CVPR project.
- Every GPU job must have a fresh, separately recorded Slurm allocation. Preserve Slurm's `CUDA_VISIBLE_DEVICES`; never select host-global GPU IDs. Query the allocation and actual device before work.
- Start with at most ONE GPU and a bounded short job (initial smoke test at most 30 minutes). The group's maximum of eight GPUs is a ceiling, not permission to request all eight. Preserve the two GPUs the user reserves for CVPR; queue safely if resources are not available.
- Login nodes are for editing, transfers and job submission, not training or large compute. Prepare downloads and binary dependencies before reserving a GPU. Compilations or substantial CPU tests must use appropriate scheduled resources.
- Do not change shared CUDA/Conda, scheduler configuration, CVPR environments, or shared GPU settings. Never run broad `pkill`, `killall`, `scancel -u`, resets or forceful resource reclamation. Cancel only project-owned job IDs recorded in the ACL manifest when necessary.
- Actual hardware in the guide is L40 48GB, not L40S. Record the measured hardware rather than relabeling it. Record actual GPU hours separately from hypothetical budgets.

## Research and quality

- Implement a small understandable PyTorch baseline before optional graph-library optimizations. Geometry FP32 training and CPU FP64 references; do not indiscriminately apply mixed precision to Lorentz products, exp/log maps or inverse hyperbolic functions.
- Meaningful tests must cover manifold/tangent constraints, log-exp consistency, gradients and finite values, exact full-neighborhood behavior, valid uniform sampling, correction fallbacks for small samples, dataset split leakage and smoke pipeline reproducibility.
- Frozen aggregation bias, variance, MSE, hierarchy metrics and trained task performance are different quantities. Do not substitute radius changes for semantic hierarchy damage, or toy checks for real-network evidence.
- Candidate corrections must state their aggregation/sampling/locality assumptions. Track clipping/fallback rates. Do not claim an estimator is unbiased or universally effective based solely on a Taylor approximation.
- Use official data sources, hashes and explicit source/version/license records. Keep raw downloads, credentials, environment packages and checkpoints out of git. Tests should not silently download data.
- Prefer auditable manifests and machine-readable summaries over manually selected numbers. Record failures. Performance comparisons must use matched resource conditions; concurrent jobs are not valid uncontended timing evidence.

## Git and status

- Remote: `https://github.com/xingxiaoyang-888/ACL-HCT.git`. Preserve existing changes; no force pushes, destructive resets or unrequested branches.
- Check staged paths/content for credentials, server access details, data dumps and generated binaries before each push. Commit/push tested source, documented setup, public-safe design material and concise results summaries only.
- Private connection inventory, scheduler outputs, local path mappings and job IDs can be stored under `.local/` and summarized to the user. Never echo private keys or tokens.
- Keep the status report honest: distinguish planned, implemented, locally tested, remotely tested, queued and completed. If blocked, complete independent work and report the precise missing dependency.
