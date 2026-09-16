# S1B equivalent CPU chunk packing

The historical fce017259a6bdc513f21b31304f096239e357a75 L40 pilot measured 8.297 s/step, including 0.928 s planning (11.19%). The remainder includes host work and synchronization and is not a kernel-only measurement. These timings motivate this patch; they do not measure its speedup.

`make_plan` copies full neighborhoods without allocating an index tensor and without consuming RNG. `aggregate_plan` preserves the original chunk boundaries and neighbor order, builds padded IDs, masks and population sizes on CPU, then transfers each complete tensor to the points device. Sampling, model, precision, mathematical reduction and correction/fallback rules are unchanged.

The independent retained old implementations in `tests/test_chunk_packing.py` compare complete plans and RNG states over repeated draws, exact outputs and diagnostics, and FP32/FP64 input gradients for none/third/jackknife. Fixtures include empty rows, full neighborhoods, padding and multiple chunks.

Validation: combined chunk-packing, benchmark-preparation and ranking checks passed (27 passed, 6 CUDA cases skipped). After strengthening the fallback assertion, the dedicated suite passed again (12 passed, 6 skipped). CUDA cases require an allocated Slurm run. No GPU speedup is claimed until an experiment job measures this revision.
