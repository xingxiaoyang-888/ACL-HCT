# Recorded scientific environment

The experiments used Linux x86-64, Python **3.11.7**, PyTorch **2.0.1+cu118**,
PyG **2.6.1**, DGL **1.1.3**, NumPy **1.24.4**, SciPy **1.10.1**, and Geoopt
**0.5.1**. The frozen final hardware stratum was **NVIDIA A800-SXM4-80GB**.
Earlier L40 experiments are separate evidence, not interchangeable timing rows.

`requirements-scientific.txt` is the recorded installed freeze, not a freshly
resolved lock. `runtime-provenance.json` records the installed versions and 42
wheel filenames, sizes and SHA256 digests. Wheels are not distributed here.
The exact Linux distribution/libc, final driver version and acquisition URL for
every inherited cached wheel were not retained in that receipt; those gaps are
not filled with guesses. No environment installation was performed for export.

A reader can create a separate Python 3.11.7 environment, obtain the named
wheels from their official package sources, check the recorded digests, and
install the requirements. For example, a wheel directory already populated
with the recorded files can be installed with:

```sh
python -m pip install --no-index --find-links /path/to/verified-wheels -r environment/requirements-scientific.txt
```

PyTorch's `+cu118` build is required by this recorded environment. DGL's bound
wheel is `dgl-1.1.3-cp311-cp311-manylinux1_x86_64.whl`: planning is CPU based;
this package does not substitute another DGL CUDA sampler. LABOR is the
installed DGL LABOR-0 implementation (`prob=None`, `importance_sampling=0`),
and the bridge rejects nonempty returned importance weights. Sampling and
planning thread scopes are part of the exported source.

Run each scientific cell in a fresh process with one allocated visible GPU.
The public CLI sets seven Torch CPU threads, FP32, disabled TF32, and disabled
cuDNN benchmarking, retaining the original scientific settings. The reader
must arrange resource isolation; the public wrapper does not certify an idle
host or continuous absence of contention. Timing on different hardware or
software is a new measurement and must be labeled accordingly.

The paper/figure build environment is separate and is documented with those
materials. Scientific requirements here do not include personal fonts.

