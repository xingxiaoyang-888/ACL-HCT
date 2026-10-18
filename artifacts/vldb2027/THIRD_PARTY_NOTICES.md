# Third-party source and data boundaries

## Native HGCN and HyboNet

The native trees are **not redistributed** in this package. No redistribution
license was verified in either pinned upstream revision:

| Model | Upstream | Revision |
|---|---|---|
| HGCN | https://github.com/HazyResearch/hgcn | `a526385744da25fc880f3da346e17d0fe33817f8` |
| HyboNet | https://github.com/chenweize1998/fully-hyperbolic-nn | `7ba8e1a06d0504971134edbb2520d75d9887ba7a` |

Obtain upstream code under the permissions applicable to your use. Providing
reconstruction instructions does not grant upstream redistribution permission.
`config/native-sources.json` gives the 19 HGCN and 30 HyboNet file allowlists,
upstream LF-normalized hashes and final accepted byte hashes. The supplied
`src/reconstruct_native.py` operates only on the reader's existing checkout.
It contains our short compatibility/observer transformations, not the native
function bodies. It never fetches or executes upstream code.

HGCN needs only deterministic UTF-8/CRLF representation. HyboNet additionally
removes one nonportable unused `os` import, exposes only RiemannianAdam in the
optimizer package initializer, and adds the historical optional aggregation
observer. The observer is inactive unless explicitly attached. Exact mixed
line endings in that five-line observer block are retained for byte identity.

## Installed scientific dependencies

PyTorch, PyTorch Geometric, DGL, Geoopt, NumPy, SciPy, NetworkX, scikit-learn
and other installed packages remain under their respective upstream licenses.
This package imports them; it does not bundle their wheels or source trees.
The recorded wheel/installed-version inventory is in
`environment/runtime-provenance.json`. The GAT scaffold and log-CE objective
are attributed in `config/arxiv-models.json` to the official DGL ogbn-arxiv
examples; the exported adapter is an explicit sampled-block adaptation and
does not claim to reproduce the official full-batch score.

## Data and checkpoints

No graph, features, task labels, predictions or checkpoint binary is included
in this source subtree. Checkpoint access and any separately released result
archive are described by the top-level artifact index. Hashes and input
schemas in this subtree are not dataset redistribution.

ogbn-arxiv is acquired from the official OGB archive; its official
documentation states the ODC-BY dataset license:
https://ogb.stanford.edu/docs/nodeprop/ . PubMed inputs are the original
Planetoid-format files at commit
`af6abf468a772cc055bec88c52e4d6f51b7c37e3` of
https://github.com/kimiyoung/planetoid . That repository has an MIT software
license; a separate PubMed-specific data license was not identified in the
recorded provenance. Do not infer one from the software license.

Historical source-audit exports reference Airport and Disease-NC. Their
original arrays and private campaign indices are not included here. The
historical fit summaries are development evidence, not an independent test.

