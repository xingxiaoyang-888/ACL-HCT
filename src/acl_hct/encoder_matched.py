"""Original Lorentz encoder with each aggregation replaced by its self message.

The inherited constructor and scorer preserve the original parameter names,
initialization, smooth radial map, exp/log maps and relation head. No graph,
neighbor representations, added activation or normalization enters this encoder.
"""
import torch

from .backbone import LorentzMeanNetwork
from .geometry import log, origin_like


class SelfMessageLorentzNetwork(LorentzMeanNetwork):
    def _features(self, features):
        parameter = self.layers[0].linear.weight
        if (not isinstance(features, torch.Tensor) or features.ndim != 2
                or features.shape[0] < 1 or features.shape[1] != parameter.shape[1]
                or features.dtype != parameter.dtype or features.device != parameter.device):
            raise ValueError('nonempty feature matrix must match model dimension/dtype/device')

    def encode(self, features):
        """Encode all supplied entities, preserving their input order."""
        self._features(features)
        inputs = features
        for layer in self.layers:
            points = layer.messages(inputs)
            # Empty neighborhoods in the original aggregate return this exact
            # self message. Keep both original log calls, including layer two.
            inputs = log(origin_like(points, self.c), points, self.c)[..., 1:]
        return points

    def forward(self, features, queries, *, unique_entities=True):
        self._features(features)
        if (not isinstance(queries, torch.Tensor) or queries.ndim != 2
                or queries.shape[1] != 2 or queries.shape[0] < 1
                or queries.dtype != torch.long or queries.device != features.device
                or ((queries < 0) | (queries >= len(features))).any()
                or type(unique_entities) is not bool):
            raise ValueError('nonempty device int64 [Q,2] queries and boolean path flag required')
        if not unique_entities:
            return self.score(self.encode(features), queries)
        # No cross-entity operations: retain repeated query occurrences via the
        # inverse map. Changed GEMM row counts require the registered FP32 gate.
        ids, inverse = torch.unique(queries.reshape(-1), sorted=True, return_inverse=True)
        return self.score(self.encode(features[ids]), inverse.reshape(-1, 2))
