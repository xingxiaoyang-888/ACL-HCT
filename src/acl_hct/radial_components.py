"""Fixed FP64 radial/orthogonal mean-removal fields, no random stream or model.

Undefined radial directions leave BOTH interventions unchanged and retain the
unapplied bias explicitly. Finite exponential moves need not preserve radius;
task effects of R/T are not additive contributions to the old O effect.
"""
import copy
import hashlib

import torch

from .geometry import check_point, curvature, dot, exp, log, norm2


def identity(value):
    raw = value.detach().cpu().contiguous().numpy()
    return {'shape': list(raw.shape), 'dtype': raw.dtype.str,
            'data_sha256': hashlib.sha256(raw.tobytes()).hexdigest()}


class RadialBiasIntervention:
    """Projection at original p using a single fixed p[root] as radial anchor."""

    @torch.no_grad()
    def __init__(self, base, bias, floor, root, c=1.):
        self.c = curvature(c)
        if (not isinstance(base, torch.Tensor) or base.dtype != torch.float64
                or base.ndim != 2 or len(base) == 0):
            raise ValueError('complete nonempty FP64 reference required')
        check_point(base, self.c)
        if type(root) is not int or not 0 <= root < len(base):
            raise ValueError('fixed original root index required')
        self._base = base.detach().clone()
        self._field(bias, 'bias')
        if (not isinstance(floor, torch.Tensor) or floor.dtype != torch.float64
                or floor.device != base.device or floor.shape != (len(base),)
                or not torch.isfinite(floor).all() or (floor < 0).any()):
            raise ValueError('complete original FP64 numerical floor required')
        self._bias = bias.detach().clone()
        self._root = root
        toward = log(self._base, self._base[root], self.c)
        length = norm2(toward).sqrt()
        threshold = torch.maximum(floor + floor[root], torch.full_like(floor, 1e-10))
        self._defined = length > threshold
        denominator = torch.where(self._defined, length, torch.ones_like(length))
        unit = -toward / denominator[:, None]
        unit[~self._defined] = 0
        radial = dot(self._bias, unit)[:, None] * unit
        orthogonal = self._bias - radial
        radial[~self._defined] = 0
        orthogonal[~self._defined] = 0
        unapplied = torch.zeros_like(self._bias)
        unapplied[~self._defined] = self._bias[~self._defined]
        self._fields = {'R': radial, 'T': orthogonal}
        self._unit = unit
        self._unapplied = unapplied
        self._length = length
        self._threshold = threshold
        self._moving = {name: norm2(field) > 0 for name, field in self._fields.items()}
        for name, field in self._fields.items():
            self._field(field, name)
        self._field(unit, 'outward unit')
        if (not torch.allclose(norm2(unit)[self._defined], torch.ones_like(length[self._defined]), atol=1e-10, rtol=1e-10)
                or (dot(unit, orthogonal)[self._defined].abs() > 1e-9).any()
                or not torch.allclose(radial + orthogonal + unapplied, self._bias, atol=1e-12, rtol=1e-10)
                or not torch.allclose((norm2(radial) + norm2(orthogonal))[self._defined],
                                      norm2(self._bias)[self._defined], atol=1e-12, rtol=1e-9)):
            raise ValueError('fixed defined-direction decomposition gate failed')
        self._metadata = {
            'scope': 'exploratory fixed-model R/T diagnostic fields; no deployment estimator',
            'curvature': self.c, 'root_index': root,
            'undefined_rule': 'distance(p,p[root]) <= max(floor_i+floor_root,1e-10); R/T copy native S',
            'projection': 'outward u=-Log_p(p[root])/Lorentz norm; radial=<b,u>_L*u; orthogonal=b-radial on defined nodes',
            'undefined_count': int((~self._defined).sum()), 'defined_count': int(self._defined.sum()),
            'zero_component_counts': {name: int((~mask).sum()) for name, mask in self._moving.items()},
            'base': identity(self._base), 'bias': identity(self._bias), 'floor': identity(floor),
            'fields': {name: identity(field) for name, field in self._fields.items()},
            'unapplied_bias': identity(unapplied), 'outward_unit': identity(unit),
            'interpretation': 'orthogonality is at fixed tangent points; finite T may change radius; R/T effects need not add to O',
            'calibration_and_archive_provenance_verified_by_adapter': False,
        }

    def _field(self, value, name):
        if (not isinstance(value, torch.Tensor) or value.dtype != torch.float64
                or value.device != self._base.device or value.shape != self._base.shape
                or not torch.isfinite(value).all() or (dot(self._base, value).abs() > 1e-9).any()):
            raise ValueError(name + ' must be a complete finite FP64 tangent field at the same base')

    @property
    def fields(self):
        return {name: value.clone() for name, value in self._fields.items()}

    @property
    def defined(self):
        return self._defined.clone()

    @property
    def outward_unit(self):
        return self._unit.clone()

    @property
    def unapplied_bias(self):
        return self._unapplied.clone()

    @property
    def direction_resolution(self):
        return {'distance': self._length.clone(), 'threshold': self._threshold.clone()}

    @property
    def metadata(self):
        return copy.deepcopy(self._metadata)

    @torch.no_grad()
    def apply(self, sample64):
        if (not isinstance(sample64, torch.Tensor) or sample64.dtype != torch.float64
                or sample64.shape != self._base.shape or sample64.device != self._base.device):
            raise ValueError('complete FP64 sample at original reference shape/device required')
        check_point(sample64, self.c)
        error = log(self._base, sample64, self.c)
        points, offsets = {}, {}
        for name, field in self._fields.items():
            offset = error - field
            self._field(offset, name + ' offset')
            moving = self._moving[name]
            moved = sample64.detach().clone()
            if moving.any():
                moved[moving] = exp(self._base[moving], offset[moving], self.c)
            points[name], offsets[name] = moved, offset
        return {'points': points, 'offsets': offsets, 'removed_fields': self.fields,
                'defined': self.defined, 'unapplied_bias': self.unapplied_bias, 'metadata': self.metadata}
