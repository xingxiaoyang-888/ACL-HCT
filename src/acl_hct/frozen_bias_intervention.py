"""Small FP64, fixed-base O/Q adapter for frozen diagnostic interventions.

The caller must verify calibration/reference weights, node order and provenance.
This module has no labels, model, input loader, repetition policy or task scorer.
Outputs stay FP64; any later native-head conversion requires a separate gate.
Q transports a standard Gaussian unit direction from the origin to each base.
This is isotropic under the tangent metric, not constrained to be orthogonal to b.
A direction is fixed across evaluation repeats.
"""
import copy
import hashlib

import torch

from .geometry import check_point, curvature, dot, exp, log, norm2, origin_like


def _tensor_identity(value):
    return {'dtype': str(value.dtype), 'shape': list(value.shape),
            'raw_sha256': hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()}


def _namespace(value):
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError('nonempty namespace without outer whitespace required')
    return value


@torch.no_grad()
def transport_from_origin(base, spatial, c=1.):
    """Parallel transport (0, spatial) along the origin-to-base geodesic.

    PT(v) = v + <base,v>_L / (1/c - <origin,base>_L) * (origin+base).
    The denominator remains positive at coincidence, where this is identity.
    """
    curvature(c)
    if (not isinstance(base, torch.Tensor) or base.dtype != torch.float64
            or base.ndim != 2 or len(base) == 0):
        raise ValueError('nonempty FP64 [nodes,coordinates] base required')
    check_point(base, c)
    if (not isinstance(spatial, torch.Tensor) or spatial.dtype != torch.float64
            or spatial.shape != (len(base), base.shape[1] - 1)
            or spatial.device != base.device or not torch.isfinite(spatial).all()):
        raise ValueError('complete finite FP64 spatial field on the base device required')
    vector = torch.cat((torch.zeros_like(spatial[:, :1]), spatial), dim=-1)
    origin = origin_like(base, c)
    scale = dot(base, vector) / (1 / c - dot(origin, base))
    result = vector + scale[:, None] * (origin + base)
    if not torch.isfinite(result).all() or (dot(base, result).abs() > 1e-9).any():
        raise ValueError('parallel transport produced an invalid tangent field')
    return result


class FrozenBiasIntervention:
    """Remove an already calibrated mean or an unrelated equal-norm field.

    Namespace separation enforces this API's random-stream contract. It cannot
    prove how the caller obtained b; reference/calibration records must be checked
    before use on real inputs. No field is masked by sampling-support or labels.
    """

    @torch.no_grad()
    def __init__(self, base, bias, *, calibration_namespace, evaluation_namespace,
                 direction_namespace, direction_seed, c=1.):
        self.c = curvature(c)
        if (not isinstance(base, torch.Tensor) or base.dtype != torch.float64
                or base.ndim != 2 or len(base) == 0):
            raise ValueError('nonempty FP64 [nodes,coordinates] base required')
        check_point(base, self.c)
        names = tuple(_namespace(x) for x in (calibration_namespace, evaluation_namespace, direction_namespace))
        if len(set(names)) != 3:
            raise ValueError('distinct calibration, evaluation and direction namespaces required')
        if type(direction_seed) is not int or direction_seed < 0:
            raise ValueError('nonnegative integer direction seed required')
        self._base = base.detach().clone()
        self._field(bias, 'bias')
        self._bias = bias.detach().clone()
        self._evaluation_namespace = names[1]
        derived = int.from_bytes(hashlib.sha256(
            f'{direction_seed}/{names[2]}/equal-norm-control-v1'.encode()).digest()[:8], 'big') % (2**63)
        generator = torch.Generator(device='cpu').manual_seed(derived)
        raw = torch.randn((len(self._base), self._base.shape[1] - 1),
                          dtype=torch.float64, generator=generator).to(self._base.device)
        lengths = raw.square().sum(-1).sqrt()
        amplitudes = norm2(self._bias).sqrt()
        self._moving = amplitudes > 0
        if not torch.isfinite(lengths).all() or (lengths <= 0).any():
            raise ValueError('degenerate origin direction; no redraw or fallback')
        direction = transport_from_origin(self._base, raw / lengths[:, None], self.c)
        self._control = direction * amplitudes[:, None]
        self._control[~self._moving] = 0
        self._field(self._control, 'control')
        self._metadata = {
            'scope': 'frozen diagnostic O/Q only; not a deployable correction or task bound',
            'precision': 'FP64 geometry and outputs; no native-head conversion',
            'curvature': self.c, 'calibration_namespace': names[0],
            'evaluation_namespace': names[1], 'direction_namespace': names[2],
            'direction_seed': direction_seed, 'direction_derived_seed': derived,
            'direction_rule': 'explicit CPU standard spatial Gaussian unit vector at origin, geodesic parallel transport to p, Lorentz equal-norm scaling; fixed across repeats',
            'base': _tensor_identity(self._base), 'bias': _tensor_identity(self._bias),
            'control': _tensor_identity(self._control),
            'calibration_provenance_verified_by_adapter': False,
            'interpretation': 'fixed tangent residuals are translated; exp need not preserve manifold variance, hierarchy or task scores',
        }

    def _field(self, value, name):
        if (not isinstance(value, torch.Tensor) or value.dtype != torch.float64
                or value.shape != self._base.shape or value.device != self._base.device
                or not torch.isfinite(value).all()):
            raise ValueError(f'{name} must be a complete finite FP64 field at the same device/base')
        if (dot(self._base, value).abs() > 1e-9).any():
            raise ValueError(f'{name} is not tangent at the fixed base')

    @property
    def base(self):
        return self._base.clone()

    @property
    def bias(self):
        return self._bias.clone()

    @property
    def control(self):
        return self._control.clone()

    @property
    def metadata(self):
        return copy.deepcopy(self._metadata)

    @torch.no_grad()
    def offsets(self, sampled_points, *, evaluation_namespace):
        if _namespace(evaluation_namespace) != self._evaluation_namespace:
            raise ValueError('registered evaluation namespace required')
        if (not isinstance(sampled_points, torch.Tensor) or sampled_points.dtype != torch.float64
                or sampled_points.shape != self._base.shape or sampled_points.device != self._base.device):
            raise ValueError('sample must match the complete FP64 reference shape/device')
        check_point(sampled_points, self.c)
        error = log(self._base, sampled_points, self.c)
        result = {'S': error, 'O': error - self._bias, 'Q': error - self._control}
        for name, field in result.items():
            self._field(field, name)
        return result

    @torch.no_grad()
    def apply(self, sampled_points, *, evaluation_namespace):
        fields = self.offsets(sampled_points, evaluation_namespace=evaluation_namespace)
        points = {'S': sampled_points.detach().clone()}
        for name in ('O', 'Q'):
            moved = sampled_points.detach().clone()
            if self._moving.any():
                moved[self._moving] = exp(self._base[self._moving], fields[name][self._moving], self.c)
            points[name] = moved
        return {'points': points, 'offsets': fields,
                'reconstructed_S': exp(self._base, fields['S'], self.c),
                'metadata': self.metadata}
