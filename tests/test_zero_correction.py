import torch
from acl_hct.geometry import from_spatial
from acl_hct.aggregation import correct


def test_repeated_origin_correction_backward():
    for method in ("third", "jackknife"):
        spatial = torch.zeros(3, 2, dtype=torch.float64, requires_grad=True)
        y, stats = correct(from_spatial(spatial), 6, method=method)
        y.sum().backward()
        assert torch.isfinite(spatial.grad).all()
        assert not stats["clipped"]
        assert torch.equal(y, torch.tensor([1., 0., 0.], dtype=torch.float64))
