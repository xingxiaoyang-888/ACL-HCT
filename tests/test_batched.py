import pytest
import torch
from acl_hct.aggregation import aggregate, correct, correct_batched
from acl_hct.geometry import from_spatial, log, exp, dot
from acl_hct.model import TinyGNN

@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('method', ['none', 'third', 'jackknife'])
@pytest.mark.parametrize('c', [.1, 1., 4.])
def test_ragged_reference_values_gradients(dtype, method, c):
    g = torch.Generator().manual_seed(77)
    u = (torch.randn(6, 5, 3, generator=g, dtype=dtype)*.12).requires_grad_()
    x = from_spatial(u, c)
    lengths = [1, 2, 3, 4, 5, 3]
    populations = [7, 7, 3, 8, 9, 6]
    mask = torch.arange(5)[None, :] < torch.tensor(lengths)[:, None]
    # Holes and NaN padding must be ignored in both value and backward.
    mask[3] = torch.tensor([True, False, True, True, True])
    padded = torch.where(mask[..., None], x, torch.full_like(x, float('nan')))
    y, stats = correct_batched(padded, torch.tensor(populations), mask, c, method, max_step=1e-5)
    reference = [correct(x[i, mask[i]], n, c, method, max_step=1e-5)
                 for i, n in enumerate(populations)]
    ref = torch.stack([r[0] for r in reference])
    atol, rtol = (3e-7, 3e-5) if dtype == torch.float32 else (3e-14, 3e-12)
    torch.testing.assert_close(y, ref, atol=atol, rtol=rtol)
    weights = torch.randn(y.shape, generator=g, dtype=dtype)
    grad = torch.autograd.grad((y*weights).sum(), u, retain_graph=True)[0]
    refgrad = torch.autograd.grad((ref*weights).sum(), u)[0]
    torch.testing.assert_close(grad, refgrad, atol=atol, rtol=rtol)
    assert torch.equal(grad[~mask], torch.zeros_like(grad[~mask]))
    assert stats['fallback'].tolist() == [r[1]['fallback'] for r in reference]
    assert stats['clipped'].tolist() == [r[1]['clipped'] for r in reference]
    assert all(not v.requires_grad and v.device == x.device for v in stats.values())
    assert (stats['step'] <= 1.00001e-5).all()

@pytest.mark.parametrize('method', ['none', 'third', 'jackknife'])
def test_empty_full_and_batch_broadcast(method):
    u = torch.tensor([[[.1,.2],[.2,-.1],[.3,.2]]], dtype=torch.float64, requires_grad=True)
    x = from_spatial(u).expand(2, 1, 3, 3)
    mask = torch.tensor([[[False]*3], [[True]*3]])
    own = from_spatial(torch.tensor([.2,.3], dtype=torch.float64))
    y, s = correct_batched(x, torch.tensor([[0],[3]]), mask, method=method, self_points=own)
    assert torch.equal(y[0,0], own)
    assert torch.equal(y[1,0], aggregate(x[1,0]))
    assert s['empty'].tolist() == [[True], [False]]
    assert torch.isfinite(torch.autograd.grad(y.sum(), u)[0]).all()

@pytest.mark.parametrize('method', ['third','jackknife'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('delta', [0., 1e-7])
def test_nonorigin_near_coincidence(method, dtype, delta):
    u = torch.tensor([[.7,-.3]]*4, dtype=dtype)
    u[1,0] += delta
    u.requires_grad_()
    x = from_spatial(u)
    y, _ = correct_batched(x[None], 9, torch.ones(1,4,dtype=torch.bool), method=method)
    assert torch.isfinite(torch.autograd.grad(y.sum(), u)[0]).all()
    torch.testing.assert_close(y[0], x[0], atol=2e-7, rtol=2e-6)

@pytest.mark.parametrize('method', ['none','third','jackknife'])
def test_model_paths_and_parameter_gradients(method):
    torch.manual_seed(31)
    model = TinyGNN().double()
    x = torch.randn(6,4,dtype=torch.float64)
    neighbors = [[],[0],[0,1],[0,1,2,4,5],[0,1,2,3,5],[1,1,2,3,4]]
    a = model(x, neighbors, torch.Generator().manual_seed(7), method=method)
    b = model(x, neighbors, torch.Generator().manual_seed(7), method=method, batched=True)
    torch.testing.assert_close(a[0], b[0], atol=1e-14, rtol=1e-12)
    ga = torch.autograd.grad(a[0].square().sum(), tuple(model.parameters()))
    gb = torch.autograd.grad(b[0].square().sum(), tuple(model.parameters()))
    for left, right in zip(ga,gb):
        torch.testing.assert_close(left,right,atol=1e-13,rtol=1e-11)
    assert b[2]['empty'][0]

@pytest.mark.parametrize('bad_n', [-1, 2, 3.5, True, torch.tensor([3.])])
def test_invalid_population(bad_n):
    with pytest.raises(ValueError):
        correct_batched(from_spatial(torch.zeros(1,3,2)), bad_n, torch.ones(1,3,dtype=torch.bool))

def test_invalid_mask_empty_and_self():
    x = from_spatial(torch.zeros(2,3,2))
    for mask in (torch.ones(2,3), torch.ones(2,2,dtype=torch.bool)):
        with pytest.raises(ValueError): correct_batched(x, 5, mask)
    with pytest.raises(ValueError): correct_batched(x, 0, torch.zeros(2,3,dtype=torch.bool))
    with pytest.raises(ValueError): correct_batched(x, 5, torch.zeros(2,3,dtype=torch.bool), self_points=x[:,0])

def test_batch_gradcheck_independent_finite_difference():
    u = torch.tensor([[[.2,-.1],[-.1,.3],[.05,.04]]], dtype=torch.float64, requires_grad=True)
    for method in ('third','jackknife'):
        assert torch.autograd.gradcheck(lambda z: correct_batched(from_spatial(z), 7,
            torch.ones(1,3,dtype=torch.bool), method=method)[0], (u,), atol=1e-7, rtol=1e-5)
