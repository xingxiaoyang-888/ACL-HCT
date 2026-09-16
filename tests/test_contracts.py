import pytest
import torch
from acl_hct.geometry import from_spatial, log, check_point
from acl_hct.aggregation import correct, sample_indices
from acl_hct.data import wordnet_nouns

@pytest.mark.parametrize('limit', [float('nan'), float('inf')])
def test_nonfinite_step_rejected(limit):
    with pytest.raises(ValueError):
        correct(from_spatial(torch.zeros(3, 2)), 5, max_step=limit)

def test_coordinate_broadcast_rejected():
    p = from_spatial(torch.zeros(2))
    with pytest.raises(ValueError):
        log(p, from_spatial(torch.zeros(1)))

def test_scalar_point_rejected():
    with pytest.raises(ValueError):
        check_point(torch.tensor(1.))

def test_missing_wordnet_pointer_count():
    with pytest.raises(ValueError, match='line 1'):
        wordnet_nouns(['00000001 00 n 02 root 0'])

def test_explicit_generator_required():
    with pytest.raises(ValueError):
        sample_indices(4, 2, None)

@pytest.mark.parametrize("bad", [[[-1]], [[2]], [[True]]])
def test_invalid_model_neighbors(bad):
    from acl_hct.model import TinyGNN
    with pytest.raises(ValueError):
        TinyGNN()(torch.zeros(1,4), bad, torch.Generator(), batched=True)

@pytest.mark.parametrize("c", [.1, 4.])
def test_geometry_batch_broadcast_nonorigin(c):
    from acl_hct.geometry import exp, dot
    p = from_spatial(torch.tensor([[[.3,.1]], [[-.2,.4]]],dtype=torch.float64),c)
    q = from_spatial(torch.tensor([[.1,.2],[-.1,.3],[.2,-.1]],dtype=torch.float64),c)
    v = log(p,q,c)
    torch.testing.assert_close(exp(p,v,c),q.expand(2,3,3),atol=2e-14,rtol=2e-13)
    torch.testing.assert_close(dot(p,v),torch.zeros(2,3,dtype=torch.float64),atol=2e-14,rtol=0)
