import os
from pathlib import Path
import shutil
import sys
from types import ModuleType
import pytest
import torch

from acl_hct.hgcn_fixture import run_fixture
from acl_hct.hgcn_sampling import make_plan
from acl_hct.hgcn_upstream import load_upstream, verify_checkout
from acl_hct.mature_hgcn import MatureHGCN, to_lorentz_fp64
from acl_hct.text_capacity import filtered_parent_ranks


@pytest.fixture(scope='module')
def upstream():
    # Default pytest stays offline. Required release uses explicit checkout;
    # the standalone fixture refuses missing/modified source and never skips.
    root = os.environ.get('ACL_HGCN_UPSTREAM_PATH')
    if not root:
        pytest.skip('explicit pinned external checkout needed for official comparison')
    return load_upstream(root)


def test_required_actual_official_output_gradient_sampling_and_reload_fixture(upstream, tmp_path):
    result = run_fixture(upstream, archive_dir=tmp_path / 'arrays')
    assert result['status'] == 'passed' and len(result['cases']) == 2
    assert len(result['disk_checkpoint_roundtrips']) == 2 and result['reproduction_archive']['arrays'] >= 100
    assert result['cases'][0]['official_output_max_error'] < 2e-6
    assert result['cases'][1]['official_output_max_error'] < 1e-10


def test_global_absolute_import_names_are_restored(upstream):
    prior = sys.modules.get('utils'); sentinel = ModuleType('utils')
    sys.modules['utils'] = sentinel
    try:
        api = load_upstream(os.environ['ACL_HGCN_UPSTREAM_PATH'])
        assert sys.modules['utils'] is sentinel and api.identity['checkout_modified'] is False
    finally:
        if prior is None:
            del sys.modules['utils']
        else:
            sys.modules['utils'] = prior


def test_modified_source_and_shadow_module_rejected(upstream, tmp_path):
    root = Path(os.environ['ACL_HGCN_UPSTREAM_PATH'])
    copied = tmp_path / 'upstream'; shutil.copytree(root, copied)
    source = copied / 'layers/hyp_layers.py'; original = source.read_bytes()
    source.write_bytes(original + b'\n# changed\n')
    with pytest.raises(ValueError, match='modified'):
        verify_checkout(copied)
    source.write_bytes(original)
    (copied / 'utils/shadow_module.py').write_text('# shadow module\n')
    with pytest.raises(ValueError, match='shadow'):
        verify_checkout(copied)


def test_curvature_buffer_and_all_official_uses_cast_together(upstream):
    model = MatureHGCN(upstream, 3, 4, 5).double()
    assert model.curvature.dtype == torch.float64
    assert all(c is model.curvature for c in model.encoder.curvatures)
    assert all(layer.linear.c is model.curvature and layer.agg.c is model.curvature for layer in model.encoder.layers)
    adj = make_plan([[], []], None, torch.Generator()).matrix(dtype=torch.float64)
    points = model.encode(torch.zeros(2, 3, dtype=torch.float64), [adj, adj])
    assert torch.equal(points, torch.zeros_like(points))


def test_head_ties_filter_multiple_true_parents_and_duplicate_query_weights(upstream):
    model = MatureHGCN(upstream, 3, 4, 5).eval()
    with torch.no_grad():
        for p in model.relation_head.parameters():
            p.zero_()
    embeddings = torch.zeros(5, 4)
    result = filtered_parent_ranks(model, embeddings, [(0, 4), (1, 4)], {4: {0, 1}}, 2)
    assert result['completed_queries'] == 2
    assert all(r['rank'] == 2 and r['candidates'] == 3 for r in result['rows'])
    assert result['query_micro_mrr'] == .5
    queries = torch.tensor([[0, 4], [0, 4], [1, 4]])
    assert model.score(torch.zeros(5, 4), queries).shape == (3,)


def test_isometry_origin_radius_and_invalid_boundary():
    p = torch.tensor([[0., 0.], [.25, 0.], [.1, -.2]], dtype=torch.float64)
    h = to_lorentz_fp64(p, .7)
    constraint = -h[:, 0].square() + h[:, 1:].square().sum(1)
    torch.testing.assert_close(constraint, torch.full((3,), -1 / .7, dtype=torch.float64), atol=1e-14, rtol=0)
    radius = torch.acosh((h[:, 0] * .7 ** .5).clamp_min(1)) / .7 ** .5
    expected = 2 * torch.atanh(.7 ** .5 * p.norm(dim=1)) / .7 ** .5
    torch.testing.assert_close(radius, expected, atol=1e-14, rtol=0)
    with pytest.raises(ValueError, match='interior'):
        to_lorentz_fp64(torch.tensor([[1., 0.]]))
