from itertools import combinations
import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch
from acl_hct.mechanisms import Case, population, predictions, evaluate_points, index_chunks
from acl_hct.geometry import from_spatial, log, norm2
from acl_hct.aggregation import aggregate


def reference():
    spec=importlib.util.spec_from_file_location('legacy',Path(__file__).parents[1]/'research/legacy/numerical_check.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def test_exact_numpy_direction_and_uncapped_formula():
    ref=reference()
    x=from_spatial(torch.tensor([[-.1,0.],[-.04,.02],[0.,0.],[.02,.01],[.15,-.02]],dtype=torch.float64))
    result=evaluate_points(x,3,chunk_size=3)
    p,b=ref.bias_prediction(x.numpy(),5,3)
    np.testing.assert_allclose(result['predictions']['second_order_oracle'],b,atol=1e-14)
    offsets=[ref.log(p,ref.third_moment_correction(x[list(ix)].numpy(),5)) for ix in combinations(range(5),3)]
    actual=result['methods']['third_unclipped']
    np.testing.assert_allclose(actual['mean_offset'],np.mean(offsets,axis=0),atol=2e-15)
    assert actual['mse']==pytest.approx(np.mean(ref.dot(np.array(offsets),np.array(offsets))),abs=1e-14)
    assert actual['mse_mc_se']==0
    assert actual['mse']>result['methods']['none']['mse']
    assert actual['mean_offset_norm']<result['methods']['none']['mean_offset_norm']


def test_reflection_symmetry_and_reverse():
    symmetric=evaluate_points(population(Case('s',N=6,family='symmetric')),3)
    assert symmetric['methods']['none']['mean_offset_norm']<1e-14
    a=evaluate_points(population(Case('a',N=6)),3)
    b=evaluate_points(population(Case('r',N=6,family='reverse')),3)
    for name in a['methods']:
        left=np.array(a['methods'][name]['mean_offset']); right=np.array(b['methods'][name]['mean_offset'])
        np.testing.assert_allclose(left[1:],-right[1:],atol=1e-14)
        assert a['methods'][name]['mse']==pytest.approx(b['methods'][name]['mse'],abs=1e-15)


def test_stream_reproducibility_and_mc_uncertainty():
    x=population(Case('a',N=8))
    a=evaluate_points(x,4,enumeration_threshold=1,mc_draws=100,chunk_size=7)
    b=evaluate_points(x,4,enumeration_threshold=1,mc_draws=100,chunk_size=16)
    assert a['sample_stream_sha256']==b['sample_stream_sha256']
    assert a['mode']=='mc' and a['methods']['none']['mse_mc_se']>0
    np.testing.assert_allclose(a['methods']['none']['mean_offset'],b['methods']['none']['mean_offset'],atol=1e-15)
    # Independent empirical calculation of covariance of the mean and unbiased squared bias.
    indices=torch.cat(list(index_chunks(8,4,11,'mc',100,11)))
    p=aggregate(x); z=log(p,aggregate(x[indices])); mean=z.mean(0)
    expected=torch.cov(z.T)/len(z)
    np.testing.assert_allclose(a['methods']['none']['mean_offset_covariance_mc'],expected.numpy(),atol=1e-15)
    bias2=norm2(mean)-norm2(z-mean).sum()/len(z)/(len(z)-1)
    assert a['methods']['none']['bias_squared_noise_corrected']==pytest.approx(float(bias2),abs=1e-15)
    assert a['methods']['none']['paired_mse_delta_vs_none']==0


def test_full_small_fallback_and_domain_failure():
    from acl_hct.mechanisms import run
    x=population(Case('a',N=6))
    for k in (1,2,6):
        result=evaluate_points(x,k)
        for name in ('third_unclipped','third_protected','jackknife_protected'):
            assert result['methods'][name]['fallback_rate']==1
        if k==6: assert result['methods']['none']['mse']<1e-28
    config={'enumeration_threshold':100,'mc_draws':10,'chunk_size':4,
            'cases':[{'name':'outside','N':6,'spread':10.}]}
    assert run(config)['cases'][0]['result']['status']=='input_or_domain_failure'


def test_boost_within_domain_preserves_mse():
    a=evaluate_points(population(Case('a',N=6)),3)
    b=evaluate_points(population(Case('b',N=6,origin_shift=.5)),3)
    for name in a['methods']:
        assert a['methods'][name]['mse']==pytest.approx(b['methods'][name]['mse'],rel=2e-12,abs=1e-14)
