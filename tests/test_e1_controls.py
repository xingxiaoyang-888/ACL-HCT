from itertools import combinations
import json
import math
from pathlib import Path
import pytest
import torch
from acl_hct.aggregation import aggregate, correct_batched
from acl_hct.e1_controls import evaluate_exact_controls, lorentz_boost, radial_scale
from acl_hct.geometry import dot, exp, from_spatial, log, norm2, origin_like
from acl_hct.mechanisms import Case, evaluate_points, population


def small_population():
    return from_spatial(torch.tensor([[-.1,.02],[-.04,-.02],[0.,.01],[.02,.04],[.15,-.03]],dtype=torch.float64))


@pytest.mark.parametrize('coefficient',[-1,0,1])
def test_fixed_scale_independent_radial_reference_and_identity(coefficient):
    y=from_spatial(torch.tensor([[.2,-.1],[.0,.3]],dtype=torch.float64))
    alpha=1+coefficient*(1/2-1/8)
    expected=from_spatial(torch.tensor([[.2,-.1],[.0,.3]],dtype=torch.float64)*alpha)
    torch.testing.assert_close(radial_scale(y,8,2,coefficient),expected,atol=1e-14,rtol=1e-13)
    assert radial_scale(y,8,8,coefficient) is y
    if coefficient==0:assert radial_scale(y,8,1,coefficient) is y


def test_exact_oracle_joint_covariance_and_radial_transverse_identities():
    x=small_population();p=aggregate(x)
    offsets=torch.stack([log(p,aggregate(x[list(ids)])) for ids in combinations(range(5),3)])
    mean=offsets.mean(0);centered=offsets-mean
    covariance=centered.T@centered/len(centered)
    result=evaluate_exact_controls(x,3,chunk_size=3)
    noise=result['methods']['oracle_centered_pm'];none=result['methods']['none']
    torch.testing.assert_close(torch.tensor(none['mean_offset'],dtype=torch.float64),mean,atol=1e-14,rtol=1e-13)
    torch.testing.assert_close(torch.tensor(noise['covariance_population_ambient'],dtype=torch.float64),covariance,atol=2e-15,rtol=1e-11)
    assert noise['mean_offset_norm']<1e-14
    assert noise['mse']==pytest.approx(float(norm2(centered).mean()),abs=2e-15)
    assert noise['mse']==pytest.approx(none['variance_population_moment'],abs=2e-15)
    assert result['noise_identity']['radial_variance_difference']==pytest.approx(0.,abs=2e-15)
    assert result['noise_identity']['transverse_variance_difference']==pytest.approx(0.,abs=2e-15)
    assert result['sample_stream_sha256']==result['second_pass_stream_sha256']
    assert result['two_pass_baseline_sum_max_difference']==0
    assert result['noise_identity']['signed_outputs']==20
    assert result['noise_identity']['independent_mc_samples']==0 and noise['mse_mc_se']==0
    assert noise['covariance_denominator']==20 and result['noise_identity']['covariance_denominator']==10
    json.dumps(result,allow_nan=False)


def test_chunk_invariance_and_old_methods_preserve_negative_result():
    x=small_population()
    a=evaluate_exact_controls(x,3,chunk_size=2,include_original=True)
    b=evaluate_exact_controls(x,3,chunk_size=7,include_original=True)
    old=evaluate_points(x,3,chunk_size=2)
    assert a['sample_stream_sha256']==b['sample_stream_sha256']
    for name in old['methods']:
        for metric in ('mse','mean_offset_norm','variance_population_moment'):
            assert a['methods'][name][metric]==pytest.approx(old['methods'][name][metric],abs=2e-15)
    for name in a['methods']:
        assert a['methods'][name]['mse']==pytest.approx(b['methods'][name]['mse'],abs=2e-15)
    assert a['methods']['third_unclipped']['mean_offset_norm']<a['methods']['none']['mean_offset_norm']
    assert a['methods']['third_unclipped']['mse']>a['methods']['none']['mse']


def test_full_and_small_k_scaling_is_not_correction_fallback():
    x=small_population()
    full=evaluate_exact_controls(x,len(x))
    for row in full['methods'].values():assert row['mse']<1e-28
    one=evaluate_exact_controls(x,1)
    assert one['methods']['scale_expand']['fallback_rate']==0
    assert one['methods']['scale_expand']['mse']!=one['methods']['none']['mse']
    assert one['methods']['scale_identity']['mean_offset']==one['methods']['none']['mean_offset']


def test_domain_failure_is_retained_and_exact_work_bound():
    x=from_spatial(torch.tensor([[2.,0.],[2.1,.1],[2.2,-.1]],dtype=torch.float64))
    result=evaluate_exact_controls(x,1,chunk_size=1)
    assert result['status']=='partial_method_failure'
    assert result['methods']['scale_expand']['status']=='domain_failure'
    assert result['methods']['scale_expand']['metrics'] is None
    assert result['methods']['none']['status']=='ok'
    with pytest.raises(ValueError,match='exact only'):evaluate_exact_controls(small_population(),3,enumeration_threshold=2)
    with pytest.raises(ValueError,match='fixed lambda'):radial_scale(small_population(),5,3,2)


def test_isometry_outputs_vectors_and_non_equivariant_origin_scale():
    x=small_population();shifted=lorentz_boost(x,.4);p=aggregate(x);q=aggregate(shifted)
    ids=torch.tensor(list(combinations(range(5),3)),dtype=torch.long)
    mask=torch.ones((len(ids),3),dtype=torch.bool)
    for method,cap in [('none',.1),('third',1e100),('third',.1),('jackknife',.1)]:
        y,_=correct_batched(x[ids],5,mask,method=method,max_step=cap)
        moved,_=correct_batched(shifted[ids],5,mask,method=method,max_step=cap)
        torch.testing.assert_close(moved,lorentz_boost(y,.4),atol=2e-14,rtol=1e-12)
        torch.testing.assert_close(log(q,moved),lorentz_boost(log(p,y),.4),atol=2e-14,rtol=1e-12)
    y=aggregate(x[ids])
    assert not torch.allclose(radial_scale(lorentz_boost(y,.4),5,3,1),lorentz_boost(radial_scale(y,5,3,1),.4),atol=1e-10,rtol=1e-10)


def test_proposal_inventory_is_static_exact_and_not_authorization():
    config=json.loads((Path(__file__).parents[1]/'configs/e1_minimum_closure_proposal.json').read_text())
    assert config['status']=='proposal_pending_explicit_user_approval'
    assert config['scale_lambdas']==[-1,0,1] and len(config['cases'])==102
    # Validate parameters and integer combinatorics only: never evaluate 102 populations here.
    for row in config['cases']:Case(**row['case']).validate()
    counts=[math.comb(row['case']['N'],row['case']['k']) for row in config['cases']]
    assert sum(counts)==435561 and max(counts)==12870
    assert sum(count for row,count in zip(config['cases'],counts) if row['block']=='historical_controls_only')==138483
