import pytest
import torch
from acl_hct.backbone import LorentzMeanNetwork
from acl_hct.geometry import from_spatial
from acl_hct.ranking import prepare_scores, score_cached, filtered_parent_ranks


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('changed_part',['head','backbone'])
def test_factorized_scores_match_direct_and_stale_cache_rejected(dtype,changed_part):
    torch.manual_seed(71);model=LorentzMeanNetwork(4,hidden=5,head_hidden=7).to(dtype=dtype).eval()
    points=from_spatial(torch.randn(9,5,dtype=dtype)*.1)
    queries=torch.cartesian_prod(torch.arange(9),torch.arange(9))
    cache=prepare_scores(model,points)
    direct=model.score(points,queries)
    cached=score_cached(model,cache,queries[:,0],queries[:,1])
    torch.testing.assert_close(cached,direct,atol=3e-8 if dtype==torch.float32 else 1e-15,rtol=1e-6 if dtype==torch.float32 else 1e-13)
    with torch.no_grad():
        parameter=model.relation_head[0].weight if changed_part=='head' else model.layers[0].linear.weight
        parameter.add_(.1)
    with pytest.raises(ValueError,match='stale'):score_cached(model,cache,queries[:,0],queries[:,1])


def test_all_candidate_filtering_ties_and_macro_micro():
    model=LorentzMeanNetwork(2,hidden=2,head_hidden=3).double().eval()
    with torch.no_grad():
        for parameter in model.relation_head.parameters():parameter.zero_()
    points=from_spatial(torch.zeros(6,2,dtype=torch.float64))
    truth={3:{0,1},4:{0}};queries=[(0,3),(1,3),(0,4)]
    result=filtered_parent_ranks(model,points,queries,truth,candidate_chunk=2)
    assert result['status']=='complete'
    assert [r['candidates'] for r in result['rows']]==[4,4,5]
    assert [r['rank'] for r in result['rows']]==[2.5,2.5,3.]
    assert result['query_micro_mrr']==pytest.approx((.4+.4+1/3)/3)
    assert result['child_macro_mrr']==pytest.approx((.4+1/3)/2)
    assert filtered_parent_ranks(model,points,queries,truth,max_seconds=0)['status']=='incomplete_time_limit'


def test_ranks_match_independent_direct_scores_without_ties():
    torch.manual_seed(31);model=LorentzMeanNetwork(2,hidden=3,head_hidden=4).double().eval()
    points=from_spatial(torch.randn(8,3,dtype=torch.float64)*.2)
    truth={3:{0,1},7:{2}};queries=[(0,3),(1,3),(2,7)]
    result=filtered_parent_ranks(model,points,queries,truth,candidate_chunk=3)
    for row in result['rows']:
        a,b=row['parent'],row['child'];allowed=[i for i in range(len(points)) if i!=b and (i not in truth[b] or i==a)]
        scores=model.score(points,torch.tensor([[i,b] for i in allowed]))
        target=scores[allowed.index(a)]
        rank=1+int((scores>target).sum())+.5*(int((scores==target).sum())-1)
        assert row['rank']==rank
