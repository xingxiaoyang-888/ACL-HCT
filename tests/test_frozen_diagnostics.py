import json
import math
from pathlib import Path
import pytest
import torch
from acl_hct.aggregation import aggregate
from acl_hct.backbone import LorentzMeanNetwork, make_plan
from acl_hct.development_view import build_view, load_development_view, make_panels, support_groups
from acl_hct.frozen_fixture import jsonable, run_fixture, synthetic_fixture
from acl_hct.frozen_forward import FrozenForward, PlanStreams
from acl_hct.frozen_stats import TangentStream, outward_directions, promote_points
from acl_hct.frozen_structure import radial_relations, require_confirmation_registration
from acl_hct.geometry import dot, exp, from_spatial, log, norm2, origin_like
from test_training_runner import prepared


def test_strict_development_loader_allowlist(prepared, monkeypatch):
    original=Path.read_text;opened=[]
    allowed={'input_manifest.json','observed_graph.json','train_queries.json','evaluator_valid.json','entity_split.json'}
    def read(path,*args,**kwargs):
        assert path.name in allowed;opened.append(path.name)
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',read)
    view=load_development_view(prepared)
    assert set(opened)==allowed
    assert view.edges==view.train_edges | view.valid_edges
    assert view.metadata['loaded_files']==opened


def test_root_dag_multiparent_unknown_and_work_bound():
    nodes=list('abcdefghz');neighbors=[[] for _ in nodes]
    train=[('a','b'),('a','c'),('b','c'),('d','e'),('e','f')]
    view=build_view(nodes,neighbors,train,[('c','g'),('f','h')],['g','h','z'])
    assert view.root=='a'  # Both roots have three distinct descendants.
    assert view.shortest['c']==1 and view.longest['c']==2
    assert view.ancestors('g')=={'a','b','c'}
    assert view.shortest.get('h') is None and view.shortest.get('z') is None
    assert 'z' not in view.ancestors('g')  # Missing path remains unknown, no negative API.
    with pytest.raises(ValueError,match='work bound'):
        build_view(nodes,neighbors,train,[],[],max_reach_visits=1)
    with pytest.raises(ValueError,match='DAG'):
        build_view(nodes,neighbors,[('a','b'),('b','a')],[],[])


def test_panel_design_weights_and_relation_freeze():
    nodes=[f'n{i:02d}' for i in range(70)]
    degrees=[0,1,3,5,9,17,33]
    graph=[[(i+j+1)%70 for j in range(degrees[i%7])] for i in range(70)]
    view=build_view(nodes,graph,[],[],nodes)
    a=make_panels(view,10);b=make_panels(view,10)
    assert a==b
    dev,confirm=a['panels'].values()
    assert set(dev['pool']).isdisjoint(confirm['pool'])
    assert set(dev['pool']) | set(confirm['pool'])==set(nodes)
    for panel in a['panels'].values():
        assert len(panel['rows'])==10
        assert sum(row['pool_mean_weight'] for row in panel['rows'])==pytest.approx(1.)
        for row in panel['rows']:
            stratum=next(s for s in panel['strata'] if s['degree']==row['stratum'])
            assert row['inclusion_probability']==stratum['selected']/stratum['population']
        assert all(r['direct_parents']==[] and r['positive_distant_ancestors']==[] for r in panel['relations'])
    _,_,real_view=synthetic_fixture()
    panels=make_panels(real_view)
    for panel in panels['panels'].values():
        for row in panel['relations']:
            assert set(row['direct_parents'])=={a for a,b in real_view.valid_edges if b==row['child']}
            assert set(row['positive_distant_ancestors']) <= real_view.ancestors(row['child'])-real_view.parents[row['child']]
            assert len(row['positive_distant_ancestors'])<=4


def scalar_layer(model,index,inputs,plan):
    messages=model.layers[index].messages(inputs)
    return torch.stack([aggregate(messages[ids],model.c) if ids else messages[i]
                        for i,ids in enumerate(plan)])


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_paired_paths_against_independent_scalar_reference_and_support(dtype):
    model,features,view=synthetic_fixture(dtype)
    graph=view.neighbors;forward=FrozenForward(model,features,graph,128)
    streams=PlanStreams(73,'test-paired');plans=streams.draw(graph,16)
    rows=forward.paired(plans)
    first=scalar_layer(model,0,features,plans[0])
    full_first=scalar_layer(model,0,features,graph)
    for name,first_input,second_plan in [('F/S',full_first,plans[1]),('S/F',first,graph),('S/S',first,plans[1])]:
        inputs=log(origin_like(first_input),first_input)[...,1:]
        expected=scalar_layer(model,1,inputs,second_plan)
        torch.testing.assert_close(rows[name]['points'],expected,atol=5e-7 if dtype==torch.float32 else 1e-14,rtol=1e-6 if dtype==torch.float32 else 1e-13)
    torch.testing.assert_close(rows['local_L1']['points'],first)
    groups=support_groups(graph,16)
    assert groups['A']==[0] and groups['P_minus_A']==list(range(1,25))
    assert any(not torch.equal(rows['S/F']['points'][i],rows['F/F']['points'][i]) for i in groups['P_minus_A'])
    torch.testing.assert_close(rows['S/S']['points'][groups['V_minus_P']],rows['F/F']['points'][groups['V_minus_P']],atol=1e-7,rtol=1e-6)
    full=forward.paired([graph,graph])
    for name in ('F/S','S/F','S/S'):assert torch.equal(full[name]['points'],full['F/F']['points'])
    assert torch.equal(full['local_L1']['points'],full['F/F_L1']['points'])
    with torch.no_grad():model.layers[0].linear.bias.add_(.01)
    with pytest.raises(ValueError,match='changed'):forward.paired(plans)


def test_independent_layer_rng_and_calibration_namespace():
    _,_,view=synthetic_fixture();a=PlanStreams(73,'development');b=PlanStreams(73,'development')
    assert a.metadata==b.metadata and a.seeds[0]!=a.seeds[1]
    assert set(a.seeds).isdisjoint(PlanStreams(73,'calibration').seeds)
    # Consuming only L1 never changes L2's stream.
    make_plan(view.neighbors,16,a.generators[0])
    assert make_plan(view.neighbors,16,a.generators[1])==make_plan(view.neighbors,16,b.generators[1])


def test_common_base_streaming_moments_and_graph_level_se():
    base=from_spatial(torch.tensor([[.1,.2],[-.1,.3],[.2,-.1]],dtype=torch.float64))
    raw=torch.randn(7,3,3,dtype=torch.float64,generator=torch.Generator().manual_seed(9))*.02
    z=raw+dot(base,raw)[...,None]*base
    stream=TangentStream(base,groups={'all':[0,1,2],'empty':[]})
    for row in z:stream.add_offsets(row)
    result=stream.finish();mean=z.mean(0);squared=dot(z,z)
    torch.testing.assert_close(result['mean_offset'],mean)
    torch.testing.assert_close(result['variance'],dot(z-mean,z-mean).mean(0))
    torch.testing.assert_close(result['mse']['mean'],squared.mean(0))
    torch.testing.assert_close(result['groups']['all']['mse']['mc_se'],squared.mean(1).std()/math.sqrt(7))
    torch.testing.assert_close(result['independent_half_cross_inner_product'],dot(z[::2].mean(0),z[1::2].mean(0)))
    assert result['groups']['empty']['mse'] is None
    assert result['groups']['all']['mse']['independent_graph_repetitions']==7
    assert result['mse_decomposition_max_residual']<1e-17


def test_signed_bias_estimators_and_undefined_directions():
    base=from_spatial(torch.zeros(2,2,dtype=torch.float64));direction,defined=outward_directions(base,base[0])
    stream=TangentStream(base,directions=direction,direction_defined=defined)
    z=torch.tensor([[0.,.1,0.],[0.,.2,0.]],dtype=torch.float64)
    for row in (z,-z,z,-z):stream.add_offsets(row)
    result=stream.finish(torch.zeros(2,dtype=torch.float64))
    assert (result['bias_squared_noise_corrected']<0).all()
    assert (result['independent_half_cross_inner_product']<0).all()
    assert not defined.any() and result['groups']['V']['projection'] is None
    assert jsonable(result)['projection']['mean']==[None,None]


def test_uniform_promotion_and_full_fp64_numerical_audit():
    model,features,view=synthetic_fixture()
    forward=FrozenForward(model,features,view.neighbors,128);audit=forward.numerical_audit()
    for row in audit['layers']:
        assert row['base'].dtype==torch.float64
        assert row['native_promotion']['constraint_before'].max()>1e-10
        assert row['native_promotion']['constraint_after'].max()<1e-13
        assert row['native_vs_fp64_full_distance'].max()>0
        assert torch.isfinite(row['empirical_numerical_floor']).all()
        assert row['self_log_norm'].eq(0).all()
    base=audit['layers'][0]['base']
    _,defined=outward_directions(base,base[0],numerical_floor=torch.full((len(base),),100.))
    assert not defined.any()


def test_radial_ties_transitions_child_macro_and_unknowns():
    nodes=list('abcdef')
    view=build_view(nodes,[[] for _ in nodes],[('a','b')],[('a','c'),('b','c'),('a','d'),('e','f')],['c','d','f'])
    panel={'rows':[{'id':n,'pool_mean_weight':1/3} for n in ('c','d','f')],
           'relations':[{'child':'c','direct_parents':['a','b'],'positive_distant_ancestors':[]},
                        {'child':'d','direct_parents':['a'],'positive_distant_ancestors':[]},
                        {'child':'f','direct_parents':['e'],'positive_distant_ancestors':[]}]}
    reference=from_spatial(torch.tensor([[0.],[.1],[.2],[0.],[.1],[.3]],dtype=torch.float64))
    points=from_spatial(torch.tensor([[.3],[.1],[.05],[0.],[.1],[.3]],dtype=torch.float64))
    result=radial_relations(view,panel,points,reference,{'V':list(range(6))})
    metrics=result['metrics']['direct']['groups']['V']
    assert metrics['covered_children']==2 and metrics['unknown_relations']==1
    # c: both orders reverse (0), d: root moved but anchor is fixed, order reverses (0).
    assert metrics['weighted_covered_child_metrics']['score']==0.
    assert metrics['weighted_covered_child_metrics']['full_score']==.75  # c=1; d=tie .5; child-macro.
    assert metrics['weighted_covered_child_metrics']['correct_to_error']==.5
    assert metrics['weighted_covered_child_mass_in_pool']==pytest.approx(2/3)


def test_confirmation_guard_and_bounded_fixture_integration():
    with pytest.raises(ValueError,match='five'):require_confirmation_registration({})
    row={'metric':'predeclared fixture metric','unit':'fraction','threshold':.01}
    registration={key:dict(row) for key in ('full_reference_ability','effect_loss_threshold','independent_support_metric','matched_noise_difference','mc_precision')}
    registration.update(frozen_before_confirmation=True,test_labels_unopened=True)
    assert require_confirmation_registration(registration)['scientific_approval_inferred'] is False
    with pytest.raises(ValueError,match='2..16'):run_fixture(100)
    result=run_fixture(2)
    assert result['status']=='complete' and result['completed_graph_repetitions']==2
    assert result['full_task_reference']['completed_queries']==7
    assert set(result['full_task_repetition_statistics'])=={'F/F','F/S','S/F','S/S'}
    assert not result['real_R0_R_Rtask_authorized']
    assert result['geometry']['F/F']['mse']['mean'].eq(0).all()
    json.dumps(jsonable(result),allow_nan=False)
