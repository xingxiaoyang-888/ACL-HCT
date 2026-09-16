import copy
from io import StringIO
import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch
from acl_hct.taxonomy import wordnet_records, mesh_records, Taxonomy
from acl_hct.data import mesh_descriptors
from acl_hct.protocols import grouped_split, observed_neighbors, training_candidates, fit_text_features
from acl_hct.backbone import LorentzMeanNetwork, LorentzMeanLayer, make_plan, aggregate_plan
from acl_hct.geometry import log, origin_like, norm2


def fixture():
    records={f'n{i:02d}':{'name':f'entity word{i}', 'definition':f'common subject token{i}', 'tree_positions':['SECRET.PATH']} for i in range(20)}
    nodes=sorted(records); edges={(nodes[(i-1)//2],nodes[i]) for i in range(1,20)}
    edges.add((nodes[2],nodes[7]))
    return Taxonomy(records,edges,{})


def test_wordnet_rich_records_and_mesh_paths():
    result=wordnet_records(['00000001 00 n 01 living_thing 0 000 | natural life',
                           '00000002 00 n 01 animal 0 001 @i 00000001 n 0000 | an instance'])
    assert result.records['00000001']['name']=='living thing'
    assert result.records['00000002']['definition']=='an instance'
    xml='<DescriptorRecordSet>'+''.join(
        f'<DescriptorRecord><DescriptorUI>{ui}</DescriptorUI><DescriptorName><String>{ui} name</String></DescriptorName>'
        '<ConceptList><Concept PreferredConceptYN="Y"><ScopeNote>definition</ScopeNote></Concept></ConceptList><TreeNumberList>'+
        ''.join(f'<TreeNumber>{p}</TreeNumber>' for p in paths)+'</TreeNumberList></DescriptorRecord>'
        for ui,paths in [('R',['A01']),('S',['B01']),('C',['A01.001','B01.001']),('D',['A01.001']),('E',['A01.001.001'])])+'</DescriptorRecordSet>'
    parsed=mesh_records(StringIO(xml)); old_nodes,old_edges=mesh_descriptors(StringIO(xml))
    assert set(parsed.records)==old_nodes and parsed.edges==old_edges
    assert parsed.records['C']['tree_positions']==['A01.001','B01.001']
    assert parsed.records['C']['path_ancestors']['B01.001']==['S']
    assert parsed.records['E']['ambiguous_prefixes']==['A01.001']
    assert parsed.records['C']['definition']=='definition'


def test_grouping_visible_graph_and_query_mask():
    data=fixture(); nodes=sorted(data.records); split=grouped_split(nodes,data.edges)
    assert split==grouped_split(reversed(nodes),reversed(sorted(data.edges)))
    assert list(map(len,split['entities'].values()))==[16,2,2]
    for group,edges in split['relations'].items():
        assert all(child in split['entities'][group] for _,child in edges)
    held=split['relations']['valid']+split['relations']['test']
    assert all((a,b) not in split['observed_directed_edges'] and (b,a) not in split['observed_directed_edges'] for a,b in held)
    queries=split['relations']['train'][:3]
    graph=observed_neighbors(nodes,split['relations']['train'],queries)
    for a,b in queries:
        assert nodes.index(b) not in graph[nodes.index(a)]
        assert nodes.index(a) not in graph[nodes.index(b)]
    plan=make_plan(graph,4,torch.Generator().manual_seed(1))
    assert all(set(row).issubset(candidates) for row,candidates in zip(plan,graph))
    candidates=training_candidates(nodes,split['entities']['train'],split['relations']['train'])
    for query,label in zip(candidates['queries'],candidates['labels']):
        assert query[1] in split['entities']['train']
        assert bool(label)==(query in split['relations']['train'])


def test_text_fit_does_not_observe_heldout_or_paths():
    pytest.importorskip('sklearn')
    data=fixture();split=grouped_split(data.records,data.edges);train=split['entities']['train']
    a,ma,sa=fit_text_features(data.records,train,4)
    changed=copy.deepcopy(data.records)
    for node in changed:
        changed[node]['tree_positions']=['different hidden path']
        if node not in train: changed[node]['name']='forbiddenheldouttoken'; changed[node]['definition']='heldoutexclusive'
    b,mb,sb=fit_text_features(changed,train,4)
    assert ma['fit_state_hash']==mb['fit_state_hash'] and ma['train_text_hash']==mb['train_text_hash']
    assert 'forbiddenheldouttoken' not in sa['vocabulary'] and 'secret' not in sa['vocabulary']
    assert ma['text_hash']!=mb['text_hash']
    np.testing.assert_array_equal(a[[sorted(changed).index(n) for n in train]],b[[sorted(changed).index(n) for n in train]])


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_two_layers_complete_sampling_frozen_and_gradients(dtype):
    torch.manual_seed(12)
    model=LorentzMeanNetwork(4,hidden=5,head_hidden=7).to(dtype=dtype)
    features=torch.randn(6,4,dtype=dtype)
    graph=[[],[0],[0,1],[0,1,2,4,5],[0,1,2,3,5],[0,1,2,3,4]]
    full=[make_plan(graph,None,torch.Generator()) for _ in range(2)]
    a,stats,trace=model.encode(features,graph,full,max_padded_messages=12,return_trace=True)
    b,_,_=model.encode(features,graph,full,method='third',max_padded_messages=12)
    assert torch.equal(a,b) and len(trace)==2 and stats[0]['empty'][0]
    cache=model.full_reference(features,graph,max_padded_messages=12)
    for layer in (0,1):
        frozen,_=model.frozen_layer(cache,layer,graph,full[layer],max_padded_messages=12)
        assert torch.equal(frozen,trace[layer]['output'])
    plans=[make_plan(graph,3,torch.Generator().manual_seed(1+i)) for i in range(2)]
    sampled,_,_=model.encode(features,graph,plans,method='third',max_padded_messages=12)
    logits=model.score(sampled,torch.tensor([[0,1],[2,3],[3,4]],dtype=torch.long))
    torch.nn.functional.binary_cross_entropy_with_logits(logits,torch.tensor([1.,0.,1.],dtype=dtype)).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(any(p.grad.abs().max()>0 for p in layer.parameters()) for layer in model.layers)


def test_width_independent_domain_and_invalid_plan():
    for hidden in (2,128,512):
        layer=LorentzMeanLayer(3,hidden,c=4.,scaled_radius=1.2)
        points=layer.messages(torch.full((2,3),100.))
        scaled=2*norm2(log(origin_like(points,4.),points,4.)).sqrt()
        assert (scaled<=1.20001).all()
    with pytest.raises(ValueError): aggregate_plan(points,[[1],[0]],[[0],[1]],4.)


def test_prepare_fixture_and_no_overwrite(tmp_path):
    pytest.importorskip('sklearn')
    spec=importlib.util.spec_from_file_location('prepare',Path(__file__).parents[1]/'scripts/prepare_benchmark.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result=module.prepare(fixture(),tmp_path/'prepared',dimension=4)
    assert result['protocol']=='B-child-grouped-80-10-10-v1'
    assert (tmp_path/'prepared'/'evaluator_truth.json').exists()
    assert (tmp_path/'prepared'/'features.npz').exists()
    import json
    saved=json.loads((tmp_path/'prepared'/'input_manifest.json').read_text())
    assert saved['split_seed']==20260914 and saved['feature_seed']==11
    assert sum(saved['entity_counts'].values())==20
    with pytest.raises(ValueError): module.prepare(fixture(),tmp_path/'prepared',dimension=4)


def test_query_masked_two_step_fixture_training():
    torch.manual_seed(71)
    data=fixture(); nodes=sorted(data.records); split=grouped_split(nodes,data.edges)
    candidates=training_candidates(nodes,split['entities']['train'],split['relations']['train'])
    index={node:i for i,node in enumerate(nodes)}
    queries=torch.tensor([[index[a],index[b]] for a,b in candidates['queries']],dtype=torch.long)
    labels=torch.tensor(candidates['labels'],dtype=torch.float64)
    # One all-query fixture batch masks every positive supervision edge first.
    graph=observed_neighbors(nodes,split['relations']['train'],split['relations']['train'])
    assert all(not row for row in graph)
    model=LorentzMeanNetwork(4,hidden=5,head_hidden=7).double()
    features=torch.randn(len(nodes),4,dtype=torch.float64)
    initial=model.layers[0].linear.weight.detach().clone()
    optimizer=torch.optim.Adam(model.parameters(),lr=.01)
    for step in range(2):
        plans=[make_plan(graph,3,torch.Generator().manual_seed(step)) for _ in range(2)]
        optimizer.zero_grad()
        points,stats,_=model.encode(features,graph,plans)
        loss=torch.nn.functional.binary_cross_entropy_with_logits(model.score(points,queries),labels)
        assert torch.isfinite(loss) and stats[0]['empty'].all()
        loss.backward(); optimizer.step()
    assert not torch.equal(initial,model.layers[0].linear.weight.detach())

def test_preparation_cli_offline_archive_and_manifest(tmp_path):
    pytest.importorskip('sklearn')
    import hashlib, json, os, subprocess, sys, tarfile
    from io import BytesIO
    lines=[]
    for i in range(12):
        pointer='000' if i==0 else f'001 @ {i-1:08d} n 0000'
        lines.append(f'{i:08d} 00 n 01 entity{i} 0 {pointer} | distinct word{i} definition')
    raw=('\n'.join(lines)+'\n').encode()
    archive_path=tmp_path/'fixture.tar.bz2'
    with tarfile.open(archive_path,'w:bz2') as archive:
        info=tarfile.TarInfo('fixture/dict/data.noun');info.size=len(raw);archive.addfile(info,BytesIO(raw))
    source={'sha256':hashlib.sha256(archive_path.read_bytes()).hexdigest(),
            'structure':wordnet_records(lines).validate(),'url':'https://example.invalid/fixture','version':'fixture','license':'test'}
    manifest=tmp_path/'source.json';manifest.write_text(json.dumps(source))
    root=Path(__file__).parents[1]; env={**os.environ,'PYTHONPATH':str(root/'src')}
    command=[sys.executable,str(root/'scripts/prepare_benchmark.py'),'--dataset','wordnet','--raw',str(archive_path),
             '--source-manifest',str(manifest),'--output-root',str(tmp_path/'out'),'--dimension','3','--threads','2']
    subprocess.run(command,cwd=tmp_path,env=env,check=True,capture_output=True)
    runtime=json.loads((tmp_path/'out/preprocessing_runtime.json').read_text())
    assert runtime['threads']==2 and runtime['raw_sha256']==source['sha256']
    assert json.loads((tmp_path/'out/input_manifest.json').read_text())['split_seed']==20260914
