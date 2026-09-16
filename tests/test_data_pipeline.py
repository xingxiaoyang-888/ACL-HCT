import pytest
import torch
from acl_hct.data import wordnet_nouns, mesh_descriptors, audit, split_relations, reachable
from acl_hct.model import TinyGNN
from acl_hct.smoke import run


def test_wordnet_fixture():
    records=["  copyright header", "00000001 00 n 01 root 0 000 | root", "00000002 00 n 01 child 0 002 @ 00000001 n 0000 @ 00000001 n 0000 | child", "00000003 00 n 01 alone 0 000 | alone"]
    nodes,edges=wordnet_nouns(records)
    assert len(nodes)==3 and edges=={("00000001","00000002")}
    assert audit(nodes,edges)["isolated"]==1
    with pytest.raises(ValueError): wordnet_nouns(records[2:])
    with pytest.raises(ValueError): wordnet_nouns(["invalid"])


def test_mesh_fixture(tmp_path):
    p=tmp_path/"tiny.xml"
    p.write_text("<DescriptorRecordSet>"+"".join(
        f"<DescriptorRecord><DescriptorUI>{ui}</DescriptorUI><TreeNumberList>"+"".join(f"<TreeNumber>{t}</TreeNumber>" for t in ts)+"</TreeNumberList></DescriptorRecord>"
        for ui,ts in [("R",["A01"]),("S",["B01"]),("C",["A01.001","B01.001"]),("I",[])])+"</DescriptorRecordSet>")
    nodes,edges=mesh_descriptors(p)
    assert edges=={("R","C"),("S","C")}
    assert audit(nodes,edges)["multi_parent"]==1
    assert audit(nodes,edges)["isolated"]==1
    p.write_text("<DescriptorRecordSet><DescriptorRecord/></DescriptorRecordSet>")
    with pytest.raises(ValueError): mesh_descriptors(p)
    with pytest.raises(ValueError): audit({"a","b"},{("a","b"),("b","a")})


def test_strict_split():
    edges={(0,1),(1,2),(0,2),(2,3),(3,4),(0,4),(1,4),(4,5),(5,0)}
    for seed in range(20):
        split=split_relations(edges,seed,.5)
        assert split==split_relations(edges,seed,.5)
        assert set(split["valid"]).isdisjoint(split["test"])
        for a,b in split["valid"]+split["test"]:
            assert (a,b) not in split["train"] and (b,a) not in split["train"]
            assert not reachable(split["train"],a,b)
            assert not reachable(split["train"],b,a)


def test_mesh_ambiguous_official_position_quarantined(tmp_path):
    p=tmp_path/"collision.xml"
    p.write_text("<DescriptorRecordSet>"+"".join(
        f"<DescriptorRecord><DescriptorUI>{ui}</DescriptorUI><TreeNumberList><TreeNumber>{t}</TreeNumber></TreeNumberList></DescriptorRecord>"
        for ui,t in [("R","B03"),("A","B03.001"),("B","B03.001"),("C","B03.001.001")])+"</DescriptorRecordSet>")
    diagnostics={}
    nodes,edges=mesh_descriptors(p,diagnostics)
    assert len(nodes)==4 and not edges
    assert diagnostics["ambiguous_positions"]=={"B03.001":["A","B"]}
    assert diagnostics["quarantined_candidate_links"]==4


def test_model_full_and_empty():
    torch.manual_seed(11); model=TinyGNN(); x=torch.randn(3,4)
    neighbors=[[1,2],[0],[]]
    a=model(x,neighbors,torch.Generator().manual_seed(1),9,"none")
    b=model(x,neighbors,torch.Generator().manual_seed(2),9,"third")
    assert torch.equal(a[0],b[0])
    a[0].sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())


def test_small_pipeline_reproducible():
    a=run(steps=2); b=run(steps=2)
    for method in a["training"]:
        assert a["training"][method]["train_losses"]==b["training"][method]["train_losses"]
        assert a["training"][method]["clipping_rate"]==0
