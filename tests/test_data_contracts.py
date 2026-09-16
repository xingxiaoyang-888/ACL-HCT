import importlib.util
from pathlib import Path
import subprocess
import pytest
from acl_hct.data import wordnet_nouns, mesh_descriptors, audit

@pytest.mark.parametrize('record', [
    '00000001 00 n ff root 0', '00000001 00 n 01 root 0 -1',
    '00000001 00 n 01 root 0 001 @ 00000002 n',
    '00000001 00 n 01 root 0 001 @ 00000002 n 0101'])
def test_explained_wordnet_errors(record):
    with pytest.raises(ValueError, match='WordNet line 1'):
        wordnet_nouns([record])

def test_wordnet_multiple_parents_instance_and_duplicates():
    roots = ['00000001 00 n 01 root 0 000', '00000002 00 n 01 other 0 000']
    child = '00000003 00 n 01 child 0 002 @ 00000001 n 0000 @i 00000002 n 0000'
    nodes, edges = wordnet_nouns(roots+[child])
    assert audit(nodes,edges)['multi_parent'] == 1
    with pytest.raises(ValueError, match='duplicate synset'):
        wordnet_nouns(roots+[child,child])

def test_download_failure_resume_and_atomic_completion(tmp_path, monkeypatch):
    spec=importlib.util.spec_from_file_location('acquire',Path(__file__).parents[1]/'scripts/acquire_data.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    target=tmp_path/'archive.gz'; partial=tmp_path/'archive.gz.part'
    def fail(args, check):
        assert '--continue-at' in args and args[args.index('--output')+1] == str(partial)
        partial.write_bytes(b'prefix')
        raise subprocess.CalledProcessError(18,args)
    monkeypatch.setattr(module.subprocess,'run',fail)
    with pytest.raises(subprocess.CalledProcessError): module.download({'url':'https://example.invalid'},target)
    assert not target.exists() and partial.read_bytes() == b'prefix'
    def finish(args, check):
        assert partial.read_bytes() == b'prefix'
        partial.write_bytes(b'complete')
    monkeypatch.setattr(module.subprocess,'run',finish)
    module.download({'url':'https://example.invalid'},target)
    assert target.read_bytes() == b'complete' and not partial.exists()
    monkeypatch.setattr(module.subprocess,'run',fail)
    module.download({'url':'https://example.invalid'},target)
    assert target.read_bytes() == b'complete'
