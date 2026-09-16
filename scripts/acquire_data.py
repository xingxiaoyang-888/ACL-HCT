"""Download official pinned releases, then audit. No extraction to arbitrary paths."""
import argparse
import datetime
import hashlib
import gzip
import json
from pathlib import Path
import subprocess
import tarfile
from acl_hct.data import wordnet_nouns, mesh_descriptors, audit

SOURCES={
 "wordnet": {"version":"3.0", "url":"https://wordnetcode.princeton.edu/3.0/WordNet-3.0.tar.bz2", "file":"WordNet-3.0.tar.bz2", "license":"https://wordnet.princeton.edu/license-and-commercial-use"},
 "mesh": {"version":"2026", "url":"https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/desc2026.gz", "file":"desc2026.gz", "license":"https://www.nlm.nih.gov/databases/download/terms_and_conditions_mesh.html"}}


def main():
    p=argparse.ArgumentParser(); p.add_argument("--root",type=Path,default=Path("data")); p.add_argument("--download-only",action="store_true"); p.add_argument("--audit-only",action="store_true"); a=p.parse_args()
    a.root.mkdir(parents=True,exist_ok=True)
    for name,source in SOURCES.items():
        target=a.root/source["file"]
        if not a.audit_only:
            subprocess.run(["curl","--fail","--location","--retry","3","--connect-timeout","20","--max-time","900","--continue-at","-","--output",str(target),source["url"]],check=True)
        if a.download_only: continue
        digest=hashlib.sha256()
        with target.open("rb") as f:
            for block in iter(lambda:f.read(1024*1024),b""): digest.update(block)
        if name=="wordnet":
            with tarfile.open(target) as archive:
                matches=[m for m in archive.getmembers() if m.name.endswith("/dict/data.noun")]
                if len(matches)!=1: raise ValueError("unexpected WordNet layout")
                stream=archive.extractfile(matches[0])
                nodes,edges=wordnet_nouns(line.decode("utf-8") for line in stream)
        else:
            with gzip.open(target, "rb") as stream:
                nodes,edges=mesh_descriptors(stream)
        result={**source,"retrieved_audited_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "sha256":digest.hexdigest(),"bytes":target.stat().st_size,"structure":audit(nodes,edges),
            "relation":"parent->child; WordNet @/@i noun hypernyms or MeSH descriptor-collapsed immediate tree parents",
            "hash_note":"locally computed integrity record, not an official signed checksum"}
        (a.root/(name+"-manifest.json")).write_text(json.dumps(result,indent=2)+"\n")
        print(json.dumps(result),flush=True)

if __name__=="__main__": main()
