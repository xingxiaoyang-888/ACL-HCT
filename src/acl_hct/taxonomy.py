"""Rich taxonomy records, kept separate from model input text and observed edges."""
from dataclasses import dataclass
from collections import defaultdict
import xml.etree.ElementTree as ET
from .data import wordnet_nouns, audit


@dataclass
class Taxonomy:
    records: dict
    edges: set
    diagnostics: dict

    def validate(self):
        return audit(set(self.records), self.edges)


def wordnet_records(lines):
    records = {}
    def capture():
        for line in lines:
            yield line
            if not line.strip() or line[0].isspace(): continue
            raw, _, definition = line.partition('|')
            fields = raw.split()
            # The edge parser validates this record before generator resumes.
            n = int(fields[3],16)
            records[fields[0]] = {'name':' '.join(fields[4+2*i].replace('_',' ') for i in range(n)),
                                  'definition':definition.strip(), 'tree_positions':[], 'path_ancestors':{}}
    nodes, edges = wordnet_nouns(capture())
    if nodes != set(records): raise ValueError('record identity mismatch')
    result=Taxonomy(records,edges,{'text_fields':['synset_lemmas','gloss'], 'relation':'noun @ and @i'})
    result.validate(); return result


def mesh_records(path):
    """Keep every original position; explicitly mark ambiguous ancestry.

    Names/preferred-concept scope notes are text. Tree codes/ancestor IDs are
    evaluation metadata only. Ambiguous owner positions never select an owner.
    """
    records={}; owners=defaultdict(set); duplicate_occurrences=0
    for _, element in ET.iterparse(path,events=('end',)):
        if element.tag!='DescriptorRecord': continue
        ui=(element.findtext('DescriptorUI') or '').strip()
        if not ui or ui in records: raise ValueError('missing or duplicate DescriptorUI')
        positions=[]
        for tree in element.findall('./TreeNumberList/TreeNumber'):
            value=(tree.text or '').strip()
            if not value: raise ValueError('missing tree position')
            duplicate_occurrences += int(ui in owners[value]); owners[value].add(ui); positions.append(value)
        notes=[(node.text or '').strip() for node in element.findall("./ConceptList/Concept[@PreferredConceptYN='Y']/ScopeNote")]
        records[ui]={'name':(element.findtext('./DescriptorName/String') or '').strip(),
                     'definition':' '.join(notes), 'tree_positions':sorted(set(positions)), 'path_ancestors':{}}
        element.clear()
    ambiguous={path:sorted(ids) for path,ids in owners.items() if len(ids)>1}
    edges=set(); quarantined=0
    for path,children in owners.items():
        if '.' not in path: continue
        parent_path=path.rsplit('.',1)[0]
        if parent_path not in owners: raise ValueError('missing parent tree position')
        if path in ambiguous or parent_path in ambiguous:
            quarantined+=len(children)*len(owners[parent_path]); continue
        parent=next(iter(owners[parent_path])); child=next(iter(children))
        if parent!=child: edges.add((parent,child))
    for ui,record in records.items():
        record['ambiguous_prefixes']=[]
        for path in record['tree_positions']:
            parts=path.split('.'); ancestors=[]
            for length in range(1,len(parts)):
                prefix='.'.join(parts[:length])
                if prefix not in owners: raise ValueError('missing ancestor tree position')
                if prefix in ambiguous: record['ambiguous_prefixes'].append(prefix)
                else:
                    ancestor=next(iter(owners[prefix]))
                    if ancestor!=ui: ancestors.append(ancestor)
            record['path_ancestors'][path]=sorted(set(ancestors))
            if path in ambiguous: record['ambiguous_prefixes'].append(path)
        record['ambiguous_prefixes']=sorted(set(record['ambiguous_prefixes']))
    result=Taxonomy(records,edges,{'tree_positions':len(owners),'ambiguous_positions':ambiguous,'quarantined_candidate_links':quarantined,
                                  'duplicate_same_owner_occurrences':duplicate_occurrences,
                                  'text_fields':['DescriptorName/String','preferred Concept/ScopeNote']})
    result.validate(); return result
