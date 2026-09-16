"""Strict train+valid positive development view and outcome-independent panels."""
from collections import defaultdict, deque
from dataclasses import dataclass
import json
from pathlib import Path
import random
from .protocols import digest, grouped_split
from .backbone import validate_neighbors

DEGREE_LABELS=('0','1-2','3-4','5-8','9-16','17-32','33+')


def degree_bin(n):
    return next(i for i,upper in enumerate((0,2,4,8,16,32,float('inf'))) if n<=upper)


@dataclass
class DevelopmentView:
    nodes: list
    neighbors: list
    train_edges: set
    valid_edges: set
    valid_entities: list
    edges: set
    parents: dict
    children: dict
    root: str | None
    reachable: set
    shortest: dict
    longest: dict
    metadata: dict

    def ancestors(self, child):
        seen=set(); stack=list(self.parents[child])
        while stack:
            node=stack.pop()
            if node not in seen:
                seen.add(node);stack.extend(self.parents[node])
        return seen


def build_view(nodes, neighbors, train_edges, valid_edges, valid_entities, max_reach_visits=2000000):
    """Exact root choice or explicit cost failure; never approximate a winning root."""
    if type(max_reach_visits) is not int or max_reach_visits<1:raise ValueError('positive reachability work bound required')
    nodes=list(nodes);known=set(nodes)
    if len(known)!=len(nodes) or any(not isinstance(n,str) for n in nodes): raise ValueError('unique string entity IDs required')
    validate_neighbors(neighbors,len(nodes))
    train=set(map(tuple,train_edges));valid=set(map(tuple,valid_edges));entities=sorted(set(valid_entities))
    if not set(entities).issubset(known): raise ValueError('unknown valid entity')
    if train & valid or any(b in entities for a,b in train) or any(b not in entities for a,b in valid):
        raise ValueError('relations must follow train/valid child grouping')
    edges=train | valid
    if any(a not in known or b not in known or a==b for a,b in edges): raise ValueError('invalid development relation')
    parents={node:set() for node in nodes};children={node:set() for node in nodes}
    for a,b in edges:parents[b].add(a);children[a].add(b)
    indegree={node:len(parents[node]) for node in nodes};remaining=indegree.copy()
    queue=deque(sorted(node for node in nodes if not indegree[node]));order=[]
    while queue:
        node=queue.popleft();order.append(node)
        for child in sorted(children[node]):
            remaining[child]-=1
            if remaining[child]==0:queue.append(child)
    if len(order)!=len(nodes): raise ValueError('H_dev must be a DAG; no fabricated depths')
    candidates=sorted(node for node in nodes if not indegree[node] and children[node])
    visits=0;counts={};best=None;reachable=set()
    for root in candidates:
        seen={root};stack=[root]
        while stack:
            node=stack.pop()
            for child in children[node]:
                visits+=1
                if visits>max_reach_visits: raise ValueError('root reachability exceeds registered work bound')
                if child not in seen:seen.add(child);stack.append(child)
        counts[root]=len(seen)-1
        if best is None or len(seen)>len(reachable):best=root;reachable=seen
    shortest={};longest={}
    if best is not None:
        shortest[best]=0;queue=deque([best])
        while queue:
            node=queue.popleft()
            for child in sorted(children[node]):
                if child not in shortest:shortest[child]=shortest[node]+1;queue.append(child)
        longest[best]=0
        for node in order:
            if node in longest:
                for child in children[node]:longest[child]=max(longest.get(child,0),longest[node]+1)
    metadata={'label_scope':'H_dev=train positives + valid positives only; missing path is unknown, not a negative',
              'h_dev_hash':digest(sorted(edges)),'graph_hash':digest(neighbors),'node_order_hash':digest(nodes),
              'reference_root':best,'root_definition':'zero indegree with descendants; largest distinct descendant count; ID tie-break',
              'root_candidate_count':len(candidates),'root_candidate_descendant_counts':counts,
              'reachable_including_root':len(reachable),'reachable_fraction':len(reachable)/len(nodes),
              'root_reach_edge_visits':visits,'depth_scope':'truncated H_dev reference-root paths, not full taxonomy depth',
              'root_unknown_nodes':len(nodes)-len(reachable)}
    return DevelopmentView(nodes,neighbors,train,valid,entities,edges,parents,children,best,reachable,shortest,longest,metadata)


def load_development_view(prepared_root, max_reach_visits=2000000):
    """Allowlist of five JSON inputs. Never opens test, full truth, text, or features."""
    root=Path(prepared_root)
    def read(name):return json.loads((root/name).read_text(encoding='utf-8'))
    manifest=read('input_manifest.json');graph=read('observed_graph.json')
    queries=read('train_queries.json');valid=read('evaluator_valid.json');entities=read('entity_split.json')
    nodes=graph['nodes'];known=set(nodes)
    if manifest['protocol']!='B-child-grouped-80-10-10-v1' or manifest['split_seed']!=20260914:
        raise ValueError('frozen protocol B V1 required')
    validate_neighbors(graph['neighbors'],len(nodes))
    if entities!=grouped_split(nodes,[],20260914)['entities']:
        raise ValueError('entity split differs from frozen outcome-independent assignment')
    if (digest(nodes)!=manifest['node_order_hash'] or digest(graph['neighbors'])!=manifest['graph_hash']
            or digest(queries)!=manifest['train_queries_hash']): raise ValueError('prepared hash mismatch')
    parts=[set(entities[key]) for key in ('train','valid','test')]
    if any(len(parts[i])!=len(entities[key]) for i,key in enumerate(('train','valid','test'))):
        raise ValueError('duplicated split entity')
    if set.union(*parts)!=known or any(parts[i]&parts[j] for i in range(3) for j in range(i)):
        raise ValueError('entity grouping is not a partition')
    if sorted(parts[0])!=sorted(manifest['text_fit_entities']): raise ValueError('training entity mismatch')
    if len(queries['queries'])!=len(queries['labels']) or any(label not in (0,1) for label in queries['labels']):
        raise ValueError('invalid supervised query labels')
    train=[tuple(pair) for pair,label in zip(queries['queries'],queries['labels']) if label==1]
    if any(b not in parts[0] for a,b in train): raise ValueError('training positive has nontraining child')
    visible={(nodes[j],nodes[i]) for i,row in enumerate(graph['neighbors']) for j in row}
    expected=set(train) | {(b,a) for a,b in train}
    if visible!=expected: raise ValueError('G_obs differs from training positive edges plus reverse')
    view=build_view(nodes,graph['neighbors'],train,valid,entities['valid'],max_reach_visits)
    view.metadata['prepared_manifest_hash']=digest(manifest)
    view.metadata['valid_queries_hash']=digest(valid)
    view.metadata['entity_split_hash']=digest(entities)
    view.metadata['loaded_files']=['input_manifest.json','observed_graph.json','train_queries.json','evaluator_valid.json','entity_split.json']
    return view


def support_groups(neighbors, fanout):
    if type(fanout) is not int or fanout<1: raise ValueError('positive fanout required')
    validate_neighbors(neighbors,len(neighbors))
    affected={i for i,row in enumerate(neighbors) if len(row)>fanout}
    potential=affected | {i for i,row in enumerate(neighbors) if any(j in affected for j in row)}
    return {'V':list(range(len(neighbors))),'A':sorted(affected),'P':sorted(potential),
            'P_minus_A':sorted(potential-affected),'V_minus_P':sorted(set(range(len(neighbors)))-potential)}


def _panel(view,pool,target,seed):
    index={node:i for i,node in enumerate(view.nodes)}
    strata=[[] for _ in DEGREE_LABELS]
    for node in sorted(pool):strata[degree_bin(len(view.neighbors[index[node]]))].append(node)
    target=min(target,len(pool));occupied=sum(bool(row) for row in strata)
    if target<occupied: raise ValueError('panel target must cover every nonempty stratum')
    quota=[0]*len(strata)
    while sum(quota)<target:
        for h,row in enumerate(strata):
            if quota[h]<len(row) and sum(quota)<target:quota[h]+=1
    rng=random.Random(seed);rows=[];summary=[]
    for h,population in enumerate(strata):
        chosen=sorted(rng.sample(population,quota[h]))
        summary.append({'degree':DEGREE_LABELS[h],'population':len(population),'selected':len(chosen)})
        for node in chosen:
            probability=quota[h]/len(population)
            rows.append({'id':node,'index':index[node],'stratum':DEGREE_LABELS[h],
                         'inclusion_probability':probability,'pool_mean_weight':1/(len(pool)*probability),
                         'h_dev_shortest':view.shortest.get(node),'h_dev_longest':view.longest.get(node),
                         'h_dev_depth_gap':view.longest[node]-view.shortest[node] if node in view.reachable else None,
                         'reference_root_reachable':node in view.reachable,'multi_parent':len(view.parents[node])>1})
    result={'pool':sorted(pool),'pool_size':len(pool),'target':target,'seed':seed,'strata':summary,'rows':rows}
    result['panel_hash']=digest(result);return result


def make_panels(view,target=1000):
    if type(target) is not int or not 1<=target<=1000: raise ValueError('panel target must be 1..1000')
    shuffled=sorted(view.valid_entities);random.Random(2026091601).shuffle(shuffled);middle=len(shuffled)//2
    panels={'development':_panel(view,shuffled[:middle],target,2026091602),
            'diagnostic_confirmation':_panel(view,shuffled[middle:],target,2026091603)}
    rng=random.Random(2026091604)
    direct_by_child=defaultdict(list)
    for parent,child in sorted(view.valid_edges):direct_by_child[child].append(parent)
    for panel in panels.values():
        relations=[]
        for row in sorted(panel['rows'],key=lambda item:item['id']):
            child=row['id'];direct=direct_by_child[child]
            distant=sorted(view.ancestors(child)-view.parents[child])
            extra=sorted(rng.sample(distant,min(4,len(distant))))
            relations.append({'child':child,'direct_parents':direct,'positive_distant_ancestors':extra,
                              'available_distant_ancestors':len(distant)})
        panel['relations']=relations;panel['relation_seed']=2026091604
        panel['relation_hash']=digest(relations)
    return {'protocol':'E2-E3-development-v1','pool_seed':2026091601,'panels':panels,
            'scope':'diagnostic pools within valid; not untouched task test entities',
            'h_dev_hash':view.metadata['h_dev_hash'],'hash':digest(panels)}
