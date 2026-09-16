"""Protocol B preparation. No transitive-reachability pruning or test-label filter."""
import hashlib
import json
import random
from collections import defaultdict
import numpy as np


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()


def grouped_split(node_ids, edges, seed=20260914):
    """Assign all entities 80/10/10; all parent relations group by child ID.

    Roots/isolates are assigned too, defining the allowed text-fit entities.
    The graph is transductive: entity IDs/text exist at transform time. A held-
    out child can be a parent of a training child through a visible train edge.
    """
    nodes=sorted(set(node_ids)); edges=sorted(set(map(tuple,edges)))
    if len(nodes)<10: raise ValueError('protocol B requires >=10 entities for nonempty 80/10/10 groups')
    known=set(nodes)
    if any(a not in known or b not in known or a==b for a,b in edges): raise ValueError('invalid relation')
    shuffled=nodes.copy(); random.Random(seed).shuffle(shuffled)
    first=int(.8*len(nodes)); second=first+int(.1*len(nodes))
    entities={'train':sorted(shuffled[:first]),'valid':sorted(shuffled[first:second]),'test':sorted(shuffled[second:])}
    group={node:name for name,ids in entities.items() for node in ids}
    relations={name:[(a,b) for a,b in edges if group[b]==name] for name in entities}
    observed=sorted(set(relations['train']) | {(b,a) for a,b in relations['train']})
    held=set(relations['valid']+relations['test'])
    if any(edge in held or edge[::-1] in held for edge in observed):
        raise ValueError('reciprocal truth violates held-out reverse-edge exclusion')
    result={'protocol':'B-child-grouped-80-10-10-v1','seed':seed,'entities':entities,
            'relations':relations,'observed_directed_edges':observed,
            'message_policy':'training parent-child relations plus reverse; no additional self loops'}
    result['split_hash']=digest(result)
    return result


def observed_neighbors(node_ids, train_edges, positive_queries=()):
    """Mask every batch positive query and reverse before computing N/sampling."""
    nodes=list(node_ids); index={node:i for i,node in enumerate(nodes)}
    if len(index)!=len(nodes): raise ValueError('duplicate node IDs')
    blocked=set(map(tuple,positive_queries)); blocked |= {(b,a) for a,b in blocked}
    neighbors=[set() for _ in nodes]
    for a,b in train_edges:
        if a not in index or b not in index or a==b: raise ValueError('invalid train edge')
        if (a,b) not in blocked:
            neighbors[index[a]].add(index[b]); neighbors[index[b]].add(index[a])
    return [sorted(row) for row in neighbors]


def training_candidates(node_ids, train_entities, train_edges, negatives_per_positive=4, seed=11):
    """Fixed uniform negatives; only this function's train labels are consulted.

    Because all true parents of a train child are in the same group, exclusion
    needs no validation/test labels. Distinct negatives per positive, no self.
    """
    nodes=sorted(set(node_ids)); known=set(nodes); allowed=set(train_entities); edges=sorted(set(map(tuple,train_edges)))
    if type(negatives_per_positive) is not int or negatives_per_positive<1: raise ValueError('invalid negative count')
    if not allowed.issubset(nodes): raise ValueError('unknown training entity')
    if any(a not in known or b not in allowed or a==b for a,b in edges): raise ValueError('queries must be training-child relations')
    parents=defaultdict(set)
    for a,b in edges: parents[b].add(a)
    generator=random.Random(seed); queries=[]; labels=[]
    for a,b in edges:
        forbidden=parents[b] | {b}
        if len(nodes)-len(forbidden)<negatives_per_positive: raise ValueError('insufficient distinct negative candidates')
        selected=[]; seen=set(); attempts=0
        while len(selected)<negatives_per_positive and attempts<100*(negatives_per_positive+1):
            candidate=nodes[generator.randrange(len(nodes))]; attempts+=1
            if candidate not in forbidden and candidate not in seen:
                selected.append(candidate); seen.add(candidate)
        if len(selected)<negatives_per_positive:
            pool=[node for node in nodes if node not in forbidden and node not in seen]
            selected+=generator.sample(pool,negatives_per_positive-len(selected))
        queries.append((a,b)); labels.append(1)
        for negative in selected:
            queries.append((negative,b)); labels.append(0)
    return {'queries':queries,'labels':labels,'negatives_per_positive':negatives_per_positive,
            'candidate_policy':'uniform fixed training-only true-parent exclusion; distinct per positive','seed':seed}


def fit_text_features(records, train_entities, dimension=128, seed=11):
    """Fit vocabulary, IDF and SVD only on train entity name/definition fields."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    import sklearn
    nodes=sorted(records); train=sorted(set(train_entities))
    if type(dimension) is not int or dimension<1 or len(train)<2 or any(node not in records for node in train):
        raise ValueError('invalid text-fit entities/dimension')
    texts={node:' '.join(str(records[node].get(field,'')) for field in ('name','definition')) for node in nodes}
    vectorizer=TfidfVectorizer(lowercase=True, dtype=np.float64)
    train_matrix=vectorizer.fit_transform([texts[node] for node in train])
    effective=min(dimension,len(train)-1,train_matrix.shape[1]-1)
    if effective<1: raise ValueError('insufficient training text vocabulary for SVD')
    reducer=TruncatedSVD(n_components=effective,random_state=seed)
    reducer.fit(train_matrix)
    features=reducer.transform(vectorizer.transform([texts[node] for node in nodes])).astype(np.float32)
    state={'vocabulary':{word:int(index) for word,index in vectorizer.vocabulary_.items()},'idf':vectorizer.idf_.copy(),'components':reducer.components_.copy()}
    manifest={'schema':'train-text-tfidf-svd-v1','nodes':nodes,'train_entities':train,
              'text_fields':['name','definition'],'text_hash':digest(texts), 'train_text_hash':digest({n:texts[n] for n in train}),
              'feature_sha256':hashlib.sha256(features.tobytes()).hexdigest(),'fit_state_hash':hashlib.sha256((digest(state['vocabulary'])+hashlib.sha256(state['idf'].tobytes()).hexdigest()+hashlib.sha256(state['components'].tobytes()).hexdigest()).encode()).hexdigest(),
              'requested_dimension':dimension,'effective_dimension':effective,'seed':seed,
              'sklearn':sklearn.__version__,'empty_texts':sum(not text.strip() for text in texts.values())}
    return features, manifest, state


def mask_indexed_queries(neighbors, positive_pairs):
    """Copy only changed rows of an already validated indexed observed graph."""
    blocked=defaultdict(set)
    for a,b in positive_pairs:
        if type(a) is not int or type(b) is not int or not 0<=a<len(neighbors) or not 0<=b<len(neighbors) or a==b:
            raise ValueError('invalid indexed positive query')
        blocked[a].add(b); blocked[b].add(a)
    result=list(neighbors)
    for node,excluded in blocked.items(): result[node]=[i for i in neighbors[node] if i not in excluded]
    return result
