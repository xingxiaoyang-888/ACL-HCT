"""Official hierarchy parsers; parent->child edges, no hidden downloads."""
from collections import defaultdict, deque
import xml.etree.ElementTree as ET
import random


def wordnet_nouns(lines):
    nodes, edges = set(), set()
    for line in lines:
        if not line.strip() or line[0].isspace():
            continue
        fields = line.split("|", 1)[0].split()
        if len(fields) < 5 or not fields[0].isdigit() or fields[2] != "n":
            raise ValueError("malformed noun record")
        node = fields[0]
        nodes.add(node)
        pos = 4 + 2*int(fields[3], 16)
        count = int(fields[pos]); pos += 1
        if len(fields) < pos + 4*count:
            raise ValueError("truncated pointer list")
        for j in range(count):
            symbol, target, kind, lexical = fields[pos+4*j:pos+4*j+4]
            if symbol in ("@", "@i") and kind == "n":
                if lexical != "0000":
                    raise ValueError("hypernym pointer must be semantic")
                edges.add((target, node))
    if any(a not in nodes or b not in nodes for a,b in edges):
        raise ValueError("missing hypernym node")
    return nodes, edges


def mesh_descriptors(path, diagnostics=None):
    """Preserve descriptors; quarantine links incident to ambiguous tree positions.

    Official releases can assign one tree position to multiple descriptors.
    Do not silently overwrite an owner or fabricate a unique hierarchy.
    """
    nodes, trees = set(), defaultdict(set)
    duplicate_occurrences = 0
    for _, e in ET.iterparse(path, events=("end",)):
        if e.tag != "DescriptorRecord":
            continue
        ui = e.findtext("DescriptorUI")
        if not ui:
            raise ValueError("missing DescriptorUI")
        if ui in nodes:
            raise ValueError("duplicate descriptor")
        nodes.add(ui)
        for t in e.findall("./TreeNumberList/TreeNumber"):
            if not t.text:
                raise ValueError("missing tree position")
            duplicate_occurrences += int(ui in trees[t.text])
            trees[t.text].add(ui)
        e.clear()
    edges = set()
    ambiguous = {p: sorted(ids) for p, ids in trees.items() if len(ids) > 1}
    quarantined_links = 0
    for path, children in trees.items():
        if "." in path:
            parent_path = path.rsplit(".", 1)[0]
            if parent_path not in trees:
                raise ValueError("missing parent tree position")
            if path in ambiguous or parent_path in ambiguous:
                quarantined_links += len(children) * len(trees[parent_path])
                continue
            parent = next(iter(trees[parent_path]))
            child = next(iter(children))
            if parent != child:
                edges.add((parent, child))
    if diagnostics is not None:
        diagnostics.update({"tree_positions": len(trees),
                            "ambiguous_positions": ambiguous,
                            "quarantined_candidate_links": quarantined_links,
                            "duplicate_same_owner_occurrences": duplicate_occurrences})
    return nodes, edges


def audit(nodes, edges):
    children, indeg = defaultdict(set), {n: 0 for n in nodes}
    for a,b in edges:
        if a not in nodes or b not in nodes or a == b:
            raise ValueError("invalid edge")
        if b not in children[a]:
            children[a].add(b); indeg[b] += 1
    roots = sorted(n for n in nodes if indeg[n] == 0)
    depth = {n:0 for n in roots}; queue = deque(roots); remaining = indeg.copy()
    while queue:
        a = queue.popleft()
        for b in children[a]:
            depth[b] = max(depth.get(b, 0), depth[a]+1)
            remaining[b] -= 1
            if remaining[b] == 0:
                queue.append(b)
    if len(depth) != len(nodes) or any(remaining.values()):
        raise ValueError("descriptor-collapsed hierarchy contains cycles")
    return {"nodes":len(nodes), "edges":len(edges), "roots":len(roots),
            "isolated":sum(indeg[n]==0 and not children[n] for n in nodes),
            "multi_parent":sum(v>1 for v in indeg.values()),
            "max_longest_root_depth":max(depth.values(), default=0),
            "max_out_degree":max(map(len, children.values()), default=0)}


def reachable(edges, start, end):
    adj = defaultdict(list)
    for a,b in edges: adj[a].append(b)
    seen, stack = {start}, [start]
    while stack:
        for b in adj[stack.pop()]:
            if b == end: return True
            if b not in seen: seen.add(b); stack.append(b)
    return False


def split_relations(edges, seed=11, holdout_fraction=.2):
    """Small-graph strict split, NOT scalable to the full corpora yet.

    Message edges are directed train relations only. Held-out relations whose
    endpoints remain connected by a directed training path are quarantined,
    not scored; reverse pairs are removed before scoring. Never add closure.
    """
    if not 0 < holdout_fraction < 1: raise ValueError("invalid fraction")
    pairs = sorted(set(edges)); random.Random(seed).shuffle(pairs)
    size = max(1, int(len(pairs)*holdout_fraction)) if pairs else 0
    held, train = pairs[:size], set(pairs[size:])
    train -= {(b,a) for a,b in held}
    safe = [e for e in held if not reachable(train, *e) and not reachable(train, e[1], e[0])]
    middle = len(safe)//2
    return {"train":sorted(train), "valid":safe[:middle], "test":safe[middle:],
            "quarantined":sorted(set(held)-set(safe))}


def synthetic_tree(levels=3, branching=3):
    edges, depth, branch = [], [0], [-1]
    frontier=[0]
    for level in range(1, levels+1):
        nxt=[]
        for p in frontier:
            for j in range(branching):
                node=len(depth); depth.append(level)
                branch.append(j if p==0 else branch[p]); edges.append((p,node)); nxt.append(node)
        frontier=nxt
    return edges, depth, branch
