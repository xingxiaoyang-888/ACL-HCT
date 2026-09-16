"""Read-only full-corpus sampling coverage, using a pinned parser revision."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import types

DEFAULT_REVISION = "9e610cbc16acfeff0d0790c63a0a11d28c4b9cb6"


def load_parser(revision):
    root = Path(__file__).resolve().parents[1]
    parser = subprocess.check_output(["git", "show", f"{revision}:src/acl_hct/data.py"], cwd=root)
    reference = types.ModuleType("pinned_reference_data")
    exec(compile(parser, "pinned_reference_data.py", "exec"), reference.__dict__)
    return reference, parser


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def quantiles(values):
    values = sorted(values)
    if not values:
        return {}
    result = {}
    for q in (0, .25, .5, .75, .9, .95, .99, 1):
        position = q * (len(values) - 1)
        lower = int(position)
        upper = min(lower + 1, len(values) - 1)
        result[str(q)] = values[lower] + (values[upper] - values[lower]) * (position - lower)
    return result


def summarize(degrees):
    n = len(degrees)
    total = sum(degrees)
    result = {
        "nodes": n, "candidate_messages": total,
        "empty_nodes": sum(d == 0 for d in degrees),
        "degree_quantiles": quantiles(degrees),
        "degree_histogram": dict(sorted(Counter(degrees).items())),
        "fanouts": [],
    }
    for f in (4, 8, 16, 32, 64):
        affected = [d for d in degrees if d > f]
        removed = sum(max(0, d - f) for d in degrees)
        result["fanouts"].append({
            "fanout": f, "affected_nodes": len(affected),
            "affected_node_fraction": len(affected) / n if n else None,
            "removed_messages": removed,
            "removed_message_fraction": removed / total if total else None,
            "k_over_N_nonempty_quantiles": quantiles([min(d, f) / d for d in degrees if d]),
            "k_over_N_affected_quantiles": quantiles([f / d for d in affected]),
        })
    return result


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--data-root", type=Path, default=Path("data"))
    cli.add_argument("--manifest-root", type=Path, default=Path("reports"))
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--revision", default=DEFAULT_REVISION)
    args = cli.parse_args()
    reference, parser_source = load_parser(args.revision)
    results = {
        "audited_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": args.revision,
        "parser_sha256": hashlib.sha256(parser_source).hexdigest(),
        "scope": "Original full taxonomy graph only, before formal train/validation/test split. No training or GPU use.",
        "self_policy": "No added self loops. Zero-neighbor self fallback does not count as an original candidate message.",
        "message_directions": {
            "parents_to_child": "Each node receives its semantic parents (in-degree in parent->child graph).",
            "children_to_parent": "Each node receives its semantic children (out-degree in parent->child graph).",
            "undirected": "Each node receives both parents and children; deduplicated, no added edges besides reverse direction.",
        },
        "datasets": {},
    }
    for name, filename in (("wordnet", "WordNet-3.0.tar.bz2"), ("mesh", "desc2026.gz")):
        path = args.data_root / filename
        published = json.loads((args.manifest_root / f"{name}-manifest.json").read_text(encoding="utf-8"))
        digest = sha256(path)
        if digest != published["sha256"]:
            raise ValueError(f"{name}: raw file no longer matches published manifest")
        extra = {}
        if name == "wordnet":
            with tarfile.open(path) as archive:
                members = [m for m in archive.getmembers() if m.name.endswith("/dict/data.noun")]
                if len(members) != 1:
                    raise ValueError("unexpected WordNet archive layout")
                with archive.extractfile(members[0]) as stream:
                    nodes, edges = reference.wordnet_nouns(line.decode("utf-8") for line in stream)
        else:
            with gzip.open(path, "rb") as stream:
                nodes, edges = reference.mesh_descriptors(stream, extra)
        structure = reference.audit(nodes, edges)
        if structure != published["structure"]:
            raise ValueError(f"{name}: pinned-parser structure differs from published evidence")
        ordered = sorted(nodes)
        incoming, outgoing = Counter(), Counter()
        for parent, child in edges:
            outgoing[parent] += 1
            incoming[child] += 1
        if any((b, a) in edges for a, b in edges):
            raise ValueError("unexpected reciprocal edges in taxonomy DAG")
        if sum(incoming.values()) != len(edges) or sum(outgoing.values()) != len(edges):
            raise AssertionError("edge-degree count mismatch")
        entry = {"source": {key: published[key] for key in ("url", "version", "license")}, "raw_sha256": digest, "structure": structure, "parser_diagnostics": extra, "directions": {}}
        for direction, degrees in (
            ("parents_to_child", [incoming[v] for v in ordered]),
            ("children_to_parent", [outgoing[v] for v in ordered]),
            ("undirected", [incoming[v] + outgoing[v] for v in ordered]),
        ):
            entry["directions"][direction] = summarize(degrees)
        results["datasets"][name] = entry
    target = args.output
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("dataset direction fanout affected_nodes affected_pct removed_message_pct")
    for name, dataset in results["datasets"].items():
        for direction, stats in dataset["directions"].items():
            for row in stats["fanouts"]:
                print(name, direction, row["fanout"], row["affected_nodes"],
                      f'{row["affected_node_fraction"]*100:.3f}',
                      f'{row["removed_message_fraction"]*100:.3f}')


if __name__ == "__main__":
    main()
