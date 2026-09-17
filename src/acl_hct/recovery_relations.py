"""Descriptive relation localization using existing frozen recovery archives.

No model, sampling, fitting or hypothesis tests. Relation groups are fixed from
reference/input metadata before seeing candidate points. Primary weighting is
the original covered-child weighting; relation subgroups retain those weights.
"""
import math
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch

from .development_view import DEGREE_LABELS
from .encoder_matched_control import validate_ranking
from .frozen_recovery import promoted
from .geometry import distance, dot, norm2
from .vector_structure import RadialPanel


def finite_array(value, shape, name):
    value = np.asarray(value)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(name + ' requires complete finite matching values')
    return value


def child_reciprocal_ranks(ranking, data):
    """Reuse the accepted full-coverage gate, then retain child-level rank values."""
    validate_ranking(ranking, data)
    values = defaultdict(list)
    for row in ranking['rows']:
        values[row['child']].append(1 / row['rank'])
    return {child: math.fsum(group) / len(group) for child, group in values.items()}


def quantile_groups(values, included, prefix):
    """Reference-only linear quartiles; equal values stay together, cuts merge."""
    selected = values[included]
    cuts = np.unique(np.quantile(selected, [.25, .5, .75], method='linear')) if len(selected) else np.array([])
    assignment = np.searchsorted(cuts, values, side='left')
    groups = {prefix + f'/bin{i + 1}': included & (assignment == i)
              for i in range(len(cuts) + 1) if (included & (assignment == i)).any()}
    return groups, cuts.tolist()


class RelationLocalization:
    """A fixed root/panel/coverage design; candidates cannot alter group membership.

Extra relation group masks must be supplied by the predeclared protocol. A mask
has one boolean per COVERED relation, in saved parent/child order. Subgroup means
use the original child weight / original covered relation count, normalized by
the selected mass. They are descriptive subgroups, not replacement primaries.
"""

    @torch.no_grad()
    def __init__(self, view, panel, base, bias, floor, c=1., *, native_full=None):
        if base.dtype != torch.float64 or base.device.type != 'cpu':
            raise ValueError('fixed complete CPU FP64 reference required')
        if (bias.shape != base.shape or bias.dtype != base.dtype or bias.device != base.device
                or not torch.isfinite(bias).all() or (dot(base, bias).abs() > 1e-9).any()):
            raise ValueError('fixed complete finite CPU FP64 bias required')
        if (not isinstance(floor, torch.Tensor) or floor.dtype != torch.float64
                or floor.device.type != 'cpu' or floor.shape != (len(base),)
                or not torch.isfinite(floor).all() or (floor < 0).any()):
            raise ValueError('complete original CPU FP64 numerical floor required')
        self.panel = RadialPanel(view, panel, base, {'V': list(range(len(base)))}, c, floor)
        if self.panel.root is None:
            raise ValueError('the original fixed reference root is required')
        self.base = base.detach().clone()
        self.nodes = list(view.nodes)
        self.started = False
        self.c = c
        self.anchor = self.base[self.panel.root].clone()
        self.reference_radial = (distance(self.anchor, self.base, c).numpy() if native_full is None
                                 else self.radial(native_full))
        self.bias_norm = norm2(bias).sqrt().numpy()
        self.children = np.asarray(self.panel.child_indices, dtype=np.int64)
        items = {row['id']: row for row in panel['rows']}
        self.degree = np.asarray([items[child]['stratum'] for child in self.panel.children])
        if not set(self.degree).issubset(DEGREE_LABELS):
            raise ValueError('original degree stratum labels required')
        self.designs = {}
        depth = [items[child].get('h_dev_shortest') for child in self.panel.children]
        if any(x is not None and (type(x) is not int or x < 0) for x in depth):
            raise ValueError('original nonnegative H_dev shortest depth or unknown required')
        depth_labels = np.asarray(['unknown' if x is None else '0' if x == 0 else '1-3' if x <= 3
                                   else '4-7' if x <= 7 else '8-15' if x <= 15 else '16+' for x in depth])
        self.depth = depth
        child_bias = self.bias_norm[self.children]
        bias_groups, bias_cuts = quantile_groups(child_bias, child_bias > 0, 'bias_positive')
        bias_groups['bias_zero'] = child_bias == 0
        child_groups = {'all': np.ones(len(self.children), dtype=bool)}
        child_groups.update({'degree/' + label: self.degree == label for label in DEGREE_LABELS})
        child_groups.update({'depth/' + label: depth_labels == label
                             for label in ('0', '1-3', '4-7', '8-15', '16+', 'unknown')})
        child_groups.update(bias_groups)
        self.child_groups = child_groups
        self.group_definitions = {'bias_norm_linear_quantile_cuts': bias_cuts,
                                  'quartile_tie_policy': 'unique cuts; equal values lower bin; only nonempty reference bins',
                                  'reference_gap_linear_quantile_cuts': {}}
        for kind, design in self.panel.design.items():
            counts = design['covered']
            owner = design['owner']
            groups = {'all': np.ones(len(owner), dtype=bool)}
            groups.update({'degree/' + label: self.degree[owner] == label for label in DEGREE_LABELS})
            groups.update({'depth/' + label: depth_labels[owner] == label
                           for label in ('0', '1-3', '4-7', '8-15', '16+', 'unknown')})
            full_gap = self.reference_radial[design['child']] - self.reference_radial[design['parent']]
            groups.update({'F_positive': full_gap > 0, 'F_negative': full_gap < 0, 'F_tie': full_gap == 0})
            resolved = np.abs(full_gap) > design['resolution']
            gap_groups, gap_cuts = quantile_groups(np.abs(full_gap), resolved, 'reference_gap_resolved')
            gap_groups['reference_gap_unresolved'] = ~resolved
            groups.update(gap_groups)
            groups.update({name: mask[owner] for name, mask in bias_groups.items()})
            self.group_definitions['reference_gap_linear_quantile_cuts'][kind] = gap_cuts
            self.designs[kind] = {
                'parent': design['parent'].copy(), 'child': design['child'].copy(), 'owner': owner.copy(),
                'reference_gap': full_gap,
                'resolution': design['resolution'].copy(),
                'weight': self.panel.weights[owner] / counts[owner],
                'groups': groups,
            }

    def add_reference_groups(self, groups):
        """Called before analysis; copy fixed masks so later mutations cannot regroup."""
        if self.started:
            raise ValueError('reference groups must freeze before candidate analysis')
        for kind, masks in groups.items():
            if kind not in self.designs:
                raise ValueError('registered direct/distant relation kind required')
            current = self.designs[kind]
            for name, mask in masks.items():
                mask = np.asarray(mask)
                if (not isinstance(name, str) or not name or name in current['groups']
                        or mask.dtype != np.bool_ or mask.shape != current['owner'].shape):
                    raise ValueError('unique named fixed complete boolean reference group required')
                current['groups'][name] = mask.copy()

    def frozen_signature(self):
        """Exact same F/reference groups and relation weights across both shards."""
        from .recovery_registration import canonical
        def identity(value):
            raw = np.ascontiguousarray(value)
            return {'shape': list(raw.shape), 'dtype': raw.dtype.str,
                    'data_sha256': hashlib.sha256(raw.tobytes()).hexdigest()}
        return canonical({'group_definitions': self.group_definitions,
                          'reference_radial': identity(self.reference_radial), 'bias_norm': identity(self.bias_norm),
                          'designs': {kind: {key: {name: identity(mask) for name, mask in value.items()}
                                             if key == 'groups' else identity(value)
                                             for key, value in design.items()}
                                      for kind, design in self.designs.items()}})

    @torch.no_grad()
    def radial(self, native):
        if (not isinstance(native, torch.Tensor) or native.dtype != torch.float32
                or native.device.type != 'cpu' or native.shape != self.base.shape):
            raise ValueError('all archived native FP32 points on CPU required')
        # The old uniformly promoted diagnostic geometry; never recenter on a
        # candidate root and never replace the original native-head embeddings.
        points = promoted(native, self.c)
        return distance(self.anchor, points, self.c).numpy()

    def coverage(self, kind):
        design = self.panel.design[kind]
        covered = design['covered'] > 0
        return {
            'sampled_children': len(self.children), 'covered_children': int(covered.sum()),
            'no_covered_relation_children': int((~covered).sum()),
            'unknown_relations': int((design['available'] - design['covered']).sum()),
            'weighted_covered_child_mass_in_pool': float(self.panel.weights[covered].sum()),
            'weighted_sampled_child_mass_in_pool': float(self.panel.weights.sum()),
        }

    def _summaries(self, kind, fields):
        design = self.designs[kind]
        result = {}
        for name, mask in design['groups'].items():
            weights = design['weight'][mask]
            mass = float(weights.sum())
            metrics = {key: float(np.sum(weights * value[mask]) / mass) if mass else None
                       for key, value in fields.items()}
            result[name] = {'covered_relations': int(mask.sum()),
                            'covered_children': len(set(design['owner'][mask].tolist())),
                            'weighted_relation_mass_in_pool': mass, 'metrics': metrics}
            if name in self.child_groups:
                selected = self.child_groups[name]
                native_design = self.panel.design[kind]
                covered = selected & (native_design['covered'] > 0)
                result[name]['child_coverage'] = {
                    'sampled_children': int(selected.sum()), 'covered_children': int(covered.sum()),
                    'no_covered_relation_children': int((selected & ~covered).sum()),
                    'unknown_relations': int((native_design['available'] - native_design['covered'])[selected].sum()),
                    'weighted_sampled_child_mass_in_pool': float(self.panel.weights[selected].sum()),
                    'weighted_covered_child_mass_in_pool': float(self.panel.weights[covered].sum()),
                }
            else:
                result[name]['unknown_relations_partitioned'] = False
        return result

    def compare_radials(self, sampled_radial, moved_radial, *, sampled_child_mrr=None, moved_child_mrr=None):
        """Pure scalar analysis also permits independent analytic-distance testing."""
        sampled = finite_array(sampled_radial, (len(self.base),), 'sampled radial')
        moved = finite_array(moved_radial, (len(self.base),), 'candidate radial')
        if (sampled < 0).any() or (moved < 0).any():
            raise ValueError('radial distances must be nonnegative')
        if (sampled_child_mrr is None) != (moved_child_mrr is None):
            raise ValueError('both paired child MRR mappings or neither required')
        if sampled_child_mrr is not None:
            if set(sampled_child_mrr) != set(moved_child_mrr):
                raise ValueError('paired complete retrieval child inventory required')
            if any(not math.isfinite(x) or not 0 <= x <= 1
                   for values in (sampled_child_mrr, moved_child_mrr) for x in values.values()):
                raise ValueError('finite evaluable child MRR required')
        delta = moved - sampled
        self.started = True
        output = {}
        for kind, design in self.designs.items():
            parent, child, owner = (design[k] for k in ('parent', 'child', 'owner'))
            sample_gap, moved_gap = sampled[child] - sampled[parent], moved[child] - moved[parent]
            s, m = self.panel.score(sample_gap), self.panel.score(moved_gap)
            f = self.panel.score(design['reference_gap'])
            fields = {
                'sampled_score': s, 'moved_score': m, 'score_change': m - s,
                'sampled_gap': sample_gap, 'moved_gap': moved_gap, 'gap_change': moved_gap - sample_gap,
                'parent_radial_change': delta[parent], 'child_radial_change': delta[child],
                'sampled_parent_radial': sampled[parent], 'sampled_child_radial': sampled[child],
                'moved_parent_radial': moved[parent], 'moved_child_radial': moved[child],
                'full_parent_radial': self.reference_radial[parent], 'full_child_radial': self.reference_radial[child],
                'full_score': f,
                'correct_to_error': ((s == 1) & (m == 0)).astype(float),
                'error_to_correct': ((s == 0) & (m == 1)).astype(float),
                'sampled_tie': (s == .5).astype(float), 'moved_tie': (m == .5).astype(float),
                'sampled_unresolved': (np.abs(sample_gap) <= design['resolution']).astype(float),
                'moved_unresolved': (np.abs(moved_gap) <= design['resolution']).astype(float),
                'full_unresolved': (np.abs(design['reference_gap']) <= design['resolution']).astype(float),
            }
            for a in (0., .5, 1.):
                for b in (0., .5, 1.):
                    fields[f'transition_{a:g}_to_{b:g}'] = ((s == a) & (m == b)).astype(float)
                    fields[f'F_to_S_{a:g}_to_{b:g}'] = ((f == a) & (s == b)).astype(float)
                    fields[f'F_to_X_{a:g}_to_{b:g}'] = ((f == a) & (m == b)).astype(float)
            if not np.allclose(fields['gap_change'], fields['child_radial_change'] - fields['parent_radial_change'],
                               rtol=0, atol=1e-12):
                raise ValueError('parent/child radial decomposition failed')
            counts = self.panel.design[kind]['covered']
            children = []
            for i, index in enumerate(self.children):
                known = int(counts[i])
                selected = owner == i
                retrieval_known = sampled_child_mrr is not None and int(index) in sampled_child_mrr
                children.append({
                    'child': int(index), 'degree_stratum': str(self.degree[i]), 'h_dev_shortest': self.depth[i],
                    'pool_mean_weight': float(self.panel.weights[i]),
                    'available_relations': int(self.panel.design[kind]['available'][i]),
                    'covered_relations': known,
                    'unknown_relations': int(self.panel.design[kind]['available'][i] - known),
                    'sampled_score': float(s[selected].mean()) if known else None,
                    'moved_score': float(m[selected].mean()) if known else None,
                    'score_change': float((m[selected] - s[selected]).mean()) if known else None,
                    'sampled_mrr': float(sampled_child_mrr[int(index)]) if retrieval_known else None,
                    'moved_mrr': float(moved_child_mrr[int(index)]) if retrieval_known else None,
                    'mrr_change': float(moved_child_mrr[int(index)] - sampled_child_mrr[int(index)]) if retrieval_known else None,
                })
            paired = [row for row in children if row['score_change'] is not None and row['mrr_change'] is not None]
            paired_mass = math.fsum(row['pool_mean_weight'] for row in paired)
            joint = {}
            for score_sign, score_name in ((-1, 'loss'), (0, 'tie'), (1, 'gain')):
                for rank_sign, rank_name in ((-1, 'loss'), (0, 'tie'), (1, 'gain')):
                    selected = [row for row in paired if np.sign(row['score_change']) == score_sign
                                and np.sign(row['mrr_change']) == rank_sign]
                    joint[f'structure_{score_name}_retrieval_{rank_name}'] = (
                        math.fsum(row['pool_mean_weight'] for row in selected) / paired_mass if paired_mass else None)
            output[kind] = {
                'coverage': self.coverage(kind),
                'relations': {k: v.copy() for k, v in design.items() if k != 'groups'},
                'changes': fields, 'groups': self._summaries(kind, fields), 'children': children,
                'joint': {'paired_covered_children': len(paired),
                          'unpaired_sampled_children': len(children) - len(paired),
                          'paired_weight_mass_in_pool': paired_mass, 'fractions': joint},
            }
        return output

    def analyze(self, native_sample, native_candidate, sampled_ranking, moved_ranking, data):
        """Existing saved points/ranks only, no forward/head rerun."""
        if list(data['nodes']) != self.nodes:
            raise ValueError('exact original node order required')
        return self.compare_radials(self.radial(native_sample), self.radial(native_candidate),
                                    sampled_child_mrr=child_reciprocal_ranks(sampled_ranking, data),
                                    moved_child_mrr=child_reciprocal_ranks(moved_ranking, data))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def check_deadline(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError('whole A CPU worker deadline reached')


def bound_json(path, expected_hash):
    raw = Path(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected_hash, 'bound JSON raw hash differs')
    return json.loads(raw)


def validated_headers(roots, bindings, config):
    """All four small records must match before opening any large NPZ."""
    from .recovery_registration import PROTOCOL, CONFIG_SHA256, canonical, source_hashes
    require(len(roots) == len(bindings['shards']) == 4, 'exactly four fixed successful shards required')
    models = {11: {}, 23: {}}
    fingerprints = {}
    rows = []
    inventory = bindings['immutable_file_inventory']
    for root, spec in zip(roots, bindings['shards']):
        row = bound_json(Path(root) / 'run.json', spec['run_raw_sha256'])
        supervisor = bound_json(Path(root) / 'supervisor.json',
                                inventory[spec['remote_output'] + '/supervisor.json']['raw_sha256'])
        progress = bound_json(Path(root) / 'progress.json', inventory[spec['remote_output'] + '/progress.json']['raw_sha256'])
        seed = row['seed']
        require(type(seed) is int and seed in models and seed == spec['seed'], 'fixed original model required')
        require(row['status'] == 'complete' and row['protocol'] == PROTOCOL
                and row['config_sha256'] == CONFIG_SHA256 and row['engineering_fixture_only'] is False
                and row['completed_repetitions'] == 8 and row['weights_unchanged'] is True
                and row['input_unchanged'] is True and row['model_updates'] == 0
                and row['provenance']['release_verified'] is True
                and row['provenance']['source_commit'] == bindings['old_science_source_commit']
                and progress['protocol'] == row['protocol'] and progress['phase'] == 'science'
                and progress['config_sha256'] == row['config_sha256']
                and progress['source_commit'] == row['provenance']['source_commit']
                and progress['source_lf_sha256'] == source_hashes()
                and progress['release'] == row['provenance']['release']
                and supervisor['status'] == 'complete' and supervisor['worker_exit_code'] == 0
                and supervisor['deadline_seconds'] == 1080, 'complete released original scientific shard required')
        require(row['identity'] == spec['frozen_identity'], 'frozen shard identity differs from binding')
        fingerprint = canonical(row['identity'])
        require(seed not in fingerprints or fingerprints[seed] == fingerprint, 'cross-shard reference identity drift')
        fingerprints[seed] = fingerprint
        require(row['repeat_range'] == spec['repeat_range'] and row['repeat_range'] in ([0, 8], [8, 16]),
                'fixed original ranges required')
        expected = list(range(*row['repeat_range']))
        require(spec['global_repeat_ids'] == expected and [x['repeat'] for x in row['observations']] == expected
                and len(row['repeat_archives']) == 8 and len(spec['archives']) == 9,
                'complete ordered original repeats and nine archive bindings required')
        for observation in row['observations']:
            repeat = observation['repeat']
            require(type(repeat) is int and repeat not in models[seed], 'duplicate global repeat')
            models[seed][repeat] = observation
        rows.append(row)
    require(all(set(rows) == set(range(16)) for rows in models.values()), 'both full original repeat inventories required')
    return rows


def bound_archive(root, descriptor, expected, deadline):
    """Bind complete manifest metadata, then reuse the accepted full NPZ reader."""
    from .diagnostic_archive import read_archive
    check_deadline(deadline)
    require(descriptor['manifest'] == Path(expected['manifest_path']).name
            and descriptor['manifest_sha256'] == expected['manifest_raw_sha256']
            and descriptor['array_file'] == Path(expected['NPZ_path']).name
            and descriptor['array_file_sha256'] == expected['NPZ_raw_sha256']
            and descriptor['array_bytes'] == expected['NPZ_bytes']
            and descriptor['arrays'] == expected['arrays_count'], 'archive descriptor differs from input binding')
    record = bound_json(Path(root) / descriptor['manifest'], expected['manifest_raw_sha256'])
    require(record['arrays'] == expected['array_metadata']
            and record['array_file_sha256'] == expected['NPZ_raw_sha256'], 'array metadata differs from binding')
    payload = read_archive(root, descriptor)
    check_deadline(deadline)
    return payload


def entry_design(entry, row, config):
    from .recovery_registration import canonical
    from .frozen_recovery import tensor_identity
    nodes = list(entry['nodes'])
    require(len(nodes) == config['prepared']['nodes_count'] and canonical(nodes) == config['prepared']['node_order_hash'],
            'original full ordered node inventory required')
    data = {'nodes': nodes, 'valid': [tuple(map(int, pair)) for pair in entry['valid']]}
    require(len(data['valid']) == config['valid_query_count']
            and canonical([[nodes[a], nodes[b]] for a, b in data['valid']]) == config['prepared']['valid_queries_hash'],
            'original complete validation query identity required')
    for name, original in (('base', 'p'), ('bias', 'b'), ('floor', 'floor')):
        expected = config['calibration'][str(row['seed'])]['fields'][original]
        require(tensor_identity(torch.from_numpy(entry[name])) == {k: expected[k] for k in ('shape', 'dtype', 'data_sha256')},
                'original calibrated field identity differs')
    for name in ('base', 'bias', 'q'):
        require(tensor_identity(torch.from_numpy(entry[name])) == row['identity'][name], 'fixed p/b/q identity differs')
    require({k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in entry['weights'].items()} == row['identity']['weights'],
            'original frozen weight identity differs')
    validate_ranking(entry['F_ranking'], data)
    require(canonical(entry['F_ranking']['rows']) == row['identity']['F_rank_hash'], 'F ranking identity differs')
    view = SimpleNamespace(nodes=nodes, root=entry['structure_view']['root'], reachable=set(entry['structure_view']['reachable']))
    require(view.root == config['calibration'][str(row['seed'])]['reference_root'], 'fixed original reference root required')
    analyzer = RelationLocalization(view, entry['panel'], torch.from_numpy(entry['base']), torch.from_numpy(entry['bias']),
                                    torch.from_numpy(entry['floor']), config['model']['c'],
                                    native_full=torch.from_numpy(entry['native_F']))
    return analyzer, data


def verify_saved_summary(kind, analyzed, sample_structure, moved_structure, sample_metrics, moved_metrics):
    """Numerically reproduce original primary/coverage/unresolved quantities."""
    s = sample_structure[kind]['groups']['V']
    m = moved_structure[kind]['groups']['V']
    coverage = analyzed['coverage']
    require(coverage == {k: v for k, v in s.items() if k != 'weighted_covered_child_metrics'}
            and coverage == {k: v for k, v in m.items() if k != 'weighted_covered_child_metrics'}, 'original coverage differs')
    metrics = analyzed['groups']['all']['metrics']
    pairs = ((metrics['sampled_score'], sample_metrics[kind]), (metrics['moved_score'], moved_metrics[kind]),
             (metrics['score_change'], moved_metrics[kind] - sample_metrics[kind]),
             (metrics['sampled_unresolved'], s['weighted_covered_child_metrics']['unresolved_fraction']),
             (metrics['moved_unresolved'], m['weighted_covered_child_metrics']['unresolved_fraction']))
    require(all(a is not None and b is not None and math.isfinite(a) and math.isfinite(b) and abs(a - b) <= 1e-12
                for a, b in pairs), 'original primary/unresolved summaries do not reproduce')


def run_analysis(config, bindings, roots, output, *, deadline, provenance):
    """Read one original repeat at a time; output references immutable old bytes."""
    from .diagnostic_archive import write_archive
    from .encoder_matched_control import atomic_json
    from .recovery_registration import validate_config, canonical
    validate_config(config)
    check_deadline(deadline)
    rows = validated_headers(roots, bindings, config)
    target = Path(output)
    result = {'status': 'running', 'protocol': 'E2-radial-components-v1/A', 'scope': 'exploratory descriptive existing-archive localization',
              'engineering_fixture_only': False, 'new_graph_samples': 0, 'model_updates': 0, 'GPU_seconds': 0,
              'old_science_source_commit': bindings['old_science_source_commit'],
              'old_run_raw_sha256': [s['run_raw_sha256'] for s in bindings['shards']],
              'provenance': provenance, 'completed_repetitions': 0, 'repetitions': [], 'fixed_models': {}}
    try:
        for root, row, spec in zip(roots, rows, bindings['shards']):
            entry = bound_archive(root, row['entry_archive'], spec['archives'][0], deadline)
            require(spec['archives'][0]['role'] == 'entry', 'entry binding order required')
            analyzer, data = entry_design(entry, row, config)
            seed = row['seed']
            signature = analyzer.frozen_signature()
            if str(seed) not in result['fixed_models']:
                static = {kind: {k: v for k, v in d.items() if k != 'groups'} for kind, d in analyzer.designs.items()}
                group_masks = {kind: d['groups'] for kind, d in analyzer.designs.items()}
                result['fixed_models'][str(seed)] = {'identity': row['identity'], 'reference_root': entry['structure_view']['root'],
                    'fixed_design_sha256': signature,
                    'group_definitions': analyzer.group_definitions,
                    'design_archive': write_archive(target, f'seed{seed}-design', {'relations': static, 'group_masks': group_masks,
                        'panel': entry['panel'], 'nodes': entry['nodes'], 'reference_radial': analyzer.reference_radial,
                        'bias_norm': analyzer.bias_norm})}
            else:
                require(signature == result['fixed_models'][str(seed)]['fixed_design_sha256'],
                        'F points/reference groups/weights differ across original shards')
            for observation, descriptor, expected in zip(row['observations'], row['repeat_archives'], spec['archives'][1:]):
                check_deadline(deadline)
                payload = bound_archive(root, descriptor, expected, deadline)
                require(expected['role'] == 'repeat' and expected['global_repeat_id'] == observation['repeat']
                        and all(payload[k] == observation[k] for k in ('repeat', 'rng', 'plan_hash')),
                        'repeat/plan/stream identity differs')
                require(set(payload['native_points']) == set(payload['rankings']) == set(payload['structure']) == {'S', 'O', 'C', 'Q'},
                        'original complete four-condition point/rank/structure inventory required')
                s_radial = analyzer.radial(torch.from_numpy(payload['native_points']['S']))
                s_rank = child_reciprocal_ranks(payload['rankings']['S'], data)
                comparisons = {}
                for condition in ('O', 'C', 'Q'):
                    x_radial = analyzer.radial(torch.from_numpy(payload['native_points'][condition]))
                    x_rank = child_reciprocal_ranks(payload['rankings'][condition], data)
                    comparisons[condition + '-S'] = analyzer.compare_radials(s_radial, x_radial,
                        sampled_child_mrr=s_rank, moved_child_mrr=x_rank)
                    for kind, analyzed in comparisons[condition + '-S'].items():
                        verify_saved_summary(kind, analyzed, payload['structure']['S'], payload['structure'][condition],
                                             observation['metrics']['S'], observation['metrics'][condition])
                    check_deadline(deadline)
                summary = {comparison: {kind: {'coverage': value['coverage'], 'groups': value['groups'], 'joint': value['joint']}
                                        for kind, value in by_kind.items()} for comparison, by_kind in comparisons.items()}
                artifact = write_archive(target, f"seed{seed}-repeat{observation['repeat']:02}",
                    {'seed': seed, 'repeat': observation['repeat'], 'comparisons': comparisons,
                     'original_full_validation_metrics': observation['metrics'], 'plan_hash': observation['plan_hash']})
                result['repetitions'].append({'seed': seed, 'repeat': observation['repeat'], 'archive': artifact,
                                             'summary': summary, 'original_full_validation_metrics': observation['metrics']})
                result['completed_repetitions'] += 1
                atomic_json(target / 'progress.json', {'status': 'running', 'seed': seed, 'repeat': observation['repeat'],
                                                       'completed_repetitions': result['completed_repetitions']})
                del payload, comparisons
            del entry, analyzer
        require(result['completed_repetitions'] == 32 and len(result['fixed_models']) == 2, 'all original paired repeats required')
        check_deadline(deadline)
        result['status'] = 'complete'
        result['summary_policy'] = 'per-repeat single-factor descriptive groups; fixed original relation weights; no subgroup significance tests'
        result['old_config_canonical_sha256'] = canonical(config)
        atomic_json(target / 'run.json', result)
        check_deadline(deadline)
        return result
    except BaseException as error:
        result['status'] = 'failed'
        result['failure'] = {'type': type(error).__name__, 'error': str(error), 'partial_results_are_complete': False}
        atomic_json(target / 'run.json', result)
        raise


def worker(args, provenance):
    require(int(os.environ['ACL_RELATION_LOCALIZATION_SUPERVISOR_PID']) == os.getppid(), 'bounded A CPU supervisor required')
    deadline = float(os.environ['ACL_RELATION_LOCALIZATION_DEADLINE'])
    torch.set_num_threads(4)
    config = json.loads(Path(args.config).read_bytes())
    bindings = json.loads(Path(args.bindings).read_bytes())
    from .encoder_matched_control import require_runtime
    require_runtime(config, torch.device('cpu'))
    require(str(np.__version__) == '1.26.4', 'registered original NumPy runtime required')
    provenance = {**provenance, 'torch': str(torch.__version__), 'numpy': str(np.__version__), 'device': 'cpu'}
    return run_analysis(config, bindings, args.shards, args.output, deadline=deadline, provenance=provenance)
