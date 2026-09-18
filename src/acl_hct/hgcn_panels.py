"""Fixed full-root hierarchy and nodewise tangent readouts for native HGCN."""
import numpy as np

from .hgcn_geometry import distance_ball_fp64, log_ball_orthonormal_fp64


class HierarchyPanel:
    def __init__(self, view, panel, full_points, c=1., gap_floor=1e-10):
        full = np.asarray(full_points)
        if (full.ndim != 2 or len(full) != len(view.nodes) or full.dtype not in (np.float32, np.float64)
                or not np.isfinite(full).all() or type(gap_floor) not in (int, float)
                or not np.isfinite(gap_floor) or gap_floor <= 0):
            raise ValueError('complete native full points and positive fixed gap floor required')
        # Validate every native point; no radius restriction and no repair.
        distance_ball_fp64(full, full, c)
        self.full = full.astype(np.float64, copy=True); self.full.setflags(write=False)
        self.c = c; self.gap_floor = float(gap_floor)
        self.root = view.nodes.index(view.root) if view.root is not None else None
        self.anchor = None if self.root is None else self.full[self.root].copy()
        if self.anchor is not None:
            self.anchor.setflags(write=False)
        index = {node: i for i, node in enumerate(view.nodes)}
        rows = {row['id']: row for row in panel['rows']}
        if len(rows) != len(panel['rows']) or len(panel['relations']) != len(rows):
            raise ValueError('one distinct row/relation record per selected child required')
        children = [row['child'] for row in panel['relations']]
        if set(children) != set(rows) or len(set(children)) != len(children):
            raise ValueError('selected child coverage mismatch')
        if any(row['index'] != index[node] for node, row in rows.items()):
            raise ValueError('panel node indices differ from fixed node order')
        self.children = np.array([index[n] for n in children], dtype=np.int64)
        self.weights = np.array([rows[n]['pool_mean_weight'] for n in children], dtype=np.float64)
        if np.any(self.weights <= 0) or not np.isfinite(self.weights).all():
            raise ValueError('positive fixed child weights required')
        for node, row in rows.items():
            pi = row['inclusion_probability']
            if not 0 < pi <= 1 or row['pool_mean_weight'] != 1 / (panel['pool_size'] * pi):
                raise ValueError('panel inverse-inclusion weight mismatch')
        self.design = {}
        full_radius = None if self.anchor is None else distance_ball_fp64(self.anchor, self.full, c)
        for kind, key in (('direct', 'direct_parents'), ('distant', 'positive_distant_ancestors')):
            parents, child_ids, owners, available = [], [], [], []
            for owner, record in enumerate(panel['relations']):
                child = record['child']; candidates = record[key]
                if len(set(candidates)) != len(candidates) or any(p not in index or p == child for p in candidates):
                    raise ValueError('distinct valid positive parents required')
                available.append(len(candidates))
                for parent in candidates:
                    if self.anchor is not None and parent in view.reachable and child in view.reachable:
                        parents.append(index[parent]); child_ids.append(index[child]); owners.append(owner)
            parent = np.array(parents, dtype=np.int64); child = np.array(child_ids, dtype=np.int64)
            owner = np.array(owners, dtype=np.int64)
            counts = np.bincount(owner, minlength=len(children)); available = np.array(available, dtype=np.int64)
            gap = full_radius[child] - full_radius[parent] if full_radius is not None else np.empty(0, dtype=np.float64)
            self.design[kind] = {'parent': parent, 'child': child, 'owner': owner, 'counts': counts,
                                 'available': available, 'full_gap': gap, 'full_score': self._score(gap)}
        self.root_known = np.array([n in view.reachable for n in children]) if self.anchor is not None else np.zeros(len(children), bool)
        self.full_child_radius = None if full_radius is None else full_radius[self.children]
        self.direction_known = self.root_known & (self.full_child_radius > self.gap_floor) if full_radius is not None else self.root_known
        self.outward = np.zeros((len(children), full.shape[1]), dtype=np.float64)
        if np.any(self.direction_known):
            chosen = self.direction_known
            self.outward[chosen] = -log_ball_orthonormal_fp64(self.full[self.children[chosen]], self.anchor, c) / self.full_child_radius[chosen, None]

    @staticmethod
    def _score(gap):
        return np.where(gap > 0, 1., np.where(gap == 0, .5, 0.))

    def _weighted(self, values, known):
        mass = float(self.weights[known].sum())
        return float(np.sum(self.weights[known] * values[known]) / mass) if mass else None

    def evaluate(self, points):
        points = np.asarray(points)
        if points.shape != self.full.shape or points.dtype not in (np.float32, np.float64):
            raise ValueError('matching complete native point matrix required')
        distance_ball_fp64(points, points, self.c)
        radius = None if self.anchor is None else distance_ball_fp64(self.anchor, points, self.c)
        output = {}
        for kind, design in self.design.items():
            gap = radius[design['child']] - radius[design['parent']] if radius is not None else np.empty(0, dtype=np.float64)
            score = self._score(gap); counts = design['counts']; covered = counts > 0
            pair = {'score': score, 'gap': gap, 'gap_change': gap - design['full_gap'],
                    'correct_to_error': ((design['full_score'] == 1) & (score == 0)).astype(np.float64),
                    'error_to_correct': ((design['full_score'] == 0) & (score == 1)).astype(np.float64),
                    'tie': (gap == 0).astype(np.float64), 'unresolved': (np.abs(gap) <= self.gap_floor).astype(np.float64)}
            child_values = {name: np.bincount(design['owner'], weights=value, minlength=len(self.children)) / np.maximum(counts, 1)
                            for name, value in pair.items()}
            output[kind] = {'metrics': {name: self._weighted(value, covered) for name, value in child_values.items()},
                            'covered_children': int(covered.sum()), 'selected_children': len(self.children),
                            'covered_pairs': len(gap), 'unknown_pairs': int(np.sum(design['available'] - counts)),
                            'weighted_covered_child_mass': float(self.weights[covered].sum()),
                            'weighted_selected_child_mass': float(self.weights.sum()),
                            'gap': gap, 'score': score, 'child_values': child_values}
        errors = log_ball_orthonormal_fp64(self.full[self.children], points[self.children], self.c)
        radial = np.sum(errors * self.outward, axis=-1)
        output['bias'] = {'nodewise_error_fp64': errors, 'signed_outward_projection_per_node': radial,
                          'weighted_outward_projection': self._weighted(radial, self.direction_known),
                          'weighted_MSE': self._weighted(np.sum(errors * errors, axis=-1), np.ones(len(self.children), bool)),
                          'root_known_nodes': int(self.root_known.sum()), 'direction_known_nodes': int(self.direction_known.sum()),
                          'weighted_direction_known_mass': float(self.weights[self.direction_known].sum()),
                          'weighted_radial_distance_change': self._weighted(radius[self.children] - self.full_child_radius, self.root_known)
                          if radius is not None else None}
        return output

    def archive_design(self):
        return {'root_index': self.root, 'fixed_full_root_point_fp64': self.anchor,
                'selected_children': self.children, 'fixed_weights': self.weights, 'relations': self.design,
                'root_known': self.root_known, 'direction_known': self.direction_known,
                'outward_unit_at_each_full_node': self.outward, 'gap_floor': self.gap_floor,
                'main_score': 'exact gap sign; exact ties half; unresolved is secondary only'}
