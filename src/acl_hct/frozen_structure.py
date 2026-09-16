"""Positive H_dev radial relations and pre-confirmation registration guard."""
import math
import torch
from .geometry import distance


def require_confirmation_registration(registration):
    """Presence/units guard only, not scientific approval of arbitrary thresholds."""
    keys=('full_reference_ability','effect_loss_threshold','independent_support_metric',
          'matched_noise_difference','mc_precision')
    if not isinstance(registration,dict) or any(key not in registration for key in keys):
        raise ValueError('confirmation requires all five preregistered evidence criteria')
    if (registration.get('frozen_before_confirmation') is not True
            or registration.get('test_labels_unopened') is not True):
        raise ValueError('registration must precede confirmation sampling and test opening')
    for key in keys:
        row=registration[key]
        if (not isinstance(row,dict) or not isinstance(row.get('metric'),str) or not row['metric'].strip()
                or not isinstance(row.get('unit'),str) or not row['unit'].strip()
                or type(row.get('threshold')) not in (int,float) or not math.isfinite(row['threshold'])
                or row['threshold']<0):
            raise ValueError(f'criterion requires metric, unit and finite nonnegative threshold: {key}')
    return {'status':'registration_fields_present','scientific_approval_inferred':False}


@torch.no_grad()
def radial_relations(view,panel,points,reference,groups,c=1.,numerical_floor=None):
    """Child-macro design-weighted ratios among covered positive relations.

    Points/reference are uniformly promoted diagnostic FP64 points. The anchor
    is always the full reference root. Unknown endpoints remain in coverage;
    no missing paths or disjoint truncated branches are converted to negatives.
    """
    if points.shape!=reference.shape or points.dtype!=torch.float64 or reference.dtype!=torch.float64:
        raise ValueError('matching diagnostic FP64 points required')
    if len(points)!=len(view.nodes):raise ValueError('complete node state required')
    index={node:i for i,node in enumerate(view.nodes)}
    root=index.get(view.root)
    radial=distance(reference[root],points,c) if root is not None else None
    full_radial=distance(reference[root],reference,c) if root is not None else None
    panel_rows={row['id']:row for row in panel['rows']}
    if numerical_floor is not None:
        if numerical_floor.shape!=(len(points),) or (numerical_floor<0).any() or not torch.isfinite(numerical_floor).all():
            raise ValueError('invalid radial numerical floor')
    result={}
    for kind,key in (('direct','direct_parents'),('distant','positive_distant_ancestors')):
        rows=[]
        for relation in panel['relations']:
            child=relation['child'];item=panel_rows[child];b=index[child];pairs=[]
            for parent in relation[key]:
                a=index[parent]
                if root is None or child not in view.reachable or parent not in view.reachable:continue
                gap=float(radial[b]-radial[a]);full_gap=float(full_radial[b]-full_radial[a])
                score=1. if gap>0 else .5 if gap==0 else 0.
                full_score=1. if full_gap>0 else .5 if full_gap==0 else 0.
                resolution=float(numerical_floor[a]+numerical_floor[b]+2*numerical_floor[root]) if numerical_floor is not None else None
                pairs.append({'parent':parent,'score':score,'full_score':full_score,
                              'tie':float(gap==0),'correct_to_error':float(full_score==1 and score==0),
                              'error_to_correct':float(full_score==0 and score==1),
                              'radial_gap':gap,'full_radial_gap':full_gap,
                              'unresolved_at_numerical_floor':abs(gap)<=resolution if resolution is not None else None})
            rows.append({'child':child,'index':b,'weight':item['pool_mean_weight'],
                         'available_relations':len(relation[key]),'covered_relations':len(pairs),
                         'unknown_relations':len(relation[key])-len(pairs),'pairs':pairs})
        summaries={}
        for name,ids in groups.items():
            members=set(ids);selected=[row for row in rows if row['index'] in members]
            covered=[row for row in selected if row['pairs']]
            mass=sum(row['weight'] for row in covered)
            metrics={}
            for metric in ('score','full_score','tie','correct_to_error','error_to_correct'):
                numerator=sum(row['weight']*sum(pair[metric] for pair in row['pairs'])/len(row['pairs']) for row in covered)
                metrics[metric]=numerator/mass if mass else None
            metrics['score_change']=metrics['score']-metrics['full_score'] if mass else None
            summaries[name]={'sampled_children':len(selected),'covered_children':len(covered),
                             'no_covered_relation_children':len(selected)-len(covered),
                             'unknown_relations':sum(row['unknown_relations'] for row in selected),
                             'weighted_covered_child_mass_in_pool':mass,
                             'weighted_sampled_child_mass_in_pool':sum(row['weight'] for row in selected),
                             'weighted_covered_child_metrics':metrics}
        result[kind]={'children':rows,'groups':summaries}
    return {'reference_root':view.root,'scope':'H_dev positive relations; weighted covered-child ratio within valid diagnostic pool, not V mean',
            'unknown_policy':'unreachable endpoints and children without positive pairs retained in coverage',
            'score_policy':'strict radial order 1, exact tie .5, reverse 0; full anchor fixed',
            'metrics':result}
