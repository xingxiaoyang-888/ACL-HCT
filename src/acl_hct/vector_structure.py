"""Cached positive-relation design; one radial GPU-to-CPU transfer per condition."""
import numpy as np
import torch
from .geometry import distance


METRICS=('score','full_score','tie','correct_to_error','error_to_correct','score_change','unresolved_fraction')


class RadialPanel:
    def __init__(self,view,panel,reference,groups,c=1.,numerical_floor=None):
        if reference.dtype!=torch.float64 or len(reference)!=len(view.nodes):raise ValueError('complete FP64 reference required')
        self.reference=reference;self.c=c;self.root=view.nodes.index(view.root) if view.root else None
        self.design={};self.groups={name:set(ids) for name,ids in groups.items()}
        index={node:i for i,node in enumerate(view.nodes)};items={row['id']:row for row in panel['rows']}
        self.children=[row['child'] for row in panel['relations']]
        self.child_indices=np.array([index[node] for node in self.children],dtype=np.int64)
        self.weights=np.array([items[node]['pool_mean_weight'] for node in self.children])
        self.group_masks={name:np.array([int(i) in members for i in self.child_indices],dtype=bool)
                          for name,members in self.groups.items()}
        floor=None if numerical_floor is None else numerical_floor.detach().cpu().numpy()
        if floor is not None and (floor.shape!=(len(reference),) or not np.isfinite(floor).all() or (floor<0).any()):
            raise ValueError('invalid numerical floor')
        full=distance(reference[self.root],reference,c).detach().cpu().numpy() if self.root is not None else None
        for kind,key in [('direct','direct_parents'),('distant','positive_distant_ancestors')]:
            parent=[];child=[];owner=[];available=[]
            for i,row in enumerate(panel['relations']):
                available.append(len(row[key]))
                for node in row[key]:
                    if self.root is not None and node in view.reachable and row['child'] in view.reachable:
                        parent.append(index[node]);child.append(index[row['child']]);owner.append(i)
            parent=np.asarray(parent,dtype=np.int64);child=np.asarray(child,dtype=np.int64);owner=np.asarray(owner,dtype=np.int64)
            counts=np.bincount(owner,minlength=len(self.children))
            full_gap=full[child]-full[parent] if full is not None else np.empty(0)
            self.design[kind]={'parent':parent,'child':child,'owner':owner,'covered':counts,
                               'available':np.asarray(available),'full_score':self.score(full_gap),
                               'resolution':floor[parent]+floor[child]+2*floor[self.root] if floor is not None and self.root is not None else None}

    @staticmethod
    def score(gap):return np.where(gap>0,1.,np.where(gap==0,.5,0.))

    @torch.no_grad()
    def evaluate(self,points):
        if points.shape!=self.reference.shape or points.dtype!=torch.float64:raise ValueError('matching complete FP64 points required')
        radial=distance(self.reference[self.root],points,self.c).detach().cpu().numpy() if self.root is not None else None
        output={}
        for kind,design in self.design.items():
            owner=design['owner'];counts=design['covered'];full=design['full_score']
            gap=radial[design['child']]-radial[design['parent']] if radial is not None else np.empty(0)
            score=self.score(gap)
            pair_values={'score':score,'full_score':full,'tie':(gap==0).astype(float),
                         'correct_to_error':((full==1)&(score==0)).astype(float),
                         'error_to_correct':((full==0)&(score==1)).astype(float),'score_change':score-full,
                         'unresolved_fraction':(np.abs(gap)<=design['resolution']).astype(float) if design['resolution'] is not None else None}
            child_values={key:np.bincount(owner,weights=value,minlength=len(counts))/np.maximum(counts,1)
                          for key,value in pair_values.items() if value is not None}
            summaries={}
            for name,selected in self.group_masks.items():
                covered=selected&(counts>0);mass=float(self.weights[covered].sum())
                metrics={key:float(np.sum(self.weights[covered]*child_values[key][covered])/mass)
                         if mass and key in child_values else None for key in METRICS}
                summaries[name]={'sampled_children':int(selected.sum()),'covered_children':int(covered.sum()),
                                 'no_covered_relation_children':int(selected.sum()-covered.sum()),
                                 'unknown_relations':int((design['available']-counts)[selected].sum()),
                                 'weighted_covered_child_mass_in_pool':mass,
                                 'weighted_sampled_child_mass_in_pool':float(self.weights[selected].sum()),
                                 'weighted_covered_child_metrics':metrics}
            output[kind]={'groups':summaries}
        return output
