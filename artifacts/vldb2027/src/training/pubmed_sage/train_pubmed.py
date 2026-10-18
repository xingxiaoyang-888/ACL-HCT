"""Fixed 500-epoch mean GraphSAGE on PubMed; no dev/test scoring or correction.

Only train and fit labels are bound/opened. The unchanged accepted SAGE and
native Uniform planner are reused with new round RNG and 500 input/3 output.
"""
import json
import time
import numpy as np
from common import torch, release, read_json, sha, save, configure, domain
from blocks import Graph, plan
from sage import SAGE
from torch.nn import functional as F


def labels(path, expected):
    with np.load(path,allow_pickle=False) as z:
        if set(z.files)!={'ids','y'} or z['ids'].tolist()!=expected or z['y'].dtype != np.int64:
            raise ValueError('released label schema/order differs')
        values=z['y'].copy()
    if values.shape!=(len(expected),) or np.any(values<0) or np.any(values>=3):
        raise ValueError('label shape/range differs')
    return dict(zip(expected,map(int,values)))


def run_job(release_path, expected_release_sha256, guard_callback):
    if not callable(guard_callback):
        raise ValueError('real Ops guard callback required')
    guard_callback()
    r,inputs,cfg,out=release(release_path,expected_release_sha256,'train_pubmed_sage')
    if set(inputs)!={'graph','ids','preparation','train','fit'} or r['seed'] not in (11,23,37):
        raise ValueError('only train and fit inputs and registered seed allowed')
    if cfg['schema']!='vldb_pubmed_sage_config_v1' or cfg['training']!={'epochs':500,'fanouts':[4,4,4],'batch_size':1024,'learning_rate':0.01,'weight_decay':0.0,'selection_every':5,'selection_repeats':4,'selection_metric':'sum_correct_then_earliest_tie'}:
        raise ValueError('frozen training configuration differs')
    if cfg['architecture'] != {'input_features':500,'hidden':256,'output_classes':3,'layers':3,'dropout':0.5,'hidden_processing':'BatchNorm-ReLU-dropout','root_transform':'independent_native_lin_r'}:
        raise ValueError('frozen architecture differs')
    configure();seed=r['seed']
    ids=read_json(inputs['ids']);receipt=read_json(inputs['preparation'])
    if ids['dataset']!='PubMed' or (ids['nodes'],ids['features'],ids['classes'])!=(19717,500,3):
        raise ValueError('prepared data metadata differs')
    if any(receipt['public'][name+'.'+ext]!=sha(inputs[name]) for name,ext in [('graph','npz'),('ids','json')]):
        raise ValueError('public preparation binding mismatch')
    if any(receipt['development'][name+'.npz']!=sha(inputs[name]) for name in ('train','fit')):
        raise ValueError('development preparation binding mismatch')
    if [len(ids['ids'][s]) for s in ('train','fit','dev','test')] != [60,500,500,1000]:
        raise ValueError('fixed split sizes differ')
    if ids['split_hash_prefix'] != cfg['split']['hash_prefix'] or receipt['config_sha256'] != r['config']['sha256']:
        raise ValueError('preparation config/split binding differs')
    with np.load(inputs['graph'],allow_pickle=False) as z:
        if set(z.files)!={'x','indptr','indices','values','diagonal'}:
            raise ValueError('public graph schema differs')
        g=Graph(torch.from_numpy(z['x'].copy()),z['indptr'].copy(),z['indices'].copy(),z['values'].copy(),z['diagonal'].copy(),'sage')
    train_y,fit_y=labels(inputs['train'],ids['ids']['train']),labels(inputs['fit'],ids['ids']['fit'])
    device='cuda:0'
    if not torch.cuda.is_available():
        raise ValueError('released training requires assigned CUDA device')
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model=SAGE(inputs=500,hidden=256,outputs=3,dropout=.5).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=.01,weight_decay=0.)
    out.mkdir(parents=True,exist_ok=False)
    binding={'release_sha256':expected_release_sha256,'source_manifest_sha256':r['source_manifest']['sha256'],
             'config_sha256':r['config']['sha256'],'inputs':{k:sha(v) for k,v in inputs.items()}}
    best_sum,best_epoch,updates=-1,None,0;start=time.perf_counter()
    with (out/'curve.jsonl').open('x',encoding='utf-8') as log:
        for epoch in range(1,501):
            model.train();shuffled=np.random.default_rng(domain('train-shuffle',seed,epoch)).permutation(ids['ids']['train'])
            loss_total,count=0.,0
            for batch,offset in enumerate(range(0,len(shuffled),1024)):
                targets=shuffled[offset:offset+1024]
                blocks=plan(g,targets,[4]*3,'train-sampling',seed,epoch,batch)
                x=g.x[torch.from_numpy(blocks[0].src)].to(device)
                y=torch.tensor([train_y[int(n)] for n in targets],device=device)
                optimizer.zero_grad(set_to_none=True)
                output=model(x,blocks);loss=F.nll_loss(output,y)
                if not bool(torch.isfinite(loss)):raise ValueError('nonfinite training loss; no retry')
                loss.backward()
                if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):raise ValueError('nonfinite gradient; no retry')
                optimizer.step();loss_total+=float(loss.detach())*len(targets);count+=len(targets);updates+=1
            row={'epoch':epoch,'updates':updates,'train_nodes':count,'train_nll':loss_total/count}
            if epoch%5==0:
                guard_callback();model.eval();correct=[]
                with torch.no_grad():
                    for repeat in range(4):
                        ncorrect=0
                        for batch,target in enumerate(ids['batches']['fit']):
                            target=np.asarray(target,dtype=np.int64)
                            blocks=plan(g,target,[4]*3,'cp-selection',seed,repeat,batch)
                            pred=model(g.x[torch.from_numpy(blocks[0].src)].to(device),blocks)
                            if not bool(torch.isfinite(pred).all()):raise ValueError('nonfinite fit outputs')
                            ncorrect+=int((pred.argmax(-1).cpu()==torch.tensor([fit_y[int(n)] for n in target])).sum())
                        correct.append(ncorrect)
                selected=sum(correct)>best_sum
                row.update(fit_correct=correct,fit_count=500,selected=selected)
                if selected:
                    best_sum,best_epoch=sum(correct),epoch
                    payload={'schema':'vldb_pubmed_sage_weights_v1','seed':seed,'epoch':epoch,
                        'model_state':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},'bindings':binding}
                    tmp=out/'selected.writing.pt';torch.save(payload,tmp);tmp.replace(out/'label-free-weights.pt')
                print(f'epoch={epoch} updates={updates} selected_epoch={best_epoch}',flush=True)
            log.write(json.dumps(row,allow_nan=False)+'\n');log.flush()
    guard_callback()
    result={'schema':'vldb_pubmed_sage_training_complete_v1','seed':seed,'epochs_executed':500,'updates':updates,
        'selected_epoch':best_epoch,'selected_correct_sum':best_sum,'selection_denominator':2000,
        'wall_seconds':time.perf_counter()-start,'weights_sha256':sha(out/'label-free-weights.pt'),
        'bindings':binding,'dev_labels_loaded':False,'test_labels_loaded':False,'test_scored':False}
    save(out/'completion.json',result)
    return result
