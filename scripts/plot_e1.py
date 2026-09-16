"""Rebuild E1 static figures using only the recorded result JSON."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(result_path, output_dir):
    document=json.loads(Path(result_path).read_text(encoding='utf-8'))
    rows=[r for r in document['cases'] if r['result']['status'] in ('ok','partial_method_failure')]
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':9,'savefig.dpi':160})
    groups={}
    for row in rows:
        case=row['case']; key=(case['family'],case['spread'],case['c'],case['d'],case['origin_shift'])
        groups.setdefault(key,[]).append(row)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for key,group in groups.items():
        group=sorted(group,key=lambda r:r['case']['k']); label=f'{key[0]} s={key[1]} c={key[2]} d={key[3]}'
        k=[r['case']['k'] for r in group]
        projection=[r['result']['methods']['none']['predicted_direction_projection'] for r in group]
        se=[r['result']['methods']['none']['projection_mc_se'] for r in group]
        axes[0].errorbar(k,projection,yerr=np.array(se)*1.96,marker='.',label=label)
        axes[1].plot(k,[r['result']['methods']['none']['prediction_cosine'] for r in group],marker='.')
    axes[0].set(xlabel='k',ylabel='Signed projection onto oracle direction',title='Common full-mean tangent base; bars = 1.96 MC SE')
    axes[1].set(xlabel='k',ylabel='Cosine with predicted direction',ylim=(-1.05,1.05),title='Undefined near zero; gaps retained')
    axes[0].legend(fontsize=6); fig.tight_layout(); fig.savefig(output_dir/'direction.png'); plt.close(fig)
    names=('none','third_unclipped','third_protected','jackknife_protected')
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for name in names:
        selected=[r for r in rows if r['result']['methods'][name]['status']=='ok']
        for axis,metric in zip(axes,('mean_offset_norm','variance_population_moment','mse')):
            axis.plot([rows.index(r) for r in selected],[r['result']['methods'][name][metric] for r in selected],'.-',label=name)
            axis.set(xlabel='Case index in JSON',ylabel=metric)
            axis.set_yscale('symlog',linthresh=1e-10)
    axes[0].set_title('MC mean norm is noisy, not exact bias')
    axes[2].legend(fontsize=7); fig.tight_layout(); fig.savefig(output_dir/'bias_variance_mse.png'); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    for family in ('symmetric','asymmetric','reverse'):
        selected=[r for r in rows if r['case']['family']==family]
        ax.scatter([r['case']['spread'] for r in selected],
                   [r['result']['methods']['none']['prediction_error_norm'] for r in selected],label=family)
    ax.set(xlabel='Tangent construction scale',ylabel='Norm of MC/exact mean minus second-order oracle',
           title='Prediction discrepancy includes MC noise; see JSON covariance')
    ax.set_yscale('symlog',linthresh=1e-10); ax.legend(); fig.tight_layout()
    fig.savefig(output_dir/'scale_prediction.png'); plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True); args=parser.parse_args()
    plot(args.input,args.output_dir)
