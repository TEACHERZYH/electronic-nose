from pathlib import Path
import os,sys,json,hashlib
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ.setdefault(k,'1')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
LABELS=['2-Phenylethanol 1000 ppm','2-Phenylethanol 200 ppm','Diacetyl 0.1 ppm','Diacetyl 1 ppm','EtOH']
PROBABILITY_COLUMNS=['p_'+x for x in LABELS]
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def write_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x))+'\n',encoding='utf8')
def load_inputs(folder=None):
    folder=Path(folder or ROOT/'data/processed')
    with np.load(folder/'primary_inputs.npz',allow_pickle=False) as f:arrays={k:f[k] for k in f.files}
    manifest=pd.read_csv(folder/'sample_manifest.csv')
    assert len(manifest)==len(arrays['X_predict'])==584
    assert manifest.sample_id.is_unique and arrays['X_rank'].shape==(360,440)
    assert manifest.groupby('split').size().to_dict()=={'test':119,'train':360,'validation':105}
    assert np.array_equal(manifest.loc[manifest.split.eq('train'),'label'].to_numpy(),arrays['y_rank'])
    assert np.array_equal(manifest.loc[manifest.split.eq('train'),'date'].to_numpy(),arrays['train_dates'])
    assert np.isfinite(arrays['X_predict']).all() and np.isfinite(arrays['X_rank']).all()
    assert manifest.groupby('date').split.nunique().max()==1
    return arrays,manifest
def bootstrap(delta,seed=20260826,replicates=10000):
    delta=np.asarray(delta,float);rng=np.random.default_rng(seed)
    draws=delta[rng.integers(0,len(delta),size=(replicates,len(delta)))].mean(1)
    lo,hi=np.quantile(draws,[.025,.975])
    return {'difference':float(delta.mean()),'ci_low':float(lo),'ci_high':float(hi),'n_dates':len(delta),'replicates':replicates,'seed':seed}
