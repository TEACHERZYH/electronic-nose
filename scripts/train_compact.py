"""Fit date-resampled compact selectors and independently tuned prediction heads."""
from _common import ROOT,LABELS,PROBABILITY_COLUMNS,load_inputs,sha,write_json
from pathlib import Path
import argparse,hashlib,json,time,warnings
import numpy as np
import pandas as pd
from joblib import Parallel,delayed,parallel_config
from enose_drift.tcss import GroupSparseCVaRMultinomial,_cluster_resample_indices,_rank
from enose_drift.compact_baselines import tune_and_fit_probabilistic_model
from enose_drift.calibration import TemperatureScaler
from enose_drift.preprocessing import select_group_columns
def fit_one(index,alpha,arrays,output,signature):
    p=output/'replicates'/f'{index:03d}.npz'
    if p.exists():
        with np.load(p,allow_pickle=False) as f:assert f['signature'].item()==signature
        return
    groups={int(g):c for g,c in zip(arrays['group_ids'],arrays['group_cols'])};seed=int(arrays['replicate_seeds'][index])
    rows,env=_cluster_resample_indices(arrays['train_dates'],arrays['environments'],np.random.default_rng(seed));started=time.monotonic()
    model=GroupSparseCVaRMultinomial(groups,alpha=alpha,group_lambda=.01,max_iter=500,tol=1e-6,stable_iterations=5,initial_step_size=1,backtracking_factor=.5).fit(arrays['X_rank'][rows],arrays['y_rank'][rows],env,classes=np.unique(arrays['y_rank']))
    magnitude=np.asarray([model.group_magnitudes()[g] for g in groups]);inclusion=magnitude>1e-6
    assert np.isfinite(model.coef_).all() and np.all(np.diff(model.objective_history_)<=1e-9)
    tmp=p.with_suffix('.tmp.npz');np.savez_compressed(tmp,signature=np.asarray(signature),seed=np.asarray(seed,dtype=np.uint32),alpha=np.asarray(alpha),inclusion=inclusion,magnitude=magnitude,ranking=np.asarray(_rank(tuple(groups),inclusion.astype(float),magnitude),dtype=int),converged=np.asarray(model.converged_),iterations=np.asarray(model.n_iter_),seconds=np.asarray(time.monotonic()-started),objective=np.asarray(model.objective_history_),environment_losses=np.asarray(model.environment_losses_),coef=model.coef_,intercept=model.intercept_);tmp.replace(p)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inputs',type=Path,default=ROOT/'data/processed');p.add_argument('--output',type=Path,required=True);p.add_argument('--alpha',type=float,choices=[0.,.75],default=.75);p.add_argument('--n-jobs',type=int,default=1);p.add_argument('--check-only',action='store_true');args=p.parse_args()
    if args.n_jobs<1:raise ValueError('n-jobs must be positive.')
    a,m=load_inputs(args.inputs);settings=json.loads((ROOT/'configs/alpha_control.json').read_text(encoding='utf8'))
    identity={'input_sha256':sha(args.inputs/'primary_inputs.npz'),'manifest_sha256':sha(args.inputs/'sample_manifest.csv'),'script_sha256':sha(__file__),'source_sha256':{f.name:sha(f) for f in (ROOT/'src/enose_drift').glob('*.py')},'alpha':args.alpha,'settings':settings}
    signature=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest();meta=args.output/'settings.json'
    if args.check_only:print('Inputs, splits, resampling seeds and configuration verified.');return
    if args.output.exists():
        if not meta.exists() or json.loads(meta.read_text(encoding='utf8'))['signature']!=signature:raise FileExistsError('Output belongs to a different workload.')
        if (args.output/'predictions.parquet').exists():raise FileExistsError('Completed output exists; no fitting repeated.')
    args.output.mkdir(parents=True,exist_ok=True);(args.output/'replicates').mkdir(exist_ok=True);write_json(meta,{'signature':signature,**identity})
    with parallel_config(backend='loky',inner_max_num_threads=1):
        for start in range(0,200,8):
            Parallel(n_jobs=args.n_jobs)(delayed(fit_one)(i,args.alpha,a,args.output,signature) for i in range(start,min(start+8,200)))
            print('Completed selectors:',min(start+8,200),'/200',flush=True)
    inclusion=[];magnitude=[]
    for i in range(200):
        with np.load(args.output/'replicates'/f'{i:03d}.npz',allow_pickle=False) as f:inclusion.append(f['inclusion']);magnitude.append(f['magnitude'])
    groups={int(g):c for g,c in zip(a['group_ids'],a['group_cols'])};frequency=np.mean(inclusion,axis=0);magnitudes=np.mean(magnitude,axis=0);ranking=_rank(tuple(groups),frequency,magnitudes);selected=ranking[:16]
    pd.DataFrame([{'rank':j+1,'sensor':int(g),'frequency':float(frequency[list(groups).index(g)]),'mean_magnitude':float(magnitudes[list(groups).index(g)])} for j,g in enumerate(ranking)]).to_csv(args.output/'sensor_ranking.csv',index=False)
    columns,_=select_group_columns(groups,selected);columns=np.r_[columns,a['always']];X=a['X_predict'];y=m.label.to_numpy();tr=m.split.eq('train').to_numpy();val=m.split.eq('validation').to_numpy();te=m.split.eq('test').to_numpy();predictions=[];tuning=[];heads=[]
    for seed in settings['head_seeds']:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');model,records=tune_and_fit_probabilistic_model('multinomial',X[np.ix_(tr,columns)],y[tr],X[np.ix_(val,columns)],y[val],seed=seed,candidate_C=(.1,1,10));temperature=TemperatureScaler().fit(model.decision_function(X[np.ix_(val,columns)]),y[val],model.classes_)
        assert model.classes_.tolist()==LABELS
        heads.append({'seed':seed,'selected_C':model.selected_C,'temperature':temperature.temperature_,'warnings':[str(w.message) for w in caught]})
        tuning.extend({'seed':seed,**r} for r in records)
        for split,take in [('validation',val),('test',te)]:
            probability=temperature.predict_proba(model.decision_function(X[np.ix_(take,columns)])) if split=='test' else model.predict_proba(X[np.ix_(take,columns)])
            for i,row in enumerate(m.loc[take].itertuples(index=False)):
                predictions.append({'method':f'alpha_{args.alpha:g}','split':split,'seed':seed,'sample_id':row.sample_id,'date_id':row.date,'y_true':row.label,'y_pred':LABELS[int(probability[i].argmax())],'selected_C':model.selected_C,**dict(zip(PROBABILITY_COLUMNS,probability[i]))})
    pd.DataFrame(predictions).to_parquet(args.output/'predictions.parquet',index=False);pd.DataFrame(tuning).to_csv(args.output/'tuning.csv',index=False);write_json(args.output/'heads.json',heads);print('Completed:',args.output)
if __name__=='__main__':main()
