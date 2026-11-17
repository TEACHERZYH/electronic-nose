"""Recompute all distributed prediction metrics without fitting any model."""
from _common import ROOT,LABELS,PROBABILITY_COLUMNS,write_json,bootstrap
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score,recall_score,matthews_corrcoef
from enose_drift.v211_classwise_gate import apply_classwise_gate
def metrics(g):
    y=g.y_true.astype(str).to_numpy();pred=g.y_pred.astype(str).to_numpy();p=g[PROBABILITY_COLUMNS].to_numpy(float)
    assert np.isfinite(p).all() and p.min()>=-1e-14 and np.allclose(p.sum(1),1,atol=1e-10)
    assert set(y).issubset(LABELS) and set(pred).issubset(LABELS)
    idx=np.array([LABELS.index(x) for x in y]);truth=np.eye(5)[idx]
    mcc=0.0 if np.unique(np.r_[y,pred]).size==1 else matthews_corrcoef(y,pred)
    return {'n_samples':len(g),'macro_f1':f1_score(y,pred,labels=LABELS,average='macro',zero_division=0),'balanced_accuracy':recall_score(y,pred,labels=LABELS,average='macro',zero_division=0),'mcc':mcc,'nll':float(-np.log(np.clip(p[np.arange(len(p)),idx],1e-15,1)).mean()),'brier':float(np.square(p-truth).sum(1).mean())}
def date_values(f):
    return pd.Series({date:metrics(g)['macro_f1'] for date,g in f.groupby(['seed','date_id'])}).unstack(0).mean(1).sort_index()
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,default=ROOT/'artifacts/recomputed');args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    paths=sorted((ROOT/'data/predictions').glob('*.parquet'))+sorted((ROOT/'data/alpha_control').glob('*_predictions.parquet'))
    frames={};records=[];counts={}
    for path in paths:
        f=pd.read_parquet(path);f=f[f.split.eq('test')].copy();frames[path.stem]=f;counts[path.stem]=len(f)
        conditions=[x for x in ['method','sensor_budget','blank_budget','mask_strategy','unavailable_count','permutation_index','permutation_repeat'] if x in f]
        for keys,g in f.groupby(conditions+['seed','date_id'],dropna=False,sort=False):
            assert not g.sample_id.duplicated().any(),(path.name,keys)
            info={k:(None if pd.isna(v) else v.item() if isinstance(v,np.generic) else v) for k,v in zip(conditions+['seed','date_id'],keys)}
            case={k:info[k] for k in conditions};case_id=path.stem+'|'+json.dumps(case,sort_keys=True,separators=(',',':'))
            records.append({'source':path.stem,'case_id':case_id,**info,**metrics(g)})
    per=pd.DataFrame(records);per.to_csv(args.output/'date_seed_metrics.csv',index=False)
    metric_names=['macro_f1','balanced_accuracy','mcc','nll','brier'];date_mean=per.groupby(['source','case_id','date_id'],sort=False)[metric_names].mean().reset_index();date_mean.to_csv(args.output/'date_metrics.csv',index=False)
    overall=date_mean.groupby(['source','case_id'],sort=False)[metric_names].mean().reset_index();overall=overall.merge(date_mean.groupby(['source','case_id']).macro_f1.agg(worst_date_macro_f1='min',n_dates='size').reset_index(),on=['source','case_id']);overall.to_csv(args.output/'overall_metrics.csv',index=False)
    reference=frames['reference_predictions'];control=frames['alpha0_predictions'];a=date_values(reference);b=date_values(control)
    assert a.index.equals(b.index) and len(a)==8
    contrasts=[{'comparison':'alpha_075_minus_alpha_0',**bootstrap((a-b).to_numpy())}]
    base=frames['reconstruction_predictions'];crg=frames['crg_predictions'];weights=pd.read_csv(ROOT/'data/selection/crg_classwise_weights.csv');global_weights=pd.read_csv(ROOT/'data/selection/global_fusion_weights.csv')
    keys=['seed','mask_strategy','unavailable_count','mask_id','date_id','sample_id'];names=['ridge_reconstruct_rbf','ridge_reconstruct_shrinkage_lda','multimask_global_extratrees'];pc=PROBABILITY_COLUMNS
    sorted_crg=crg.sort_values(keys).reset_index(drop=True);stacks=[]
    for name in names:
        part=base[base.method.eq(name)].sort_values(keys).reset_index(drop=True)
        assert part[keys].equals(sorted_crg[keys]) and part.y_true.equals(sorted_crg.y_true)
        stacks.append(part[pc].to_numpy())
    stack=np.stack(stacks,axis=1);replayed=[]
    for seed,g in sorted_crg.groupby('seed',sort=False):
        w=weights[weights.seed.eq(seed)].pivot(index='method',columns='class',values='classwise_weight').loc[names,LABELS].to_numpy()
        replayed.append(float(abs(apply_classwise_gate(stack[g.index],w)-g[pc].to_numpy()).max()))
    assert max(replayed)<1e-12
    eq=frames['equal_fusion_predictions'].sort_values(keys).reset_index(drop=True)
    assert eq[keys].equals(sorted_crg[keys]);assert abs(stack.mean(1)-eq[pc].to_numpy()).max()<1e-12
    subset=lambda f:f[f.mask_strategy.eq('random')&f.unavailable_count.eq(16)]
    crg_dates=date_values(subset(crg))
    for name in ['equal_fusion_predictions','global_fusion_predictions']:
        baseline_dates=date_values(subset(frames[name]));assert crg_dates.index.equals(baseline_dates.index)
        contrasts.append({'comparison':'crg_minus_'+name+'_random16',**bootstrap((crg_dates-baseline_dates).to_numpy())})
    pd.DataFrame(contrasts).to_csv(args.output/'paired_comparisons.csv',index=False)
    known={'alpha075':float(a.mean()),'alpha0':float(b.mean()),'crg_random16':float(crg_dates.mean()),'equal_random16':float(date_values(subset(frames['equal_fusion_predictions'])).mean())}
    expected=json.loads((ROOT/'data/metadata/expected_scores.json').read_text(encoding='utf8'))
    for key,value in expected.items():assert abs(known[key]-value)<1e-12,(key,known[key],value)
    write_json(args.output/'audit.json',{'status':'PASS','model_fits':0,'prediction_rows_by_source':counts,'date_seed_cells':len(per),'case_count':len(overall),'crg_saved_probability_replay_max_error':max(replayed),'known_scores':known,'notes':['Fixed five-class metrics.','Seeds averaged within each date; dates weighted equally.','Global fusion shares the fixed beta=10 setting.','Alpha comparison uses a 500-iteration budget; alpha0 met the original stopping rule in 0 of 200 fits.']})
    print(json.dumps({'status':'PASS','scores':known,'output':str(args.output)},ensure_ascii=False))
if __name__=='__main__':main()
