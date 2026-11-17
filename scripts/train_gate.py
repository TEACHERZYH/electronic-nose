"""Fit reconstruction models and validation-based classwise fusion."""
from _common import ROOT,load_inputs,write_json
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import yaml
from joblib import Parallel,delayed,parallel_config
from enose_drift.v17_nonlinear import prepare_v17_inputs
from enose_drift.v211_classwise_gate import run_seed
from enose_drift.preparation import _bundle_frames
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inputs',type=Path,default=ROOT/'data/processed');p.add_argument('--output',type=Path,required=True);p.add_argument('--n-jobs',type=int,default=1);p.add_argument('--check-only',action='store_true');args=p.parse_args()
    if args.n_jobs<1:raise ValueError('n-jobs must be positive.')
    if args.output.exists():raise FileExistsError('Choose a new output folder.')
    a,manifest=load_inputs(args.inputs);groups={int(g):c for g,c in zip(a['group_ids'],a['group_cols'])}
    ranking=tuple(pd.read_csv(ROOT/'data/metadata/sensor_ranking.csv').sort_values('rank').group.astype(int))
    config=yaml.safe_load((ROOT/'configs/crg.yaml').read_text(encoding='utf8'))
    standardized,labels,ids,dates,split_indices,compact_groups=prepare_v17_inputs(a['X_predict'],manifest,groups,ranking,always_include_columns=a['always'])
    if args.check_only:print('Inputs, 62-group ordering, configuration and preprocessing verified.');return
    with parallel_config(backend='loky',inner_max_num_threads=1):
        results=Parallel(n_jobs=min(args.n_jobs,5))(delayed(run_seed)(int(seed),standardized,labels,ids,dates,split_indices,compact_groups,ranking,config,'crg_local') for seed in config['seeds'])
    bundles=[b for result in results for b in result[0]];predictions,date_metrics=_bundle_frames(bundles)
    args.output.mkdir(parents=True);predictions.to_parquet(args.output/'predictions.parquet',index=False);date_metrics.to_csv(args.output/'date_metrics.csv',index=False)
    for index,name in [(1,'base_candidates'),(2,'training_masks'),(3,'beta_candidates'),(4,'classwise_weights')]:pd.concat([r[index] for r in results],ignore_index=True).to_csv(args.output/(name+'.csv'),index=False)
    write_json(args.output/'settings.json',config);print('Completed:',args.output)
if __name__=='__main__':main()
