"""Extract aligned features from the source archive; this may take tens of minutes."""
from _common import ROOT,sha,write_json
from pathlib import Path
import argparse,time,json
import numpy as np
import yaml
from enose_drift.preparation import prepare_primary
from enose_drift.preprocessing import RobustTrainTransformer
from enose_drift.tcss import make_consecutive_environments,_cluster_resample_indices
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',type=Path,default=ROOT/'data/raw/Dataset.zip');p.add_argument('--output',type=Path,required=True);p.add_argument('--check-only',action='store_true');args=p.parse_args()
    meta=json.loads((ROOT/'data/dataset.json').read_text(encoding='utf8'))
    if sha(args.archive)!=meta['archive_sha256']:raise ValueError('Archive SHA-256 mismatch.')
    if args.output.exists():raise FileExistsError('Use a new output folder; distributed features are never overwritten.')
    if args.check_only:print('Archive identity and output path verified.');return
    started=time.monotonic();config=yaml.safe_load((ROOT/'configs/primary.yaml').read_text(encoding='utf8'))
    prepared=prepare_primary(args.archive,config,include_sensitivity=False);v=prepared.variants['primary'];manifest=prepared.scored_manifest.reset_index(drop=True)
    train=manifest.split.eq('train').to_numpy();transformer=RobustTrainTransformer().fit(v.X_by_budget[0][train]);dates=manifest.loc[train,'date'].astype(str).to_numpy(dtype=str)
    seeds=np.asarray([int(s.generate_state(1,dtype=np.uint32)[0]) for s in np.random.SeedSequence(11).spawn(200)],dtype=np.uint32);env=make_consecutive_environments(dates,6)
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output/'primary_inputs.npz',X_rank=transformer.transform(v.X_by_budget[0][train]),X_predict=v.X_by_budget[1],y_rank=manifest.loc[train,'label'].astype(str).to_numpy(dtype=str),train_dates=dates,environments=env,group_ids=np.asarray(list(v.groups),dtype=int),group_cols=np.stack([v.groups[g] for g in v.groups]),always=np.asarray(v.always_include_columns),replicate_seeds=seeds)
    manifest.to_csv(args.output/'sample_manifest.csv',index=False)
    write_json(args.output/'feature_index.json',{'feature_names':list(v.feature_names),'groups':{str(g):v.groups[g].tolist() for g in v.groups},'always_include_columns':list(map(int,v.always_include_columns))})
    write_json(args.output/'preparation.json',{'source_sha256':sha(args.archive),'rows':len(manifest),'features':len(v.feature_names),'elapsed_seconds':time.monotonic()-started,'scaling_fit_rows':int(train.sum())})
    print('Features prepared:',args.output)
if __name__=='__main__':main()
