"""Verify every distributed file and validate portable input tables."""
from _common import ROOT,sha,load_inputs
from pathlib import Path
import argparse,json
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT);args=p.parse_args();root=args.root.resolve()
    manifest=json.loads((root/'SHA256_manifest.json').read_text(encoding='utf8'))
    errors=[]
    for item in manifest:
        path=root/item['file']
        if not path.is_file() or sha(path)!=item['sha256']:errors.append(item['file'])
    if errors:raise ValueError('Missing or changed files: '+', '.join(errors[:10]))
    arrays,frame=load_inputs(root/'data/processed')
    print(json.dumps({'status':'PASS','verified_files':len(manifest),'rows':len(frame),'training_rows':len(arrays['X_rank'])}))
if __name__=='__main__':main()
