"""Download the unchanged source archive and verify its identity."""
from _common import ROOT,sha
from pathlib import Path
import argparse,json,urllib.request
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'data/raw/Dataset.zip');args=p.parse_args()
    meta=json.loads((ROOT/'data/dataset.json').read_text(encoding='utf8'));out=args.output
    if out.exists():
        if out.stat().st_size!=meta['archive_bytes'] or sha(out)!=meta['archive_sha256']:raise ValueError('Existing file fails identity verification; it was not overwritten.')
        print('Verified existing archive.');return
    out.parent.mkdir(parents=True,exist_ok=True);tmp=out.with_suffix('.download')
    if tmp.exists():raise FileExistsError('Incomplete download exists; inspect it before restarting.')
    req=urllib.request.Request(meta['download_url'],headers={'User-Agent':'enose-drift-toolkit/1.0'})
    with urllib.request.urlopen(req,timeout=120) as response,tmp.open('xb') as f:
        while True:
            block=response.read(1024*1024)
            if not block:break
            f.write(block)
    if tmp.stat().st_size!=meta['archive_bytes'] or sha(tmp)!=meta['archive_sha256']:raise ValueError('Downloaded archive fails identity verification; partial file retained.')
    tmp.replace(out);print('Download verified:',out)
if __name__=='__main__':main()
