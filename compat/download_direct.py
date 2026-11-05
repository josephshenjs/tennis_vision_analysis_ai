"""Download pinned public assets with integrity verification and no per-file HEAD calls."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]

def get_json(url):
    with urllib.request.urlopen(url, timeout=120) as response:
        return json.load(response)

def download(job):
    url, dest, size, digest, lfs = job
    def valid(p):
        if not p.is_file() or p.stat().st_size != size:
            return False
        h = hashlib.sha256() if lfs else hashlib.sha1()
        if not lfs:
            h.update(f'blob {size}\0'.encode())
        with p.open('rb') as f:
            for b in iter(lambda:f.read(1024*1024), b''):
                h.update(b)
        return h.hexdigest() == digest
    if valid(dest):
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + '.direct-part')
    for attempt in range(20):
        try:
            with urllib.request.urlopen(url, timeout=120) as response, part.open('wb') as f:
                while True:
                    chunk = response.read(1024*1024)
                    if not chunk:
                        break
                    f.write(chunk)
            if not valid(part):
                raise RuntimeError(f'Integrity check failed: {dest}')
            part.replace(dest)
            return
        except Exception as error:
            print(f'Retrying {dest.name}: {error}', flush=True)
            if attempt == 19:
                raise
            delay = min(60, 2**attempt * 3)
            if getattr(error, 'code', None) == 429:
                delay = max(60, int(error.headers.get('Retry-After', '60')))
            time.sleep(delay)

def main():
    manifest = {}
    for repo, kind, dest, prefix in [
        ('linfeng302/RacketVision-Models','model',ROOT/'models',''),
        ('linfeng302/RacketVision','dataset',ROOT/'RacketVision/source/data','tennis/'),
    ]:
        category = 'datasets' if kind == 'dataset' else 'models'
        info = get_json(f'https://huggingface.co/api/{category}/{repo}?blobs=true')
        revision = info['sha']
        manifest[repo] = revision
        jobs=[]
        for entry in info['siblings']:
            name=entry['rfilename']
            if (prefix and not name.startswith(prefix)) or (not prefix and not name.startswith('checkpoints/')):
                continue
            urlprefix = 'datasets/' if kind == 'dataset' else ''
            lfs=entry.get('lfs')
            jobs.append((f'https://huggingface.co/{urlprefix}{repo}/resolve/{revision}/{name}',dest/name,entry['size'], lfs['sha256'] if lfs else entry['blobId'], bool(lfs)))
        # Make videos and split files available first.
        jobs.sort(key=lambda j: (0 if str(j[1]).endswith('.mp4') else 1, str(j[1])))
        print(f'{repo}: {len(jobs)} files, {sum(j[2] for j in jobs):,} bytes',flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
            for i,_ in enumerate(pool.map(download,jobs),1):
                if i%100==0 or i==len(jobs):
                    print(f'Verified {i}/{len(jobs)} files',flush=True)
    mapping={'balltrack_best.pth':'BallTrack','epoch_300.pth':'RacketPose','best_PCK_epoch_90.pth':'RacketPose','crosslstm_long_badminton.pth':'TrajPred','crosslstm_long_tabletennis.pth':'TrajPred','crosslstm_short_tennis.pth':'TrajPred'}
    import shutil
    for name,module in mapping.items():
        target=ROOT/'RacketVision/source'/module/'checkpoints'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/'models/checkpoints'/name,target)
    (ROOT/'download_manifest.json').write_text(json.dumps(manifest,indent=2))
    print('All requested assets downloaded and integrity verified.',flush=True)

if __name__=='__main__':
    main()
