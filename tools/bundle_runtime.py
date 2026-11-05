"""Build a local embedded Python runtime from a verified working environment.
Run using the existing .venv-racket interpreter during packaging, not by end users.
"""
from pathlib import Path
import hashlib
import importlib.metadata as metadata
import json
import shutil
import urllib.request
import zipfile
from packaging.requirements import Requirement

APP=Path(__file__).resolve().parents[1]
RUNTIME=APP/'runtime'
SITE=RUNTIME/'Lib/site-packages'
ROOTS=['torch','torchvision','numpy','opencv-python','mmcv','mmengine','mmdet','mmpose','catboost']


def main():
    RUNTIME.mkdir(parents=True,exist_ok=True); SITE.mkdir(parents=True,exist_ok=True)
    archive=RUNTIME/'python-3.11.9-embed-amd64.zip'
    url='https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip'
    if not archive.exists(): urllib.request.urlretrieve(url,archive)
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            target=(RUNTIME/name).resolve()
            if not target.is_relative_to(RUNTIME.resolve()): raise ValueError('Invalid Python archive path')
        z.extractall(RUNTIME)
    # Import path stays local. The application launcher inserts its own package parent.
    (RUNTIME/'python311._pth').write_text('python311.zip\n.\nLib/site-packages\nimport site\n')
    pending=list(ROOTS); installed={}; copied=0
    while pending:
        name=pending.pop(); normalized=name.lower().replace('_','-')
        if normalized in installed: continue
        dist=metadata.distribution(name)
        installed[normalized]=dist.version
        for dependency in dist.requires or []:
            req=Requirement(dependency)
            if req.marker is None or req.marker.evaluate({'extra':''}): pending.append(req.name)
        for relative in dist.files or []:
            target=(SITE/str(relative)).resolve()
            # Distribution scripts point outside site-packages and are not runtime libraries.
            if not target.is_relative_to(SITE.resolve()): continue
            source=Path(dist.locate_file(relative))
            if not source.is_file(): continue
            target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists() or target.stat().st_size!=source.stat().st_size:
                shutil.copy2(source,target)
            copied+=1
        print(f'Bundled {name} {dist.version}',flush=True)
    manifest=dict(python='3.11.9 embedded Windows x64',python_url=url,
                  python_archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                  distributions=installed,distribution_file_count=copied)
    (RUNTIME/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(f'Runtime ready: {RUNTIME}')

if __name__=='__main__': main()
