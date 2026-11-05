"""Install the reference CatBoost checkpoint from the published v1.1 release."""
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]/'models/bounce'
URL = 'https://github.com/skisurfer13/RacketVision/releases/download/v1.1/models.zip'

def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    archive = ROOT/'reference_models_v1.1.zip'
    model = ROOT/'ctb_regr_bounce.cbm'
    if model.is_file() and (ROOT/'provenance.json').is_file():
        meta=json.loads((ROOT/'provenance.json').read_text())
        if hashlib.sha256(model.read_bytes()).hexdigest()==meta['model_sha256']:
            print(f'Already installed and checksum verified: {model}')
            return
    if not archive.is_file() or not zipfile.is_zipfile(archive):
        partial=archive.with_suffix('.zip.part')
        print('Downloading reference release (~490 MB)...',flush=True)
        urllib.request.urlretrieve(URL,partial)
        partial.replace(archive)
    with zipfile.ZipFile(archive) as z:
        names=[n for n in z.namelist() if n.endswith('ctb_regr_bounce.cbm')]
        if len(names)!=1:
            raise RuntimeError('Expected exactly one bounce model in release archive')
        model.write_bytes(z.read(names[0]))
    provenance=dict(source=URL, reference_repository='https://github.com/jack16888/RacketVision',
                    archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                    model_sha256=hashlib.sha256(model.read_bytes()).hexdigest())
    (ROOT/'provenance.json').write_text(json.dumps(provenance,indent=2))
    print(f'Installed: {model}')

if __name__=='__main__':
    main()
