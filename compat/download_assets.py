"""Download tennis data and all published weights; safe to rerun after interruption."""
from pathlib import Path
import json
import shutil
from huggingface_hub import snapshot_download, HfApi

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'RacketVision' / 'source'

def main():
    api = HfApi()
    manifest = {}
    for repo, kind, dest, patterns in [
        ('linfeng302/RacketVision', 'dataset', SOURCE / 'data', ['tennis/**', 'README.md']),
        ('linfeng302/RacketVision-Models', 'model', ROOT / 'models', ['checkpoints/**', 'README.md']),
    ]:
        revision = api.repo_info(repo, repo_type=kind).sha
        print(f'Downloading {repo} @ {revision}', flush=True)
        snapshot_download(repo, repo_type=kind, revision=revision,
                          local_dir=str(dest), allow_patterns=patterns, max_workers=4)
        manifest[repo] = revision
    mapping = {
        'balltrack_best.pth': 'BallTrack',
        'epoch_300.pth': 'RacketPose',
        'best_PCK_epoch_90.pth': 'RacketPose',
        'crosslstm_long_badminton.pth': 'TrajPred',
        'crosslstm_long_tabletennis.pth': 'TrajPred',
        'crosslstm_short_tennis.pth': 'TrajPred',
    }
    for name, module in mapping.items():
        target = SOURCE / module / 'checkpoints' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / 'models' / 'checkpoints' / name, target)
    (ROOT / 'download_manifest.json').write_text(json.dumps(manifest, indent=2))
    print('Downloads complete; weights installed in upstream module directories.')

if __name__ == '__main__':
    main()
