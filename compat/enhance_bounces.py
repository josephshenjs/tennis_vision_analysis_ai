"""Generate CatBoost candidates, infer RacketVision poses, then review evidence."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video',type=Path,required=True,help='Original MP4, aligned with CSV')
    p.add_argument('--csv',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--candidates',type=Path,help='Optional existing CatBoost candidates JSON')
    p.add_argument('--rackets',type=Path,help='Optional cached RacketVision evidence JSON')
    p.add_argument('--racket-python',type=Path,default=ROOT/'runtime/python.exe')
    p.add_argument('--overwrite',action='store_true')
    a=p.parse_args()
    if a.rackets is None and not a.racket_python.is_file():
        raise FileNotFoundError('Racket inference environment missing; see BOUNCE_REVIEW.md.')
    candidates=a.candidates
    if candidates is None:
        from .bounce_detection import process_csv
        candidates=process_csv(a.csv,a.video,a.output_dir/'catboost',render_video=False,overwrite=a.overwrite)['events']
    rackets=a.rackets or a.output_dir/'racket_evidence.json'
    if a.rackets is None:
        command=[str(a.racket_python),str(ROOT/'run.py'),'_legacy','infer_racket_evidence',
                 '--video',str(a.video),'--csv',str(a.csv),'--candidates',str(candidates),'--output',str(rackets)]
        if a.overwrite: command.append('--overwrite')
        subprocess.run(command,check=True)
    command=[sys.executable,str(ROOT/'run.py'),'_legacy','review_bounce_candidates','--video',str(a.video),
             '--csv',str(a.csv),'--candidates',str(candidates),'--rackets',str(rackets),
             '--output-dir',str(a.output_dir/'review')]
    if a.overwrite: command.append('--overwrite')
    subprocess.run(command,check=True)

if __name__=='__main__': main()
