"""Single public entry point. Heavy dependencies live only in the local runtime."""
import argparse
import asyncio
import os
from pathlib import Path
import subprocess
import sys
from .paths import ROOT,BOUNCE_CHECKPOINT


def use_local_runtime(argv):
    executable=ROOT/'runtime/python.exe'
    if Path(sys.executable).resolve()==executable.resolve(): return None
    if not executable.is_file():
        raise RuntimeError(f'Bundled runtime missing: {executable}. This application requires its runtime folder.')
    # run.py locates the package relative to itself; no external PYTHONPATH or venv.
    return subprocess.call([str(executable),str(ROOT/'run.py'),*argv])


def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0]=='_legacy':
        allowed={'test_video','detect_bounces','calibrate_court','review_bounce_candidates','infer_racket_evidence','enhance_bounces','fuse_tracking_windows'}
        if len(argv)<2 or argv[1] not in allowed: raise ValueError('Unknown compatibility command.')
        code=use_local_runtime(argv)
        if code is not None: return code
        import importlib
        sys.argv=[argv[1],*argv[2:]]
        importlib.import_module('.compat.'+argv[1],package='rally_analysis').main()
        return 0
    p=argparse.ArgumentParser(prog='rally_analysis',description='Portable CUDA tennis analysis: tracking, bounce candidates and racket evidence review.')
    commands=p.add_subparsers(dest='command',required=True)
    analyze=commands.add_parser('analyze',help='Analyze one video, multiple videos or a directory')
    analyze.add_argument('inputs',nargs='+',type=Path)
    analyze.add_argument('--court-calibration','--calibration',type=Path,default=ROOT/'calibrations/home_court.json')
    analyze.add_argument('--output-dir',type=Path,default=ROOT/'outputs')
    analyze.add_argument('--far-roi',type=int,nargs=4,metavar=('X','Y','WIDTH','HEIGHT'))
    analyze.add_argument('--court-margin-m',type=float,default=1.,help='Lateral run-off margin for target-court corridor, in ground metres')
    analyze.add_argument('--ball-lift',type=float,default=.25,help='Upward image-space flight allowance as a fraction of image height')
    analyze.add_argument('--single-window',action='store_true',help='Disable the additional far-court view')
    analyze.add_argument('--no-local-recovery',action='store_true',help='Disable bounded local high-resolution retries for missing ball detections')
    analyze.add_argument('--device',default='cuda:0')
    analyze.add_argument('--batch-size',type=int,default=4,help='Source frames per batch; two views use twice as many model inputs')
    analyze.add_argument('--jobs',type=int,default=2,help='Maximum in-flight clips; CUDA inference remains serialized')
    analyze.add_argument('--cpu-workers',type=int,default=2)
    analyze.add_argument('--prefetch',type=int,default=2,help='Bounded decoded/preprocessed batches')
    analyze.add_argument('--median-samples',type=int,default=100)
    analyze.add_argument('--max-frames',type=int,default=0)
    analyze.add_argument('--threshold',type=float,default=.5)
    analyze.add_argument('--bounce-threshold',type=float,default=.55)
    analyze.add_argument('--bounce-model',type=Path,default=BOUNCE_CHECKPOINT)
    analyze.add_argument('--amp',action='store_true',help='Optional mixed precision; may change marginal detections, validate before production')
    analyze.add_argument('--no-video',action='store_true',help='Only export CSV/JSON, skipping final video encoding')
    analyze.add_argument('--overwrite',action='store_true')
    commands.add_parser('self-test',help='Run bundled CPU unit tests using the local runtime')
    commands.add_parser('calibrate',help='Click four court corners and generate standard keypoints',add_help=False)
    commands.add_parser('review',help='Review existing candidates and racket JSON',add_help=False)
    if argv and argv[0]=='self-test':
        code=use_local_runtime(argv)
        if code is not None: return code
        import unittest
        tests=unittest.defaultTestLoader.discover(str(ROOT/'tests'))
        return 0 if unittest.TextTestRunner(verbosity=2).run(tests).wasSuccessful() else 1
    if argv and argv[0] in ('calibrate','review'):
        code=use_local_runtime(argv)
        if code is not None: return code
        sys.argv=[f'rally_analysis {argv[0]}',*argv[1:]]
        if argv[0]=='calibrate':
            from .compat.calibrate_court import main as command
        else:
            from .compat.review_bounce_candidates import main as command
        command(); return 0
    args=p.parse_args(argv)
    if not args.device.startswith('cuda'): p.error('CUDA is required; use --device cuda:0 (or another CUDA index).')
    if not 1<=args.batch_size<=32 or not 1<=args.jobs<=16 or not 1<=args.cpu_workers<=32 or not 1<=args.prefetch<=8:
        p.error('Supported ranges: batch-size 1..32, jobs 1..16, cpu-workers 1..32, prefetch 1..8.')
    if not 1<=args.median_samples<=500 or args.max_frames<0 or not 0<args.threshold<1 or not 0<args.bounce_threshold<1:
        p.error('Invalid sample count, frame limit or score threshold.')
    if not 0<=args.court_margin_m<=5 or not 0<=args.ball_lift<=.75:
        p.error('court-margin-m must be 0..5 and ball-lift 0..0.75.')
    if args.single_window and args.far_roi: p.error('--far-roi cannot be used with --single-window.')
    videos=[]
    for item in args.inputs:
        if item.is_dir(): videos.extend(sorted(p.resolve() for p in item.iterdir() if p.suffix.lower() in {'.mp4','.mov','.mkv','.avi'}))
        elif item.is_file(): videos.append(item.resolve())
        else: p.error(f'Input not found: {item}')
    args.videos=list(dict.fromkeys(videos))
    if not args.videos: p.error('No supported videos found.')
    args.court_calibration=args.court_calibration.resolve(); args.output_dir=args.output_dir.resolve()
    if not args.court_calibration.is_file(): p.error(f'Calibration missing: {args.court_calibration}')
    code=use_local_runtime(argv)
    if code is not None: return code
    os.environ.setdefault('OMP_NUM_THREADS','4')
    os.environ.setdefault('MKL_NUM_THREADS','4')
    from .pipeline import analyze_videos
    try: return asyncio.run(analyze_videos(args))
    except KeyboardInterrupt:
        print('Cancelled. Completed outputs are preserved.',file=sys.stderr); return 130
    except Exception as error:
        print(f'ERROR: {error}',file=sys.stderr); return 1
