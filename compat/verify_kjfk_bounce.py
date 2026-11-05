"""Verify kjfk bounce checkpoint using its author's exact feature/event code.
Run with .venv-kjfk/Scripts/python.exe, not the CUDA/CatBoost environment.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import warnings

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'reference_tennis_auto_scoring'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv',type=Path,required=True)
    p.add_argument('--calibration',type=Path,required=True)
    p.add_argument('--video',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--model',type=Path,default=ROOT/'models/kjfk_bounce/bounce_model.joblib')
    p.add_argument('--threshold',type=float,default=.5)
    p.add_argument('--max-gap',type=int,default=8,help='Author max frame-step for filling gaps (8 means up to 7 missing frames)')
    p.add_argument('--smooth-window',type=int,default=5)
    p.add_argument('--event-policy',choices=['author','score-nms'],default='author',help='Original event layer or classify first then suppress by probability')
    p.add_argument('--no-video',action='store_true')
    p.add_argument('--overwrite',action='store_true')
    a=p.parse_args()
    import numpy as np
    import cv2
    import sklearn
    from sklearn.exceptions import InconsistentVersionWarning
    from tennis.trajectory import from_detections
    from tennis.bounce_learned import LearnedBounceDetector,propose,FEATURE_NAMES
    from tennis.bounce import detect_events
    from .court_geometry import load_calibration,transform
    if not 0<a.threshold<1 or a.max_gap<1:
        p.error('Invalid threshold or gap limit.')
    a.output_dir.mkdir(parents=True,exist_ok=True)
    stem=a.csv.stem.removesuffix('_predictions')
    paths={name:a.output_dir/f'{stem}_kjfk_{name}.{suffix}' for name,suffix in
           [('candidates','csv'),('events','json'),('report','json')]}
    if not a.no_video: paths['video']=a.output_dir/f'{stem}_kjfk_bounces.mp4'
    if not a.overwrite and any(f.exists() for f in paths.values()): raise FileExistsError('Output exists.')
    cap=cv2.VideoCapture(str(a.video))
    try:
        if not cap.isOpened(): raise RuntimeError('Cannot open video.')
        width,height=int(cap.get(3)),int(cap.get(4)); fps=cap.get(cv2.CAP_PROP_FPS)
    finally: cap.release()
    if abs(fps-30)>.5: raise ValueError('This adapter currently verifies near-30-FPS clips only; temporal resampling needed for other rates.')
    calibration=load_calibration(a.calibration,width,height)
    with a.csv.open(newline='') as f: rows=list(csv.DictReader(f))
    records=[]
    for i,row in enumerate(rows):
        if int(row['Frame'])!=i: raise ValueError('CSV must retain consecutive original frame numbers.')
        if int(row['Visibility']):
            image=np.array([float(row['X']),float(row['Y'])])
            court=transform([image],calibration['H_image_to_court_m'])[0]
            if np.isfinite(image).all() and np.isfinite(court).all():
                records.append(dict(frame=i,image=image,court=court,confidence=float(row['Confidence'])))
    # Author training uses the same from_detections defaults (gap 8, smoothing 5).
    trajectory=from_detections(records,max_gap=a.max_gap,smooth_window=a.smooth_window)
    with warnings.catch_warnings():
        warnings.simplefilter('error',InconsistentVersionWarning)
        detector=LearnedBounceDetector.load(a.model,threshold=a.threshold)
    if detector.model.n_features_in_!=len(FEATURE_NAMES): raise ValueError('Feature schema mismatch.')
    candidates=propose(trajectory)
    scored=detector.predict(candidates)
    by_frame={candidate.frame:prob for candidate,prob in scored}
    # Use original event-layer deduplication, without player boxes. No IN/OUT export.
    # Do not allow the upstream fallback-on-model-error to hide failure.
    class CheckedDetector:
        def accept(self,items):
            return [(c,by_frame[c.frame]) for c in items if by_frame[c.frame]>=a.threshold]
    events=detect_events(trajectory,fps=fps,bounce_model=CheckedDetector())
    author_frames={e.frame for e in events if e.type.value=='bounce'}
    final_frames=set(author_frames)
    if a.event_policy=='score-nms':
        final_frames=set()
        separation=max(2,int(fps/6))
        for candidate,prob in sorted(scored,key=lambda item:(-item[1],item[0].frame)):
            if prob>=a.threshold and all(abs(candidate.frame-kept)>=separation for kept in final_frames):
                final_frames.add(candidate.frame)
    audit=[]
    for candidate,prob in scored:
        audit.append(dict(Frame=candidate.frame,TimeSeconds=candidate.frame/fps,
                          Probability=prob,AboveThreshold=int(prob>=a.threshold),
                          SelectedEvent=int(candidate.frame in final_frames),
                          **dict(zip(FEATURE_NAMES,map(float,candidate.features)))))
    with paths['candidates'].open('w',newline='') as f:
        names=['Frame','TimeSeconds','Probability','AboveThreshold','SelectedEvent']+FEATURE_NAMES
        w=csv.DictWriter(f,fieldnames=names); w.writeheader(); w.writerows(audit)
    exported=[]
    for frame in sorted(final_frames):
        sample=trajectory.get(frame)
        exported.append(dict(frame=frame,time_seconds=frame/fps,x=float(sample.image[0]),
                             y=float(sample.image[1]),score=by_frame[frame],interpolated=bool(sample.interpolated)))
    paths['events'].write_text(json.dumps(dict(threshold=a.threshold,events=exported),indent=2))
    source_revision=subprocess.check_output(['git','-C',str(ROOT/'reference_tennis_auto_scoring'),'rev-parse','HEAD'],text=True).strip()
    focus={}
    for frame,label in [(79,'user_confirmed_bounce'),(166,'user_confirmed_not_bounce'),(210,'user_confirmed_bounce')]:
        focus[str(frame)]=dict(label=label,matching_events=[e for e in exported if abs(e['frame']-frame)<=2],
                              nearby_candidates=[r for r in audit if abs(r['Frame']-frame)<=3])
    report=dict(model_repo='kjfk/tennis-bounce-detector',model_revision='95bc7a22032c350ce9a57c83053d9692ad23642b',
                model_sha256=hashlib.sha256(a.model.read_bytes()).hexdigest(),author_source_revision=source_revision,
                sklearn=sklearn.__version__,numpy=np.__version__,input_csv=str(a.csv),calibration=str(a.calibration),
                threshold=a.threshold,max_gap_frame_step=a.max_gap,smooth_window=a.smooth_window,
                player_boxes_supplied=False,event_policy=a.event_policy,author_event_frames=sorted(author_frames),feature_count=len(FEATURE_NAMES),proposals=len(audit),
                above_threshold=sum(r['AboveThreshold'] for r in audit),event_frames=sorted(final_frames),
                focus=focus,note='Same upstream ball coordinates; author features, proposals and smoothing; selected event policy recorded separately. Three partial user labels only; no overall accuracy claim. Scores are model probabilities, not verified calibration.')
    paths['report'].write_text(json.dumps(report,indent=2))
    if not a.no_video:
        from .bounce_detection import render
        render(a.video,paths['video'],exported,len(rows))
    print(json.dumps(dict(event_frames=sorted(final_frames),proposals=len(audit),
                          focus={k:dict(label=v['label'],matched=[e['frame'] for e in v['matching_events']],
                                        candidates=[(r['Frame'],round(r['Probability'],4),r['SelectedEvent']) for r in v['nearby_candidates']]) for k,v in focus.items()}),indent=2))

if __name__=='__main__': main()
