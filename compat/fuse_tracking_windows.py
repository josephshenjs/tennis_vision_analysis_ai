"""Experimental offline fusion of full-court and far-window tracking CSVs.

Uses full-court observations as anchors. Never select solely by confidence.
"""
import argparse
import csv
import json
from pathlib import Path
import cv2
import numpy as np


def fuse(full,far,roi,agreement=35.,speed_gate=35.,anchor_radius=8):
    if len(full)!=len(far): raise ValueError('Window CSV lengths differ.')
    def point(row): return np.array([float(row['X']),float(row['Y'])])
    def seen(row): return bool(int(row['Visibility'])) and np.isfinite(point(row)).all()
    x,y,w,h=roi
    def local(i):
        p=point(far[i])
        return seen(far[i]) and x<=p[0]<x+w and y<=p[1]<y+h
    out=[]; audit=[]
    for i,(base,zoom) in enumerate(zip(full,far)):
        if int(base['Frame'])!=i or int(zoom['Frame'])!=i:
            raise ValueError('CSV frames must be consecutive and aligned.')
        if abs(float(base['TimeSeconds'])-float(zoom['TimeSeconds']))>1e-4:
            raise ValueError('Window CSV timestamps differ.')
        use=False; reason='full_court' if seen(base) else 'missing'
        if local(i):
            # Require support on at least two nearby frames; reject isolated detections.
            neighbors=[j for j in range(max(0,i-3),min(len(far),i+4)) if j!=i and local(j)
                       and np.linalg.norm(point(zoom)-point(far[j]))<=speed_gate*abs(i-j)]
            if len(neighbors)>=2:
                if seen(base):
                    if np.linalg.norm(point(base)-point(zoom))<=agreement:
                        use=True; reason='far_agrees_with_full'
                    else:
                        reason='full_kept_window_disagreement'
                else:
                    anchors=[j for j in range(max(0,i-anchor_radius),min(len(full),i+anchor_radius+1))
                             if seen(full[j]) and np.linalg.norm(point(zoom)-point(full[j]))<=speed_gate*abs(i-j)+agreement]
                    if anchors:
                        use=True; reason='far_bridges_full_gap'
                    else:
                        reason='far_rejected_no_full_anchor'
            else:
                reason='far_rejected_isolated'
        row=dict(zoom if use else base)
        out.append(row)
        audit.append(dict(Frame=i,SelectedWindow='far' if use else ('full' if seen(base) else 'none'),Reason=reason,
                          FullVisible=int(seen(base)),FarVisible=int(seen(zoom))))
    return out,audit


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--full-csv',type=Path,required=True)
    p.add_argument('--far-csv',type=Path,required=True)
    p.add_argument('--far-calibration',type=Path,required=True)
    p.add_argument('--video',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--overwrite',action='store_true')
    a=p.parse_args()
    def read(path):
        with path.open(newline='') as f: return list(csv.DictReader(f))
    full,far=read(a.full_csv),read(a.far_csv)
    roi=json.loads(a.far_calibration.read_text())['tracking_roi_xywh']
    rows,audit=fuse(full,far,roi)
    a.output_dir.mkdir(parents=True,exist_ok=True); stem=a.video.stem
    csv_path=a.output_dir/f'{stem}_predictions.csv'
    video_path=a.output_dir/f'{stem}_tracked.mp4'
    audit_path=a.output_dir/f'{stem}_fusion_audit.csv'
    summary_path=a.output_dir/f'{stem}_fusion_summary.json'
    outputs=[csv_path,video_path,audit_path,summary_path]+[a.output_dir/f'{stem}_{suffix}' for suffix in ['bounces.json','bounce_scores.csv','bounces.mp4']]
    if not a.overwrite and any(f.exists() for f in outputs): raise FileExistsError('Output exists; use --overwrite.')
    for path,items in [(csv_path,rows),(audit_path,audit)]:
        with path.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(items[0])); w.writeheader(); w.writerows(items)
    cap=cv2.VideoCapture(str(a.video)); writer=None
    try:
        if not cap.isOpened(): raise RuntimeError('Cannot open original video.')
        width,height=int(cap.get(3)),int(cap.get(4)); fps=cap.get(cv2.CAP_PROP_FPS)
        writer=cv2.VideoWriter(str(video_path),cv2.VideoWriter_fourcc(*'mp4v'),fps,(width,height))
        if not writer.isOpened(): raise RuntimeError('Cannot encode fusion preview.')
        for row,decision in zip(rows,audit):
            ok,frame=cap.read()
            if not ok: raise RuntimeError('Video shorter than tracking CSV.')
            if int(row['Visibility']):
                cv2.circle(frame,(int(row['X']),int(row['Y'])),8,(0,255,0),2)
            x,y,w,h=roi; cv2.rectangle(frame,(x,y),(x+w,y+h),(255,180,0),1)
            cv2.putText(frame,f'Frame {row["Frame"]} | selected {decision["SelectedWindow"]}',(15,30),0,.7,(0,255,0),2)
            writer.write(frame)
    finally:
        cap.release()
        if writer is not None: writer.release()
    from .bounce_detection import process_csv
    process_csv(csv_path,video_path,a.output_dir,overwrite=a.overwrite)
    from collections import Counter
    summary=dict(full_csv=str(a.full_csv),far_csv=str(a.far_csv),far_roi_xywh=roi,
                 fusion_parameters=dict(agreement_px=35,speed_gate_px_per_frame=35,anchor_radius_frames=8),
                 frames=len(rows),full_detected=sum(int(r['Visibility']) for r in full),
                 far_detected=sum(int(r['Visibility']) for r in far),
                 fused_detected=sum(int(r['Visibility']) for r in rows),
                 decisions=dict(Counter(r['Reason'] for r in audit)),
                 note='Offline experimental association; counts are not labelled accuracy. Full-window conflicts retained.')
    summary_path.write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
