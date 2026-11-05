"""RacketVision RTMDet+RTMPose evidence near CatBoost candidate frames.
Run in .venv-racket. Coordinates and frame IDs remain in original-video pixels.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'RacketVision/source/RacketPose'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video',type=Path,required=True)
    p.add_argument('--csv',type=Path,required=True)
    p.add_argument('--candidates',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--radius',type=int,default=6)
    p.add_argument('--context-size',type=int,default=600)
    p.add_argument('--bbox-threshold',type=float,default=.3)
    p.add_argument('--overwrite',action='store_true')
    a=p.parse_args()
    if a.output.exists() and not a.overwrite: raise FileExistsError(a.output)
    if a.radius<0 or a.context_size<64: p.error('Invalid radius or context size.')
    import cv2
    import numpy as np
    import torch
    from mmengine.registry import DefaultScope
    from mmdet.apis import init_detector,inference_detector
    from mmpose.apis import init_model,inference_topdown
    from .review_bounce_candidates import ball_at
    with a.csv.open(newline='') as f: rows=list(csv.DictReader(f))
    if any(int(r['Frame'])!=i for i,r in enumerate(rows)): raise ValueError('CSV frames not consecutive.')
    events=json.loads(a.candidates.read_text())['events']
    frames=sorted({i for e in events for i in range(max(0,e['frame']-a.radius),min(len(rows),e['frame']+a.radius+1))})
    det_checkpoint=SOURCE/'checkpoints/epoch_300.pth'
    pose_checkpoint=SOURCE/'checkpoints/best_PCK_epoch_90.pth'
    det=init_detector(str(SOURCE/'configs/detection/rtmdet_m_racket_infer.py'),str(det_checkpoint),device=a.device)
    pose=init_model(str(SOURCE/'configs/pose/rtmpose_m_racket_infer.py'),str(pose_checkpoint),device=a.device)
    det.eval(); pose.eval()
    cap=cv2.VideoCapture(str(a.video))
    if not cap.isOpened(): raise RuntimeError('Cannot open video.')
    width,height=int(cap.get(3)),int(cap.get(4)); fps=cap.get(5)
    output=dict(source=str(a.video.resolve()),source_frames=len(rows),image_size=[width,height],fps=fps,
                model='RacketVision RTMDet-M + RTMPose-M',sport='tennis',category_id=2,
                bbox_threshold=a.bbox_threshold,radius_frames=a.radius,context_size=a.context_size,
                det_sha256=hashlib.sha256(det_checkpoint.read_bytes()).hexdigest(),
                pose_sha256=hashlib.sha256(pose_checkpoint.read_bytes()).hexdigest(),frames={})
    try:
        for count,i in enumerate(frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES,i); ok,frame=cap.read()
            if not ok: raise RuntimeError(f'Cannot decode {i}')
            ball,interpolated=ball_at(rows,i)
            if ball is None:
                x=y=0; w=width; h=height
            else:
                w=min(a.context_size,width); h=min(a.context_size,height)
                x=int(np.clip(ball[0]-w/2,0,width-w)); y=int(np.clip(ball[1]-h/2,0,height-h))
            image=frame[y:y+h,x:x+w]
            with torch.inference_mode(),DefaultScope.overwrite_default_scope('mmdet'):
                instances=inference_detector(det,image).pred_instances
            boxes=instances.bboxes.cpu().numpy(); scores=instances.scores.cpu().numpy(); labels=instances.labels.cpu().numpy()
            keep=(labels==2)&(scores>=a.bbox_threshold)
            boxes=boxes[keep]; scores=scores[keep]
            keep_area=np.array([0<(b[2]-b[0])*(b[3]-b[1])<w*h*.5 for b in boxes],bool)
            boxes=boxes[keep_area][:4]; scores=scores[keep_area][:4]
            predictions=[]
            if len(boxes):
                with torch.inference_mode(),DefaultScope.overwrite_default_scope('mmpose'):
                    results=inference_topdown(pose,image,boxes)
                for box,score,result in zip(boxes,scores,results):
                    pts=result.pred_instances.keypoints[0].copy()
                    pts+=np.array([x,y])
                    predictions.append(dict(bbox=(box+np.array([x,y,x,y])).tolist(),bbox_score=float(score),
                                            keypoints=pts.tolist(),keypoint_scores=result.pred_instances.keypoint_scores[0].tolist()))
            output['frames'][str(i)]=dict(evaluated=True,roi_xywh=[x,y,w,h],predictions=predictions)
            if count%10==0: print(f'Racket frames {count+1}/{len(frames)}, frame {i}, rackets {len(predictions)}',flush=True)
    finally: cap.release()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(output,indent=2))
    print(f'Saved racket evidence: {a.output}')

if __name__=='__main__': main()
