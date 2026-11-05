"""Evidence-based review of CatBoost candidates using ball and RacketVision poses.
No retraining, score calibration, court-plane exclusion, or IN/OUT decisions.
"""
import argparse
import csv
import json
from pathlib import Path
import cv2
import numpy as np


def valid_ball(row):
    return bool(int(row['Visibility'])) and np.isfinite([float(row['X']),float(row['Y'])]).all()


def quality(rows,frame,width,height,radius=4):
    low=max(0,frame-radius); high=min(len(rows),frame+radius+1)
    observed=[i for i in range(low,high) if valid_ball(rows[i])]
    max_gap=current=0
    for i in range(low,high):
        current=0 if i in observed else current+1
        max_gap=max(max_gap,current)
    speeds=[]
    for a,b in zip(observed,observed[1:]):
        p=np.array([float(rows[a]['X']),float(rows[a]['Y'])]); q=np.array([float(rows[b]['X']),float(rows[b]['Y'])])
        speeds.append(float(np.linalg.norm(q-p)/(b-a)))
    diagonal=float(np.hypot(width,height))
    peak=max(speeds,default=0)/diagonal
    before=[i for i in observed if i<frame]; after=[i for i in observed if i>frame]
    change=None
    if len(before)>=2 and len(after)>=2:
        def xy(i): return np.array([float(rows[i]['X']),float(rows[i]['Y'])])
        incoming=(xy(before[-1])-xy(before[0]))/(before[-1]-before[0])
        outgoing=(xy(after[-1])-xy(after[0]))/(after[-1]-after[0])
        denom=np.linalg.norm(incoming)*np.linalg.norm(outgoing)
        if denom>1e-6: change=float(np.degrees(np.arccos(np.clip(np.dot(incoming,outgoing)/denom,-1,1))))
    return dict(observed_share=len(observed)/(high-low),missing_share=1-len(observed)/(high-low),
                max_missing_run=max_gap,max_step_diagonal_fraction=peak,
                direction_change_deg=change,before_observations=len(before),after_observations=len(after),
                window_frames=[low,high-1])


def ball_at(rows,frame,max_gap=2):
    if valid_ball(rows[frame]):
        return np.array([float(rows[frame]['X']),float(rows[frame]['Y'])]),False
    left=next((i for i in range(frame-1,max(-1,frame-max_gap-1),-1) if valid_ball(rows[i])),None)
    right=next((i for i in range(frame+1,min(len(rows),frame+max_gap+1)) if valid_ball(rows[i])),None)
    if left is None or right is None or right-left-1>max_gap: return None,False
    p=np.array([float(rows[left]['X']),float(rows[left]['Y'])]); q=np.array([float(rows[right]['X']),float(rows[right]['Y'])])
    return p+(q-p)*(frame-left)/(right-left),True


def head_distance(ball,pred,min_score=.3,min_bbox_score=None):
    """Distance to a confidently estimated racket head, in racket-head lengths."""
    proposed=pred.get('proposal_method')=='trajectory_guided_pose'
    if proposed:
        if pred.get('proposal_verified_by_frame') is None: return None
    elif float(pred.get('bbox_score',0))<(min_score if min_bbox_score is None else min_bbox_score): return None
    points=np.asarray(pred.get('keypoints',[]),float)
    scores=np.asarray(pred.get('keypoint_scores',[]),float)
    if points.shape!=(5,2) or scores.shape!=(5,) or not np.isfinite(points).all(): return None
    head=points[[0,1,3,4]]; confidence=scores[[0,1,3,4]]
    if proposed and (confidence.mean()<.5 or min(scores[0],scores[1])<.45): return None
    good=head[confidence>=min_score]
    if len(good)<3 or min(scores[0],scores[1])<min_score: return None
    scale=float(np.linalg.norm(points[0]-points[1]))
    if scale<4: return None
    hull=cv2.convexHull(good.astype(np.float32)).reshape(-1,1,2)
    distance=max(0.,-float(cv2.pointPolygonTest(hull,tuple(map(float,ball)),True)))
    return dict(distance_px=distance,distance_head_lengths=distance/scale,
                head_length_px=scale,keypoint_mean_score=float(confidence.mean()))



def short_gap_motion(rows,index):
    """Two-sided short-gap extrapolation for an impact hidden by missing detections.
    Straight interpolation across a hit erases its turning point. Require both
    independently extrapolated sides near the same confidently posed racket head.
    These positions are evidence hypotheses, never written into tracking CSV.
    """
    if valid_ball(rows[index]): return None
    left=index-1
    while left>=0 and not valid_ball(rows[left]): left-=1
    right=index+1
    while right<len(rows) and not valid_ball(rows[right]): right+=1
    if left<2 or right+2>=len(rows) or right-left-1>2: return None
    before=list(range(left-2,left+1)); after=list(range(right,right+3))
    if not all(valid_ball(rows[i]) for i in before+after): return None
    def extrapolate(indices):
        xy=np.array([[float(rows[i]['X']),float(rows[i]['Y'])] for i in indices])
        times=np.array(indices)-index
        velocity=np.diff(xy,axis=0); acceleration=velocity[1]-velocity[0]
        if np.linalg.norm(acceleration)>.5*max(1.,np.linalg.norm(velocity,axis=1).mean()):
            return np.array([np.nan,np.nan]),float('inf')
        design=np.column_stack([times*times,times,np.ones(3)])
        model=np.linalg.lstsq(design,xy,rcond=None)[0]
        residual=float(np.max(np.linalg.norm(design@model-xy,axis=1)))
        return model[2],residual
    a,ea=extrapolate(before); b,eb=extrapolate(after)
    return dict(incoming_xy=a.tolist(),outgoing_xy=b.tolist(),fit_error_px=max(ea,eb),
                agreement_px=float(np.linalg.norm(a-b)),missing_frames=list(range(left+1,right)),
                support_frames=before+after)


def gap_contact(rows,index,pred):
    """Require both flight sides near the same confident head in a short gap."""
    motion=short_gap_motion(rows,index)
    if motion is None: return None
    a=np.array(motion['incoming_xy']); b=np.array(motion['outgoing_xy'])
    da=head_distance(a,pred,min_bbox_score=.15); db=head_distance(b,pred,min_bbox_score=.15)
    if da is None or db is None: return None
    scale=da['head_length_px']
    if motion['fit_error_px']>max(4.,.15*scale) or motion['agreement_px']>scale: return None
    if max(da['distance_head_lengths'],db['distance_head_lengths'])>.65: return None
    mid=head_distance((a+b)/2,pred,min_bbox_score=.15)
    if mid['distance_head_lengths']>.35: return None
    return dict(frame=index,ball_interpolated=False,ball_position_method='two_sided_short_gap_extrapolation',
                proposal_method=pred.get('proposal_method','detected_racket'),
                proposal_verified_by_frame=pred.get('proposal_verified_by_frame'),
                bbox_score=float(pred.get('bbox_score',0)),**motion,**mid)


def review(events,rows,racket_frames,width,height):
    decisions=[]
    for event in events:
        frame=int(event['frame']); evidence=quality(rows,frame,width,height)
        contacts=[]; gap_contacts=[]; evaluated=0; pose_frames=0; proposal_frames=0
        for i in range(max(0,frame-3),min(len(rows),frame+4)):
            record=racket_frames.get(str(i))
            if record is None or not record.get('evaluated',False): continue
            evaluated+=1
            if any(not p.get('proposal_method') for p in record.get('predictions',[])): pose_frames+=1
            if any(p.get('proposal_method') for p in record.get('predictions',[])): proposal_frames+=1
            if abs(i-frame)<=2 and not valid_ball(rows[i]):
                for pred in record.get('predictions',[]):
                    contact=gap_contact(rows,i,pred)
                    if contact is not None: gap_contacts.append(contact)
            ball,interpolated=ball_at(rows,i)
            if ball is None: continue
            for pred in record.get('predictions',[]):
                distance=head_distance(ball,pred)
                if distance is not None:
                    contacts.append(dict(frame=i,ball_interpolated=interpolated,**distance))
        all_contacts=contacts+gap_contacts
        best=min(all_contacts,key=lambda x:x['distance_head_lengths']) if all_contacts else None
        central_usable=len({c['frame'] for c in contacts if abs(c['frame']-frame)<=1})
        evidence.update(racket_evaluated_frames=evaluated,racket_detected_frames=pose_frames,racket_pose_proposal_frames=proposal_frames,
                        central_usable_pose_frames=central_usable,
                        nearest_racket_contact=best,
                        nearest_gap_racket_contact=min(gap_contacts,key=lambda c:c['distance_head_lengths']) if gap_contacts else None)
        reasons=[]; status='supported_bounce_candidate'
        jump=evidence['max_step_diagonal_fraction']>.06
        incomplete=(evidence['observed_share']<.6 or evidence['max_missing_run']>2 or
                    min(evidence['before_observations'],evidence['after_observations'])<2)
        near=best is not None and best['distance_head_lengths']<=.35
        central_contact=any(abs(c['frame']-frame)<=1 and c['distance_head_lengths']<=.35 for c in contacts)
        if central_contact and evidence['direction_change_deg'] is not None and evidence['direction_change_deg']>=60:
            # Two independent cues are required; proximity alone never vetoes a bounce.
            status='likely_racket_hit'; reasons=['racket_head_contact_and_motion_reversal']
        elif gap_contacts and evidence['direction_change_deg'] is not None and evidence['direction_change_deg']>=60:
            status='likely_racket_hit'; reasons=['short_gap_racket_contact_and_motion_reversal']
        elif jump or incomplete:
            status='uncertain_trajectory'; reasons=[r for flag,r in [(jump,'large_position_jump'),(incomplete,'insufficient_observed_trajectory')] if flag]
        elif near:
            status='uncertain_contact'; reasons=['near_racket_without_clear_motion_reversal']
        elif evaluated==0 or not contacts or central_usable==0:
            status='uncertain_racket_evidence'; reasons=['no_usable_racket_pose_near_candidate']
        else:
            reasons=['trajectory_supported','no_close_racket_contact_in_evaluated_frames']
        decisions.append(dict(**event,status=status,reasons=reasons,evidence=evidence))
    return decisions


def render_review(video,output,events,rows,racket_frames):
    cap=cv2.VideoCapture(str(video)); writer=None
    colors={'supported_bounce_candidate':(0,220,0),'likely_racket_hit':(255,100,0)}
    try:
        fps=cap.get(5); width,height=int(cap.get(3)),int(cap.get(4))
        writer=cv2.VideoWriter(str(output),cv2.VideoWriter_fourcc(*'mp4v'),fps,(width,height))
        if not cap.isOpened() or not writer.isOpened(): raise RuntimeError('Cannot open review video.')
        for i in range(len(rows)):
            ok,image=cap.read()
            if not ok: raise RuntimeError('Video shorter than CSV.')
            for pred in racket_frames.get(str(i),{}).get('predictions',[]):
                points=np.array(pred['keypoints'],float)
                for a,b in [(0,3),(3,1),(1,4),(4,0),(1,2)]:
                    cv2.line(image,tuple(points[a].astype(int)),tuple(points[b].astype(int)),(255,0,255),2)
            for event in events:
                if 0<=i-event['frame']<=int(.4*fps):
                    color=colors.get(event['status'],(0,180,255))
                    cv2.circle(image,(int(event['x']),int(event['y'])),15,color,3)
                    cv2.putText(image,f'{event["status"]} | frame {event["frame"]}',(15,65),0,.6,color,2)
            cv2.putText(image,f'Frame {i}',(15,30),0,.7,(0,255,0),2)
            writer.write(image)
    finally:
        cap.release()
        if writer is not None: writer.release()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv',type=Path,required=True)
    p.add_argument('--candidates',type=Path,required=True)
    p.add_argument('--rackets',type=Path,required=True)
    p.add_argument('--video',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--overwrite',action='store_true')
    a=p.parse_args()
    with a.csv.open(newline='') as f: rows=list(csv.DictReader(f))
    candidates=json.loads(a.candidates.read_text())
    if any(not 0<=int(e['frame'])<len(rows) for e in candidates['events']):
        raise ValueError('Candidate frame outside tracking data.')
    rackets=json.loads(a.rackets.read_text())
    cap=cv2.VideoCapture(str(a.video)); width,height=int(cap.get(3)),int(cap.get(4)); cap.release()
    if rackets['image_size']!=[width,height] or rackets['source_frames']!=len(rows): raise ValueError('Racket data and video/CSV dimensions or frame counts differ.')
    if any(int(r['Frame'])!=i for i,r in enumerate(rows)): raise ValueError('Tracking CSV frames must be consecutive.')
    reviewed=review(candidates['events'],rows,rackets['frames'],width,height)
    a.output_dir.mkdir(parents=True,exist_ok=True)
    path=a.output_dir/'bounce_review.json'; video=a.output_dir/'bounce_review.mp4'
    if not a.overwrite and (path.exists() or video.exists()): raise FileExistsError('Review outputs exist.')
    report=dict(summary=dict(supported_bounce_frames=[e['frame'] for e in reviewed if e['status']=='supported_bounce_candidate'],
                             likely_hit_frames=[e['frame'] for e in reviewed if e['status']=='likely_racket_hit'],
                             uncertain_frames=[e['frame'] for e in reviewed if e['status'].startswith('uncertain')]),
                metadata=dict(method='CatBoost candidate + observed trajectory + RacketVision racket pose',
                              candidates=str(a.candidates),tracking=str(a.csv),rackets=str(a.rackets),
                              rule_version=2,calibrated_probability=False,
                              thresholds=dict(min_observed_share=.6,max_missing_run=2,
                                              max_step_diagonal_fraction=.06,
                                              racket_contact_head_lengths=.35,
                                              hit_direction_change_deg=60,central_pose_radius_frames=1),
                              note='Rule-based evidence review, not a retrained classifier. Uncertain events are retained, not silently rejected.'),events=reviewed)
    path.write_text(json.dumps(report,indent=2))
    render_review(a.video,video,reviewed,rows,rackets['frames'])
    print(json.dumps([(e['frame'],e['status'],e['reasons'],e['evidence']['nearest_racket_contact']) for e in reviewed],indent=2))

if __name__=='__main__': main()
