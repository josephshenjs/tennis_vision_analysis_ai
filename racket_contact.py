"""Bounded pose proposals for short occlusions at strong trajectory turns.
Proposals are explicitly distinct from detector observations and must be
corroborated by a nearby independently detected racket before use as evidence.
"""
import numpy as np
from .video_io import frames,prefetch
from .compat.review_bounce_candidates import quality,short_gap_motion,head_distance


def recover_contact_poses(engine,video,info,rows,events,result,pose):
    from mmpose.apis import inference_topdown
    from mmengine.registry import DefaultScope
    retry_indices=set(); diagonal=np.hypot(info['width'],info['height'])
    for event in events:
        centre=int(event['frame']); evidence=quality(rows,centre,info['width'],info['height'])
        if (evidence['direction_change_deg'] or 0)<90: continue
        for index in range(max(0,centre-2),min(len(rows),centre+3)):
            motion=short_gap_motion(rows,index)
            if motion is not None and motion['agreement_px']<=.04*diagonal and motion['fit_error_px']<=8:
                retry_indices.add(index)
    retries=[]
    if not retry_indices: return retries
    with prefetch(frames(video,selected=sorted(retry_indices)),2) as images,engine.torch.inference_mode():
        for index,image in images:
            engine.check_cancelled(); motion=short_gap_motion(rows,index)
            centre=(np.array(motion['incoming_xy'])+motion['outgoing_xy'])/2
            if not np.isfinite(centre).all() or not (0<=centre[0]<info['width'] and 0<=centre[1]<info['height']): continue
            size=float(np.clip(.09*diagonal,100,240)); boxes=[]
            for dx,dy in ((-.1,-.1),(0.,0.),(.1,.1)):
                cx=centre[0]+dx*size; cy=centre[1]+dy*size
                boxes.append([max(0,cx-size/2),max(0,cy-.45*size),
                              min(info['width']-1,cx+size/2),min(info['height']-1,cy+.45*size)])
            with DefaultScope.overwrite_default_scope('mmpose'):
                poses=inference_topdown(pose,image,np.asarray(boxes,np.float32))
            kept=0
            for box,posed in zip(boxes,poses):
                points=posed.pred_instances.keypoints[0]; scores=posed.pred_instances.keypoint_scores[0]
                head=points[[0,1,3,4]]; scale=float(np.linalg.norm(points[0]-points[1]))
                if scores[[0,1,3,4]].mean()<.5 or min(scores[0],scores[1])<.45 or scale<4: continue
                corroboration=None
                for j in range(max(0,index-3),min(len(rows),index+4)):
                    if j==index: continue
                    for pred in result['frames'].get(str(j),{}).get('predictions',[]):
                        if pred.get('proposal_method'): continue
                        other=np.asarray(pred['keypoints'])
                        if head_distance(other[[0,1,3,4]].mean(0),pred) is None: continue
                        if np.linalg.norm(head.mean(0)-other[[0,1,3,4]].mean(0))<=max(3*scale,.08*diagonal):
                            corroboration=j; break
                    if corroboration is not None: break
                if corroboration is None: continue
                # No detector score is invented for a pose-only proposal.
                prediction=dict(bbox=list(map(float,box)),bbox_score=0.,keypoints=points.tolist(),
                                keypoint_scores=scores.tolist(),proposal_method='trajectory_guided_pose',
                                proposal_verified_by_frame=corroboration)
                result['frames'][str(index)]['predictions'].append(prediction); kept+=1
            retries.append(dict(frame=index,proposals=len(boxes),retained=kept))
    return retries
