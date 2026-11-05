"""Target-court flight corridor and continuity-aware multi-window association.
The corridor is a ground footprint lifted vertically in the image, not a planar
homography applied to airborne balls. It is a configurable camera prior.
"""
from collections import deque
import cv2
import numpy as np
from .compat.court_geometry import transform


class CourtGate:
    def __init__(self,calibration,width,height,margin_m=1.,lift_fraction=.25):
        if not 0<=margin_m<=5 or not 0<=lift_fraction<=.75:
            raise ValueError('Court margin must be 0..5m; flight lift must be 0..0.75 image height.')
        self.width=width; self.height=height
        self.margin_m=margin_m; self.lift_fraction=lift_fraction
        # Run-off is deliberately retained; this is not an IN/OUT test.
        corners=[[-margin_m,-2],[10.97+margin_m,-2],[10.97+margin_m,25.77],[-margin_m,25.77]]
        polygon=transform(corners,calibration['H_court_m_to_image'])
        if not np.isfinite(polygon).all(): raise ValueError('Invalid calibration flight corridor.')
        ground=np.zeros((height,width),np.uint8)
        cv2.fillConvexPoly(ground,np.round(polygon).astype(np.int32),255)
        lift=int(round(height*lift_fraction))
        self.mask=cv2.dilate(ground,np.ones((lift+1,1),np.uint8),anchor=(0,0),borderType=cv2.BORDER_CONSTANT,borderValue=0)

    def allows(self,x,y):
        ix,iy=int(round(x)),int(round(y))
        return 0<=ix<self.width and 0<=iy<self.height and bool(self.mask[iy,ix])

    def model_mask(self,roi):
        x,y,w,h=roi
        return cv2.resize(self.mask[y:y+h,x:x+w],(512,288),interpolation=cv2.INTER_NEAREST)>0

    def settings(self):
        return dict(margin_m=self.margin_m,lift_fraction=self.lift_fraction,
                    allowed_image_fraction=float(np.mean(self.mask>0)),method='expanded_court_vertical_flight_corridor')


def candidate_supported(stream,index,gate,speed_limit):
    row=stream[index]
    if not int(row['Visibility']): return False
    point=np.array([float(row['X']),float(row['Y'])])
    support=0
    for j in range(max(0,index-2),min(len(stream),index+3)):
        other=stream[j]
        if j==index or not int(other['Visibility']): continue
        p=np.array([float(other['X']),float(other['Y'])])
        if gate.allows(*p) and np.linalg.norm(point-p)<=speed_limit*abs(j-index): support+=1
    return support>=2


def fuse_target(full,far,gate,fps=30.):
    """Choose by target-court support and established motion, never raw confidence alone."""
    if far is not None and len(full)!=len(far): raise ValueError('Window frame counts differ.')
    streams=[('full',full)]+([('far',far)] if far is not None else [])
    speed_limit=.06*np.hypot(gate.width,gate.height)*30/fps
    history=deque(maxlen=2); output=[]; audit=[]
    for i,base in enumerate(full):
        options=[]; spatial_rejected=0; motion_rejected=0
        for name,stream in streams:
            row=stream[i]
            if int(row['Frame'])!=i: raise ValueError('Window frames must be aligned and consecutive.')
            if abs(float(row['TimeSeconds'])-float(base['TimeSeconds']))>1e-4: raise ValueError('Window timestamps differ.')
            if not int(row['Visibility']): continue
            point=np.array([float(row['X']),float(row['Y'])])
            if not np.isfinite(point).all() or not gate.allows(*point):
                spatial_rejected+=1; continue
            if history and i-history[-1][0]<=12:
                last_index,last_point=history[-1]; dt=i-last_index
                if np.linalg.norm(point-last_point)>speed_limit*min(dt,3):
                    motion_rejected+=1; continue
                predicted=last_point
                if len(history)==2:
                    earlier,previous=history[0]
                    velocity=(last_point-previous)/(last_index-earlier)
                    predicted=last_point+velocity*min(dt,3)
                cost=float(np.linalg.norm(point-predicted))
            else:
                if not candidate_supported(stream,i,gate,speed_limit): continue
                cost=0.
            options.append((name,row,point,cost))
        if options:
            # When views agree, use far-window detail; on conflict use motion error.
            if len(options)==2 and np.linalg.norm(options[0][2]-options[1][2])<=35:
                selected=next(o for o in options if o[0]=='far'); reason='far_agrees_with_target'
            else:
                selected=min(options,key=lambda o:(o[3],0 if o[0]=='far' else 1))
                reason='motion_consistent_'+selected[0]
            name,row,point,_=selected
            if history and i-history[-1][0]>12: history.clear()
            history.append((i,point)); chosen=dict(row)
        else:
            name='none'; reason='rejected_outside_target' if spatial_rejected else ('rejected_motion_jump' if motion_rejected else 'missing_or_unconfirmed_birth')
            chosen=dict(base,Visibility=0,X=0,Y=0,Confidence=0.)
        output.append(chosen)
        audit.append(dict(Frame=i,SelectedWindow=name,Reason=reason,SpatialRejected=spatial_rejected,MotionRejected=motion_rejected))
    return output,audit
