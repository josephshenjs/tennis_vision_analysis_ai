"""Monocular ballistic speed estimates anchored at observed bounce candidates.
Inspired by court calibration / RANSAC in doi:10.1038/s41598-025-06365-3.
This is an additional monocular model, not the paper's multi-camera method.
"""
import numpy as np
from .compat.court_geometry import transform

GRAVITY = 9.81


def camera_projection(calibration):
    """Recover vertical direction using square pixels, zero skew, centred principal point.
    An optional camera_matrix (3x3 intrinsics) avoids the focal self-calibration.
    Ground-plane homography alone is never applied to airborne observations.
    """
    h = np.asarray(calibration['H_court_m_to_image'], float)
    size = calibration['image_size']; cx=size['width']/2; cy=size['height']/2
    if 'camera_matrix' in calibration:
        k=np.asarray(calibration['camera_matrix'],float)
    else:
        a=h[:,0]; b=h[:,1]
        ax,ay=a[:2]-[cx*a[2],cy*a[2]]
        bx,by=b[:2]-[cx*b[2],cy*b[2]]
        coefficients=np.array([a[2]*b[2], a[2]**2-b[2]**2])
        values=-np.array([ax*bx+ay*by, ax*ax+ay*ay-bx*bx-by*by])
        denominator=coefficients@coefficients
        if denominator<1e-16: raise ValueError('focal_length_unobservable')
        f2=float(coefficients@values/denominator)
        if not np.isfinite(f2) or f2<=0: raise ValueError('invalid_focal_calibration')
        focal=np.sqrt(f2)
        if not .2*size['width']<focal<10*size['width']: raise ValueError('implausible_focal_length')
        k=np.array([[focal,0,cx],[0,focal,cy],[0,0,1.]])
    r=np.linalg.solve(k,h[:,:2]); norms=np.linalg.norm(r,axis=0)
    if abs(np.dot(r[:,0],r[:,1])/np.prod(norms))>.15 or max(norms)/min(norms)>1.2:
        raise ValueError('inconsistent_camera_calibration')
    vertical=np.cross(r[:,0],r[:,1]); vertical*=np.mean(norms)/np.linalg.norm(vertical)
    p=np.column_stack([h[:,:2],k@vertical,h[:,2]])
    centre=-np.linalg.solve(p[:,:3],p[:,3])
    if centre[2]<0: p[:,2]*=-1
    return p


def fit_arc(p,t,xy):
    # X(t) = [xb+vx*t, yb+vy*t, vz*t-g*t²/2]; t=0 at bounce.
    n=len(t); basis=np.zeros((n,3,5))
    basis[:,0,0]=1; basis[:,1,1]=1
    basis[:,0,2]=t; basis[:,1,3]=t; basis[:,2,4]=t
    projected=np.einsum('ij,njk->nik',p[:,:3],basis)
    constant=p[:,3][None,:]-.5*GRAVITY*t[:,None]**2*p[:,2][None,:]
    a=(projected[:,:2,:]-xy[:,:,None]*projected[:,2:3,:]).reshape(-1,5)
    b=(xy*constant[:,2:3]-constant[:,:2]).reshape(-1)
    solution,_,rank,_=np.linalg.lstsq(a,b,rcond=None)
    if rank<5: raise ValueError('degenerate_trajectory')
    return solution


def project_arc(p,t,s):
    xyz=np.column_stack([s[0]+s[2]*t,s[1]+s[3]*t,s[4]*t-.5*GRAVITY*t*t])
    q=xyz@p[:,:3].T+p[:,3]
    with np.errstate(divide='ignore',invalid='ignore'): pixels=q[:,:2]/q[:,2:3]
    return pixels,xyz,q[:,2]



def motion_proxy(rows,low,bounce,calibration,fps):
    """Low-confidence court-plane displacement proxy, explicitly not 3D reconstruction.
    Used only when no valid ballistic model can be recovered, never labelled supported.
    """
    indices=[i for i in range(max(low,bounce-round(.45*fps)),bounce+1)
             if int(rows[i]['Visibility']) and np.isfinite([float(rows[i]['X']),float(rows[i]['Y'])]).all()]
    if len(indices)<2: return None
    xy=np.array([[float(rows[i]['X']),float(rows[i]['Y'])] for i in indices])
    points=transform(xy,calibration['H_image_to_court_m'])
    velocities=[]
    for j in range(1,len(indices)):
        dt=(indices[j]-indices[j-1])/fps
        if dt>3/fps: continue
        velocity=(points[j]-points[j-1])/dt
        speed=float(np.linalg.norm(velocity)*3.6)
        if np.isfinite(speed) and 1<=speed<=300: velocities.append(speed)
    if not velocities: return None
    speed=float(np.median(velocities))
    return dict(method='court_plane_motion_proxy',support_level='low',speed_kmh=speed,
                speed_mph=speed/1.609344,speed_frame=indices[0],end_frame=bounce,
                observation_count=len(indices),selection_cost=100.,
                fitted_landing_court_m=transform([[float(rows[bounce]['X']),float(rows[bounce]['Y'])]],
                                                calibration['H_image_to_court_m'])[0].tolist())


def estimate_speeds(rows,events,calibration,fps):
    if not np.isfinite(fps) or fps<=0: raise ValueError('Invalid video FPS')
    try: p=camera_projection(calibration); camera_error=None
    except (ValueError,KeyError,np.linalg.LinAlgError) as error: p=None; camera_error=str(error)
    ordered=sorted(events,key=lambda e:int(e['frame'])); estimates=[]
    for event in ordered:
        if event.get('status')=='likely_racket_hit': continue
        bounce=int(event['frame'])
        if not 0<=bounce<len(rows): raise ValueError('Bounce outside tracking frames')
        # Six-frame review lookahead is already consumed by the evidence pipeline.
        available=bounce+6
        result=dict(bounce_frame=bounce,landing_available_frame=bounce+3,available_frame=available,status='unavailable',
                    reason=camera_error or 'insufficient_trajectory',best_prediction=None,
                    event_status=event.get('status'),candidates=[])
        estimates.append(result)
        result['confidence']='low' if event.get('status')!='supported_bounce_candidate' else 'supported'
        ground=transform([[event['x'],event['y']]],calibration['H_image_to_court_m'])[0]
        if not np.isfinite(ground).all(): result['reason']='invalid_landing'; continue
        result['landing_court_m']=ground.tolist()
        previous=max((int(e['frame']) for e in ordered if int(e['frame'])<bounce),default=-1)
        low=max(previous+1,bounce-int(.9*fps),0)
        indices=[]; missing=0
        # Keep real observations across at most two missing frames; never interpolate.
        # Do not cross a long gap or a previous impact with one flight model.
        for i in range(bounce,low-1,-1):
            row=rows[i]
            if not int(row['Visibility']) or not np.isfinite([float(row['X']),float(row['Y'])]).all():
                missing+=1
                if missing>2: break
                continue
            missing=0
            indices.append(i)
        indices=sorted(indices)
        for duration in (.25,.4,.6,.9):
            selected=[i for i in indices if i>=bounce-round(duration*fps)]
            if p is None or len(selected)<3 or (selected[-1]-selected[0])/fps<.06: continue
            t=(np.array(selected)-bounce)/fps
            xy=np.array([[float(rows[i]['X']),float(rows[i]['Y'])] for i in selected])
            threshold=max(2.,calibration['image_size']['width']*.0025)
            rng=np.random.default_rng(0); best=None
            subsets=[np.arange(len(t))]+([rng.choice(len(t),5,replace=False) for _ in range(64)] if len(t)>=5 else [])
            for subset in subsets:
                try: s=fit_arc(p,t[subset],xy[subset])
                except (ValueError,np.linalg.LinAlgError): continue
                predicted,_,_=project_arc(p,t,s)
                errors=np.linalg.norm(predicted-xy,axis=1); mask=np.isfinite(errors)&(errors<=threshold)
                key=(int(mask.sum()),-float(np.median(errors[mask])) if mask.any() else -np.inf)
                if best is None or key>best[0]: best=(key,mask)
            if best is None or best[1].sum()<3: continue
            mask=best[1]
            try: s=fit_arc(p,t[mask],xy[mask])
            except (ValueError,np.linalg.LinAlgError): continue
            predicted,xyz,depth=project_arc(p,t,s)
            error=float(np.sqrt(np.mean(np.sum((predicted[mask]-xy[mask])**2,axis=1))))
            landing_error=float(np.linalg.norm(s[:2]-ground))
            velocity=np.array([s[2],s[3],s[4]-GRAVITY*t[0]])
            speed=float(np.linalg.norm(velocity)*3.6)
            if (not np.isfinite(speed) or not 5<=speed<=300 or s[4]>=0 or
                xyz[:,2].min()<-.05 or xyz[:,2].max()>12 or np.any(depth*depth[-1]<=0) or
                landing_error>5): continue
            strong=(mask.sum()>=7 and mask.mean()>=.75 and error<=threshold and landing_error<=1.5 and -t[0]>=.12)
            candidate=dict(method='ballistic_ransac',support_level='supported' if strong else 'low',speed_kmh=speed,speed_mph=speed/1.609344,
                speed_frame=selected[0],end_frame=bounce,fit_rmse_px=error,
                landing_error_m=landing_error,inlier_share=float(mask.mean()),
                observation_count=len(selected),inlier_count=int(mask.sum()),
                fitted_landing_court_m=s[:2].tolist(),velocity_at_bounce_mps=s[2:].tolist(),
                selection_cost=(0 if strong else 10)+error/threshold+landing_error+2*(1-mask.mean())+.1/(-t[0]))
            result['candidates'].append(candidate)
        if not result['candidates']:
            proxy=motion_proxy(rows,low,bounce,calibration,fps)
            if proxy is not None: result['candidates'].append(proxy)
        if result['candidates']:
            result['best_prediction']=min(result['candidates'],key=lambda c:c['selection_cost'])
            best_prediction=result['best_prediction']
            weak=result['confidence']=='low' or best_prediction['support_level']=='low'
            result.update(status='estimated_low_confidence' if weak else 'estimated',
                          confidence='low' if weak else 'supported',
                          reason='best_available_estimate' if weak else None)
            # Select by model support and reprojection quality, never largest speed.
            if result['best_prediction']['method']=='ballistic_ransac':
                result['landing_court_m']=result['best_prediction']['fitted_landing_court_m']
        else:
            result['reason']=camera_error or 'no_measurable_trajectory'
            prior=next((e for e in reversed(estimates[:-1]) if e.get('best_prediction')),None)
            if prior is not None:
                result['best_prediction']=dict(prior['best_prediction'],method='previous_speed_prior',
                    support_level='low',selection_cost=200.,prior_bounce_frame=prior['bounce_frame'])
                result['candidates'].append(result['best_prediction'])
                result.update(status='estimated_low_confidence',confidence='low',reason='previous_speed_prior')
    return dict(schema_version=2,method='monocular_bounce_anchored_ballistic_ransac',
        reference='https://doi.org/10.1038/s41598-025-06365-3',
        speed_definition='Best available pre-bounce speed; method identifies 3D ballistic, ground-plane motion proxy or previous-speed prior; not guaranteed racket-contact speed',
        assumptions=['fixed camera','metric court calibration','square pixels and centred principal point unless camera_matrix supplied',
                     'gravity-only flight (no drag/spin)','reviewed bounce is on ground'],
        calibrated_accuracy=False,events=estimates)
