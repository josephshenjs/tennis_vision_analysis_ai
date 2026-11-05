"""Fixed-camera target court geometry in original image pixels.

Four corners are ordered far-left, far-right, near-right, near-left.
Ground-plane homographies must not be used as airborne ball coordinates.
"""
import cv2
import numpy as np

WIDTH_M, LENGTH_M = 10.97, 23.77
SINGLES_WIDTH_M, SERVICE_DISTANCE_FROM_NET_M = 8.23, 6.40
ALLEY_M = (WIDTH_M-SINGLES_WIDTH_M)/2
CENTRE_X_M = WIDTH_M/2
FAR_SERVICE_Y_M = LENGTH_M/2-SERVICE_DISTANCE_FROM_NET_M
NEAR_SERVICE_Y_M = LENGTH_M/2+SERVICE_DISTANCE_FROM_NET_M
CORNERS_M = np.array([[0,0],[WIDTH_M,0],[WIDTH_M,LENGTH_M],[0,LENGTH_M]],np.float64)
# Same 14-keypoint convention as reference_RacketVision, in court metres.
KEYPOINT_NAMES = ['far_doubles_left','far_doubles_right','near_doubles_left','near_doubles_right',
                  'far_singles_left','near_singles_left','far_singles_right','near_singles_right',
                  'far_service_left','far_service_right','near_service_left','near_service_right',
                  'far_service_centre','near_service_centre']
KEYPOINTS_M = np.array([[0,0],[WIDTH_M,0],[0,LENGTH_M],[WIDTH_M,LENGTH_M],
                       [ALLEY_M,0],[ALLEY_M,LENGTH_M],
                       [WIDTH_M-ALLEY_M,0],[WIDTH_M-ALLEY_M,LENGTH_M],
                       [ALLEY_M,FAR_SERVICE_Y_M],[WIDTH_M-ALLEY_M,FAR_SERVICE_Y_M],
                       [ALLEY_M,NEAR_SERVICE_Y_M],[WIDTH_M-ALLEY_M,NEAR_SERVICE_Y_M],
                       [CENTRE_X_M,FAR_SERVICE_Y_M],[CENTRE_X_M,NEAR_SERVICE_Y_M]],np.float64)
EXTRA_POINTS_M = dict(far_baseline_centre=[CENTRE_X_M,0],
                      near_baseline_centre=[CENTRE_X_M,LENGTH_M],
                      net_ground_left=[0,LENGTH_M/2],net_ground_right=[WIDTH_M,LENGTH_M/2],
                      net_ground_centre=[CENTRE_X_M,LENGTH_M/2])
LINES = [(0,1),(0,2),(1,3),(2,3),(4,5),(6,7),(8,9),(10,11),(12,13)]


def transform(points, matrix):
    return cv2.perspectiveTransform(np.asarray(points,np.float64).reshape(-1,1,2),
                                    np.asarray(matrix,np.float64)).reshape(-1,2)


def validate_corners(corners, width, height):
    p=np.asarray(corners,np.float64)
    if p.shape!=(4,2) or not np.isfinite(p).all():
        raise ValueError('Exactly four finite corners are required.')
    if (p[:,0]<0).any() or (p[:,0]>=width).any() or (p[:,1]<0).any() or (p[:,1]>=height).any():
        raise ValueError('All corners must lie inside the original image.')
    contour=p.astype(np.float32).reshape(-1,1,2)
    if not cv2.isContourConvex(contour) or cv2.contourArea(contour)<max(100,width*height*0.001):
        raise ValueError('Corners must form a convex, non-degenerate court; check their order.')
    # Order is meaningful: never silently reorder a user's camera orientation.
    if np.mean(p[:2,1])>=np.mean(p[2:,1]) or p[0,0]>=p[1,0] or p[3,0]>=p[2,0]:
        raise ValueError('Use far-left, far-right, near-right, near-left order (end-court camera).')
    return p


def validate_roi(roi, width, height):
    x,y,w,h=map(int,roi)
    if x<0 or y<0 or w<2 or h<2 or x+w>width or y+h>height:
        raise ValueError('ROI must be x y width height, fully within the original image.')
    return [x,y,w,h]


def build_calibration(corners, width, height, camera_id, tracking_roi=None, padding=0.15):
    p=validate_corners(corners,width,height)
    if not np.isfinite(padding) or padding<0:
        raise ValueError('Padding must be non-negative.')
    h=cv2.getPerspectiveTransform(p.astype(np.float32),CORNERS_M.astype(np.float32))
    inverse=np.linalg.inv(h)
    if not np.isfinite(h).all() or not np.isfinite(inverse).all():
        raise ValueError('Invalid homography.')
    if tracking_roi is None:
        lo=p.min(axis=0); hi=p.max(axis=0); span=hi-lo
        lo=np.maximum(0,np.floor(lo-span*padding)).astype(int)
        hi=np.minimum([width,height],np.ceil(hi+span*padding)).astype(int)
        tracking_roi=[int(lo[0]),int(lo[1]),int(hi[0]-lo[0]),int(hi[1]-lo[1])]
    roi=validate_roi(tracking_roi,width,height)
    x,y,w,rh=roi
    if ((p[:,0]<x)|(p[:,0]>x+w)|(p[:,1]<y)|(p[:,1]>y+rh)).any():
        raise ValueError('Tracking ROI must contain all four court corners.')
    return dict(schema_version=1,camera_id=camera_id,image_size=dict(width=width,height=height),
                corner_order=['far_left','far_right','near_right','near_left'],
                corners_px=p.tolist(),corners_normalized=(p/[width,height]).tolist(),
                court_size_m=dict(width=WIDTH_M,length=LENGTH_M,singles_width=SINGLES_WIDTH_M,
                                  service_distance_from_net=SERVICE_DISTANCE_FROM_NET_M),
                geometry_method='four_corner_standard_court_homography',
                keypoint_names=KEYPOINT_NAMES,keypoints_court_m=KEYPOINTS_M.tolist(),
                extra_points_px={name:transform([point],inverse)[0].tolist() for name,point in EXTRA_POINTS_M.items()},
                H_image_to_court_m=h.tolist(),H_court_m_to_image=inverse.tolist(),
                keypoints_px=transform(KEYPOINTS_M,inverse).tolist(),tracking_roi_xywh=roi,
                crop_to_image_offset_xy=[x,y],
                note='Homography is for the court ground plane only. Tracking ROI is a separate image crop; no IN/OUT decisions.')


def load_calibration(path, width=None, height=None):
    import json
    from pathlib import Path
    data=json.loads(Path(path).read_text(encoding='utf8'))
    size=data['image_size']
    if width is not None and (width!=size['width'] or height!=size['height']):
        raise ValueError('Calibration image size differs from this video; recalibrate instead of silently scaling.')
    validate_corners(data['corners_px'],size['width'],size['height'])
    validate_roi(data['tracking_roi_xywh'],size['width'],size['height'])
    return data


def overlay(frame, calibration):
    out=frame.copy(); pts=np.array(calibration['keypoints_px'])
    tint=out.copy()
    cv2.fillConvexPoly(tint,np.array(calibration['corners_px'],np.int32),(60,160,60))
    out=cv2.addWeighted(out,.85,tint,.15,0)
    for a,b in LINES:
        cv2.line(out,tuple(np.round(pts[a]).astype(int)),tuple(np.round(pts[b]).astype(int)),(0,255,255),2)
    for i,point in enumerate(pts):
        if i<4: continue
        xy=tuple(np.round(point).astype(int))
        cv2.circle(out,xy,4,(0,255,255),-1)
        cv2.putText(out,f'P{i}',(xy[0]+5,xy[1]-5),0,.4,(0,255,255),1)
    # Ground-plane centre line, not the elevated top of the physical net.
    net=transform([[0,LENGTH_M/2],[WIDTH_M,LENGTH_M/2]],calibration['H_court_m_to_image'])
    cv2.line(out,tuple(net[0].astype(int)),tuple(net[1].astype(int)),(255,180,0),2)
    for i,(x,y) in enumerate(calibration['corners_px']):
        cv2.circle(out,(int(x),int(y)),7,(0,0,255),-1)
        cv2.putText(out,str(i+1),(int(x)+9,int(y)-8),0,.7,(0,0,255),2)
    x,y,w,h=calibration['tracking_roi_xywh']
    cv2.rectangle(out,(x,y),(x+w-1,y+h-1),(255,0,255),2)
    return out
