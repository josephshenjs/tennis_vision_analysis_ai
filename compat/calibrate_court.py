"""Select and calibrate one tennis court in fixed-camera video (native pixels).

Default: click four doubles-court corners (far-left, far-right, near-right,
near-left). All other ground keypoints are projected from standard court dimensions.
Press S to save. No dragging or per-point correction. --auto optionally initializes corners.
References: jack16888/RacketVision court_line_detector; CourtCheck calibrate_court.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import cv2
import numpy as np
from .court_geometry import (build_calibration, overlay, load_calibration,
                            validate_roi, transform)

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS=ROOT/'models/court/keypoints_model_trained.pth'


def read_frame(video=None, image=None, index=30):
    if image:
        frame=cv2.imread(str(image))
        if frame is None:
            raise RuntimeError(f'Cannot read image: {image}')
        return frame,0
    cap=cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise RuntimeError(f'Cannot open video: {video}')
        count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        index=min(index,count-1) if count>0 else index
        cap.set(cv2.CAP_PROP_POS_FRAMES,index)
        ok,frame=cap.read()
        if not ok:
            raise RuntimeError(f'Cannot decode frame {index}')
        return frame,index
    finally:
        cap.release()


def detect_corners(frame,roi,weights,device):
    import torch
    from torchvision.models import resnet50
    from torchvision import transforms
    if device=='auto':
        device='cuda' if torch.cuda.is_available() else 'cpu'
    if device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; use --device cpu.')
    x,y,w,h=roi
    # No ImageNet download: the checkpoint already contains the full network.
    model=resnet50(weights=None)
    model.fc=torch.nn.Linear(model.fc.in_features,28)
    state=torch.load(weights,map_location='cpu',weights_only=True)
    state=state.get('state_dict',state)
    model.load_state_dict({k.removeprefix('module.'):v for k,v in state.items()},strict=True)
    model.to(device).eval()
    preprocess=transforms.Compose([transforms.ToPILImage(),transforms.Resize((224,224)),
                                  transforms.ToTensor(),transforms.Normalize(
                                  mean=[.485,.456,.406],std=[.229,.224,.225])])
    rgb=cv2.cvtColor(frame[y:y+h,x:x+w],cv2.COLOR_BGR2RGB)
    with torch.inference_mode():
        points=model(preprocess(rgb).unsqueeze(0).to(device)).cpu().numpy().reshape(14,2)
    points=points*np.array([w/224,h/224])+np.array([x,y])
    # Reference keypoints 0,1,2,3 = FL,FR,NL,NR.
    return points[[0,1,3,2]],points


def display_scale(frame):
    return min(1.0,1280/frame.shape[1],800/frame.shape[0])


def select_roi(frame,title):
    scale=display_scale(frame)
    shown=cv2.resize(frame,None,fx=scale,fy=scale)
    roi=cv2.selectROI(title,shown,showCrosshair=True,fromCenter=False)
    cv2.destroyWindow(title)
    if roi[2]==0 or roi[3]==0:
        raise KeyboardInterrupt('ROI selection cancelled')
    x,y,w,h=np.round(np.array(roi)/scale).astype(int)
    x=max(0,x); y=max(0,y)
    return validate_roi([x,y,min(w,frame.shape[1]-x),min(h,frame.shape[0]-y)],frame.shape[1],frame.shape[0])


def select_corners(frame,initial,camera_id,tracking_roi,padding):
    """Select exactly four anchors; no dragging or per-point corrections."""
    height,width=frame.shape[:2]; scale=display_scale(frame)
    points=[list(p) for p in initial] if initial is not None else []
    title='Select 4 corners | S save | R select all again | Q cancel'
    names=['FAR LEFT','FAR RIGHT','NEAR RIGHT','NEAR LEFT']
    cv2.namedWindow(title,cv2.WINDOW_AUTOSIZE)

    def mouse(event,x,y,flags,param):
        if event==cv2.EVENT_LBUTTONDOWN and len(points)<4 and y<int(round(height*scale)):
            points.append([float(np.clip(x/scale,0,width-1)),
                           float(np.clip(y/scale,0,height-1))])
    cv2.setMouseCallback(title,mouse)
    try:
        while True:
            calibration=None; vis=frame.copy()
            if len(points)==4:
                try:
                    calibration=build_calibration(points,width,height,camera_id,tracking_roi,padding)
                    vis=overlay(frame,calibration)
                    message='Standard court points generated. S = save; R = choose all four again'
                except ValueError as error:
                    message=str(error)+' | R = restart'
            else:
                message=f'Click {len(points)+1}/4: {names[len(points)]} DOUBLES baseline corner'
            for i,point in enumerate(points):
                xy=tuple(np.round(point).astype(int))
                cv2.circle(vis,xy,7,(0,0,255),-1)
                cv2.putText(vis,str(i+1),(xy[0]+10,xy[1]-10),0,.7,(0,0,255),2)
            shown=cv2.resize(vis,None,fx=scale,fy=scale)
            # Put instructions below the image so they never hide selectable pixels.
            panel=np.full((65,shown.shape[1],3),25,np.uint8)
            cv2.putText(panel,message,(10,23),0,.47,(255,255,255),1,cv2.LINE_AA)
            cv2.putText(panel,'Order: FAR LEFT > FAR RIGHT > NEAR RIGHT > NEAR LEFT | No point editing',
                        (10,49),0,.47,(255,255,255),1,cv2.LINE_AA)
            cv2.imshow(title,np.vstack([shown,panel]))
            key=cv2.waitKey(20)&0xff
            if cv2.getWindowProperty(title,cv2.WND_PROP_VISIBLE)<1 or key in (ord('q'),ord('Q'),27):
                return None
            if key in (ord('s'),ord('S')) and calibration is not None:
                return calibration
            if key in (ord('r'),ord('R')):
                points.clear()
    finally:
        cv2.destroyWindow(title)


def export_video(source,calibration,target,max_frames=0,crop=False):
    cap=cv2.VideoCapture(str(source)); writer=None
    try:
        if not cap.isOpened():
            raise RuntimeError(f'Cannot open {source}')
        width,height=int(cap.get(3)),int(cap.get(4)); fps=cap.get(cv2.CAP_PROP_FPS)
        expected=calibration['image_size']
        if (width,height)!=(expected['width'],expected['height']):
            raise ValueError('Video resolution differs from calibration.')
        x,y,w,h=calibration['tracking_roi_xywh']
        # main() prepares an even-sized ROI before export.
        if crop and (w%2 or h%2):
            raise ValueError('Video crop width and height must be even. Adjust tracking ROI or use the saved crop image.')
        writer=cv2.VideoWriter(str(target),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h) if crop else (width,height))
        if not writer.isOpened():
            raise RuntimeError('Cannot open export encoder.')
        count=0
        while not max_frames or count<max_frames:
            ok,frame=cap.read()
            if not ok:
                break
            writer.write(frame[y:y+h,x:x+w] if crop else overlay(frame,calibration))
            count+=1
        return count
    finally:
        cap.release()
        if writer is not None:
            writer.release()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--video',type=Path)
    source.add_argument('--image',type=Path)
    parser.add_argument('--camera-id',required=True)
    parser.add_argument('--frame',type=int,default=30)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--roi',type=int,nargs=4,metavar=('X','Y','WIDTH','HEIGHT'),help='Target area for model initialization only')
    parser.add_argument('--tracking-roi',type=int,nargs=4,metavar=('X','Y','WIDTH','HEIGHT'))
    parser.add_argument('--corners',type=float,nargs=8,help='FL x/y, FR x/y, NR x/y, NL x/y in original pixels')
    parser.add_argument('--load',type=Path,help='Load an existing calibration for preview/save at the same resolution')
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--auto',action='store_true',help='Optional ResNet initialization; default is manual four-corner selection')
    mode.add_argument('--manual',action='store_true',help='Select four corners manually (default; retained for compatibility)')
    parser.add_argument('--weights',type=Path,default=DEFAULT_WEIGHTS)
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    parser.add_argument('--padding',type=float,default=.15,help='Default crop padding relative to court bounding box; inspect for high balls')
    parser.add_argument('--no-interactive',action='store_true',help='Save supplied coordinates or an explicitly unreviewed automatic proposal')
    parser.add_argument('--export-crop',action='store_true',help='Also export a native-pixel cropped MP4 (even-sized ROI required)')
    parser.add_argument('--export-preview',action='store_true',help='Also export full-frame court-overlay MP4')
    parser.add_argument('--max-frames',type=int,default=0)
    parser.add_argument('--overwrite',action='store_true')
    args=parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+',args.camera_id):
        parser.error('camera-id may contain only letters, numbers, underscore, hyphen.')
    if args.frame<0 or args.max_frames<0 or args.padding<0 or not np.isfinite(args.padding):
        parser.error('Frame, max-frames and padding must be non-negative.')
    if args.image and (args.export_crop or args.export_preview):
        parser.error('Video exports require --video.')
    if args.corners and args.load:
        parser.error('--corners and --load are mutually exclusive.')
    output=args.output or ROOT/'calibrations'/f'{args.camera_id}.json'
    targets=[output,output.with_suffix('.preview.jpg'),output.with_suffix('.crop.jpg'),
             output.with_suffix('.court_mask.png'),output.with_suffix('.topdown.jpg')]
    if args.export_crop: targets.append(output.with_suffix('.crop.mp4'))
    if args.export_preview: targets.append(output.with_suffix('.preview.mp4'))
    input_paths={p.resolve() for p in (args.video,args.image,args.weights) if p is not None}
    if any(p.resolve() in input_paths for p in targets):
        parser.error('Output paths must differ from input paths.')
    if not args.overwrite and any(p.exists() for p in targets):
        parser.error('Outputs exist; choose another --output or pass --overwrite.')
    frame,index=read_frame(args.video,args.image,args.frame)
    height,width=frame.shape[:2]
    roi=validate_roi(args.roi,width,height) if args.roi else None
    tracking_roi=validate_roi(args.tracking_roi,width,height) if args.tracking_roi else None
    initial=None; detected=None; method='manual'; weight_hash=None
    if args.load:
        saved=load_calibration(args.load,width,height)
        initial=saved['corners_px']; tracking_roi=tracking_roi or saved['tracking_roi_xywh']; method='loaded'
    elif args.corners:
        initial=np.array(args.corners).reshape(4,2); method='manual_coordinates'
    elif args.auto:
        if roi is None and not args.no_interactive:
            roi=select_roi(frame,'Select ONE target court area | Enter confirm | Esc cancel')
        roi=roi or [0,0,width,height]
        if not args.manual and args.weights.is_file():
            try:
                initial,detected=detect_corners(frame,roi,args.weights,args.device)
                method='resnet50_proposal'
                weight_hash=hashlib.sha256(args.weights.read_bytes()).hexdigest()
                print('Model initialized corners; review generated lines before saving.',flush=True)
            except (RuntimeError,ValueError,ImportError) as error:
                if args.no_interactive: raise
                print(f'Automatic initialization failed: {error}. Select four corners manually.')
        elif not args.manual:
            print('Court weights missing; using manual four-corner calibration.')
    if initial is None and not args.no_interactive:
        print('Click four DOUBLES corners: far-left, far-right, near-right, near-left. Remaining points are computed automatically.',flush=True)
    if args.no_interactive:
        if initial is None:
            parser.error('Non-interactive mode requires --corners, --load, or --auto with working model weights.')
        calibration=build_calibration(initial,width,height,args.camera_id,tracking_roi,args.padding)
        review='automatic_unreviewed' if method=='resnet50_proposal' else 'provided_coordinates_unreviewed'
    else:
        result=select_corners(frame,initial,args.camera_id,tracking_roi,args.padding)
        if result is None:
            print('Cancelled; no calibration saved.'); return
        calibration=result
        review='confirmed_interactively'
    if args.export_crop:
        x,y,w,h=calibration['tracking_roi_xywh']
        if w%2 or h%2:
            # Shrinking the crop could discard useful content; expand or shift within image.
            w+=w%2; h+=h%2
            x=min(x,width-w); y=min(y,height-h)
            if min(x,y)<0:
                parser.error('Cannot form an even-sized crop for this image.')
            calibration=build_calibration(calibration['corners_px'],width,height,args.camera_id,[x,y,w,h],args.padding)
    calibration.update(source=str((args.video or args.image).resolve()),calibration_frame=index,
                       method=method,review_status=review,created_utc=datetime.now(timezone.utc).isoformat(),
                       initialization_roi_xywh=roi,detected_keypoints_px=detected.tolist() if detected is not None else None,
                       model_sha256=weight_hash)
    output.parent.mkdir(parents=True,exist_ok=True)
    # Save previews first; only mark completion with JSON after successful exports.
    def write_image(path,image):
        if not cv2.imwrite(str(path),image): raise RuntimeError(f'Cannot write {path}')
    write_image(output.with_suffix('.preview.jpg'),overlay(frame,calibration))
    x,y,w,h=calibration['tracking_roi_xywh']
    write_image(output.with_suffix('.crop.jpg'),frame[y:y+h,x:x+w])
    mask=np.zeros((height,width),np.uint8)
    cv2.fillConvexPoly(mask,np.array(calibration['corners_px'],np.int32),255)
    write_image(output.with_suffix('.court_mask.png'),mask)
    # Rectification is for inspecting ground geometry only, not ball tracking.
    metres_to_px=np.diag([30.,30.,1.])
    topdown=cv2.warpPerspective(frame,metres_to_px@np.array(calibration['H_image_to_court_m']),(330,714))
    write_image(output.with_suffix('.topdown.jpg'),topdown)
    if args.export_crop or args.export_preview:
        if args.video is None: parser.error('Video exports require --video.')
        if args.export_crop:
            calibration['crop_video_frames']=export_video(args.video,calibration,output.with_suffix('.crop.mp4'),args.max_frames,True)
        if args.export_preview:
            calibration['preview_video_frames']=export_video(args.video,calibration,output.with_suffix('.preview.mp4'),args.max_frames)
    output.write_text(json.dumps(calibration,indent=2),encoding='utf8')
    print(f'Saved: {output} | status: {review} | crop: {calibration["tracking_roi_xywh"]}')


if __name__=='__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Cancelled; no calibration saved.')
