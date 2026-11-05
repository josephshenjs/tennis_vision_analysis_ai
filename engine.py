"""Persistent single-GPU model engine. Invoked only on its dedicated thread."""
from collections import deque
import hashlib
from pathlib import Path
import sys
import time
import cv2
import numpy as np
from .paths import BALL_SOURCE,BALL_CHECKPOINT,RACKET_SOURCE
from .video_io import frames,prefetch
from .compat.review_bounce_candidates import ball_at


class Engine:
    def check_cancelled(self):
        if getattr(self,'stop',None) is not None and self.stop.is_set():
            raise InterruptedError('Analysis cancelled.')

    def __init__(self,device='cuda:0',amp=False,prefetch_batches=2):
        import torch
        cv2.setNumThreads(1)
        self.torch=torch
        if not device.startswith('cuda') or not torch.cuda.is_available():
            raise RuntimeError('CUDA is required. Use the configured .venv-racket interpreter and an NVIDIA GPU.')
        self.device=torch.device(device if ':' in device else 'cuda:0')
        torch.cuda.set_device(self.device)
        torch.set_num_threads(min(4,torch.get_num_threads()))
        torch.backends.cudnn.benchmark=True
        self.amp=amp; self.capacity=prefetch_batches
        self.ball=None; self.det=None; self.pose=None; self.hashes={}; self.load_counts={'ball':0,'racket':0}
        print(f'CUDA: {torch.cuda.get_device_name(self.device)} | torch {torch.__version__} | AMP {amp}',flush=True)

    def ball_model(self):
        self.check_cancelled()
        if self.ball is None:
            self.load_counts['ball']+=1
            sys.path.insert(0,str(BALL_SOURCE))
            from model.tracknet_v3 import TrackNet
            self.ball=TrackNet(in_dim=15,out_dim=4,last_only=True)
            state=self.torch.load(BALL_CHECKPOINT,map_location='cpu',weights_only=False)
            state=state.get('state_dict',state)
            self.ball.load_state_dict({k.removeprefix('module.'):v for k,v in state.items()},strict=True)
            self.ball.to(self.device).eval()
        return self.ball

    def track(self,video,info,rois,batch_size=4,threshold=.5,median_samples=100,max_frames=0,court_gate=None,selected_indices=None):
        start=time.perf_counter(); torch=self.torch; model=self.ball_model()
        count=min(info['declared_frames'],max_frames) if max_frames and info['declared_frames']>0 else (max_frames or info['declared_frames'])
        if count<=0: raise ValueError('A known video frame count is required for median sampling.')
        sample_indices=np.unique(np.linspace(0,count-1,min(count,median_samples),dtype=int))
        samples=[[] for _ in rois]
        for _,image in frames(video,limit=count,selected=sample_indices):
            self.check_cancelled()
            for view,(x,y,w,h) in enumerate(rois):
                samples[view].append(cv2.resize(image[y:y+h,x:x+w],(512,288)))
        if not samples[0]: raise RuntimeError('No background frames decoded.')
        backgrounds=[np.moveaxis(np.median(np.stack(s),axis=0).astype(np.uint8).astype(np.float32)/255,-1,0) for s in samples]
        del samples
        def batches():
            histories=[deque(maxlen=4) for _ in rois]; data=[]; indices=[]
            for index,image in frames(video,limit=max_frames,selected=selected_indices):
                self.check_cancelled()
                indices.append(index)
                for v,(x,y,w,h) in enumerate(rois):
                    resized=np.moveaxis(cv2.resize(image[y:y+h,x:x+w],(512,288)).astype(np.float32)/255,-1,0)
                    if not histories[v]: histories[v].extend([resized]*3)
                    histories[v].append(resized)
                    data.append(np.concatenate([backgrounds[v],*histories[v]],axis=0))
                if len(indices)==batch_size:
                    yield indices,np.stack(data); data=[]; indices=[]
            if indices: yield indices,np.stack(data)
        output=[[] for _ in rois]
        model_masks=[court_gate.model_mask(roi) for roi in rois] if court_gate is not None else [None]*len(rois)
        from .compat.test_video import model_to_image
        with prefetch(batches(),self.capacity) as batches_ready,torch.inference_mode():
            for indices,batch in batches_ready:
                if getattr(self,'stop',None) is not None and self.stop.is_set(): raise InterruptedError('Analysis cancelled.')
                tensor=torch.from_numpy(batch).pin_memory().to(self.device,non_blocking=True)
                with torch.autocast(device_type='cuda',enabled=self.amp):
                    predictions=model(frames=tensor)[0][:,0].float().cpu().numpy()
                for j,index in enumerate(indices):
                    for v,roi in enumerate(rois):
                        heatmap=predictions[j*len(rois)+v]
                        if model_masks[v] is not None:
                            heatmap=np.where(model_masks[v],heatmap,0.)
                        contours,_=cv2.findContours((heatmap>threshold).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                        x=y=0; confidence=0.
                        if contours:
                            bx,by,bw,bh=max((cv2.boundingRect(c) for c in contours),key=lambda r:r[2]*r[3])
                            x,y=model_to_image(bx+bw/2,by+bh/2,roi)
                            confidence=float(heatmap[by:by+bh,bx:bx+bw].mean())
                        output[v].append(dict(Frame=index,TimeSeconds=f'{index/info["fps"]:.6f}',Visibility=int(bool(contours)),X=x,Y=y,Confidence=round(confidence,6)))
                if len(output[0])%100<batch_size: print(f'{Path(video).name}: tracked {len(output[0])}',flush=True)
        if not output[0]: raise RuntimeError('No video frames decoded.')
        return output,dict(seconds=time.perf_counter()-start,device=str(self.device),amp=self.amp,
                           frames=len(output[0]),rois=rois,input_size=[512,288],batch_frames=batch_size,
                           model_inputs_per_batch=batch_size*len(rois),prefetch_batches=self.capacity,
                           checkpoint=str(BALL_CHECKPOINT),court_gate=court_gate.settings() if court_gate is not None else None)

    def racket_models(self):
        self.check_cancelled()
        if self.det is None:
            print('Loading racket models (first use; imports and CUDA initialization may take time)...',flush=True)
            self.load_counts['racket']+=1
            from mmdet.apis import init_detector
            from mmpose.apis import init_model
            det_path=RACKET_SOURCE/'checkpoints/epoch_300.pth'
            pose_path=RACKET_SOURCE/'checkpoints/best_PCK_epoch_90.pth'
            self.check_cancelled()
            print('Initializing racket detector...',flush=True)
            self.det=init_detector(str(RACKET_SOURCE/'configs/detection/rtmdet_m_racket_infer.py'),str(det_path),device=str(self.device))
            self.check_cancelled()
            print('Initializing racket pose model...',flush=True)
            self.pose=init_model(str(RACKET_SOURCE/'configs/pose/rtmpose_m_racket_infer.py'),str(pose_path),device=str(self.device))
            self.check_cancelled()
            self.det.eval(); self.pose.eval()
            self.hashes={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in [('det',det_path),('pose',pose_path)]}
            print('Racket models ready.',flush=True)
        return self.det,self.pose

    def recover_gaps(self,video,info,rows,gate,batch_size=4,threshold=.5,median_samples=40):
        from .gap_recovery import plan_recovery,merge_recovery
        start=time.perf_counter(); result=[dict(r) for r in rows]; attempts=[]
        plans=plan_recovery(rows,info['width'],info['height'],info['fps'])
        for plan in plans:
            self.check_cancelled()
            print(f'{Path(video).name}: local recovery frames {plan["start"]}..{plan["end"]-1}',flush=True)
            tracked,_=self.track(video,info,[plan['roi']],batch_size,threshold,median_samples,
                                 len(rows),gate,plan['selected'])
            recovered,reason=merge_recovery(result,tracked[0],plan,gate,info['fps'])
            attempts.append(dict(start_frame=plan['start'],end_frame=plan['end']-1,
                                 roi_xywh=plan['roi'],recovered_frames=recovered,reason=reason))
        return result,dict(seconds=time.perf_counter()-start,attempts=attempts,
                           recovered_frames=sorted({i for a in attempts for i in a['recovered_frames']}),
                           method='local_tracknet_verified_observations',max_attempts=4)

    def rackets(self,video,info,rows,events,radius=6,context_size=600,bbox_threshold=.3):
        start=time.perf_counter()
        result=dict(source=str(Path(video).resolve()),source_frames=len(rows),image_size=[info['width'],info['height']],
                    fps=info['fps'],model='RacketVision RTMDet-M + RTMPose-M',sport='tennis',category_id=2,
                    bbox_threshold=bbox_threshold,radius_frames=radius,context_size=context_size,frames={})
        if not events:
            result['seconds']=time.perf_counter()-start
            return result
        det,pose=self.racket_models()
        from mmdet.apis import inference_detector
        from mmpose.apis import inference_topdown
        from mmengine.registry import DefaultScope
        selected=sorted({i for e in events for i in range(max(0,int(e['frame'])-radius),min(len(rows),int(e['frame'])+radius+1))})
        result.update(self.hashes)
        with prefetch(frames(video,selected=selected),2) as images,self.torch.inference_mode():
            for count,(index,frame) in enumerate(images):
                if getattr(self,'stop',None) is not None and self.stop.is_set(): raise InterruptedError('Analysis cancelled.')
                ball,_=ball_at(rows,index)
                height,width=frame.shape[:2]
                if ball is None: x=y=0; w=width; h=height
                else:
                    w=min(context_size,width); h=min(context_size,height)
                    x=int(np.clip(ball[0]-w/2,0,width-w)); y=int(np.clip(ball[1]-h/2,0,height-h))
                image=frame[y:y+h,x:x+w]
                with DefaultScope.overwrite_default_scope('mmdet'):
                    instances=inference_detector(det,image).pred_instances
                self.check_cancelled()
                boxes=instances.bboxes.cpu().numpy(); scores=instances.scores.cpu().numpy(); labels=instances.labels.cpu().numpy()
                keep=(labels==2)&(scores>=bbox_threshold); boxes=boxes[keep]; scores=scores[keep]
                area=np.array([0<(b[2]-b[0])*(b[3]-b[1])<w*h*.5 for b in boxes],bool)
                boxes=boxes[area][:4]; scores=scores[area][:4]; predictions=[]
                if len(boxes):
                    with DefaultScope.overwrite_default_scope('mmpose'):
                        poses=inference_topdown(pose,image,boxes)
                    for box,score,pose_result in zip(boxes,scores,poses):
                        keypoints=pose_result.pred_instances.keypoints[0].copy()+np.array([x,y])
                        predictions.append(dict(bbox=(box+np.array([x,y,x,y])).tolist(),bbox_score=float(score),
                                                keypoints=keypoints.tolist(),keypoint_scores=pose_result.pred_instances.keypoint_scores[0].tolist()))
                result['frames'][str(index)]=dict(evaluated=True,roi_xywh=[x,y,w,h],predictions=predictions)
                if count%20==0: print(f'{Path(video).name}: racket evidence {count+1}/{len(selected)}',flush=True)
        from .racket_contact import recover_contact_poses
        result['contact_pose_retry']=recover_contact_poses(self,video,info,rows,events,result,pose)
        if len(result['frames'])!=len(selected): raise RuntimeError('Racket evidence video ended before required frames.')
        result['seconds']=time.perf_counter()-start
        return result
