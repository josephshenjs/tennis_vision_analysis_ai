"""Asynchronous clip scheduler with one CUDA owner and bounded CPU workers."""
import asyncio
from collections import Counter,deque
from concurrent.futures import ThreadPoolExecutor
import csv
from functools import partial
import hashlib
import json
from pathlib import Path
import threading
import time
import traceback
import cv2
import numpy as np
from .compat.bounce_detection import analyze
from .compat.court_geometry import load_calibration,validate_roi,transform
from .compat.review_bounce_candidates import review
from .video_io import metadata,frames,prefetch
from .court_tracking import CourtGate,fuse_target
from .ball_speed import estimate_speeds
from .shot_overlay import draw_shot_overlay


def write_json(path,value):
    path=Path(path); temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2),encoding='utf8'); temporary.replace(path)


def write_csv(path,rows):
    with Path(path).open('w',newline='') as f:
        if rows:
            writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def far_window(calibration,width,height,explicit=None):
    if explicit is not None: return validate_roi(explicit,width,height)
    if calibration.get('far_tracking_roi_xywh') is not None:
        return validate_roi(calibration['far_tracking_roi_xywh'],width,height)
    # Geometry gives a starting view, not a guarantee that all airborne balls fit.
    court=np.array(calibration['corners_px'])[:2]
    net=transform([[0,11.885],[10.97,11.885]],calibration['H_court_m_to_image'])
    points=np.vstack([court,net]); lo=points.min(0); hi=points.max(0)
    span=hi-lo
    lo=np.maximum(0,np.floor(lo-np.array([.12*span[0],max(160.,2*span[1])]))).astype(int)
    hi=np.minimum([width,height],np.ceil(hi+[.12*span[0],max(50.,.6*span[1])])).astype(int)
    return validate_roi([*lo,*(hi-lo)],width,height)


def render(video,path,rows,events,rackets,stop,speed_report=None):
    from .web_video import VideoWriter
    info=metadata(video); writer=VideoWriter(video,path,info)
    if not writer.isOpened(): raise RuntimeError('Cannot open output video encoder.')
    trail=deque(maxlen=12); count=0
    try:
        with prefetch(frames(video,limit=len(rows)),2) as images:
            for index,image in images:
                if stop.is_set(): raise InterruptedError('Analysis cancelled.')
                row=rows[index]
                if int(row['Visibility']):
                    point=(int(row['X']),int(row['Y'])); trail.append(point)
                    for a,b in zip(list(trail),list(trail)[1:]): cv2.line(image,a,b,(0,200,255),2)
                    cv2.circle(image,point,7,(0,255,0),2)
                else: trail.clear()
                draw_shot_overlay(image,speed_report or {'events':[]},index)
                writer.write(image); count+=1
        if count!=len(rows): raise RuntimeError('Output video ended before all prediction frames.')
    finally: writer.release()
    return count


def prepare_jobs(videos,output,calibration_path,overwrite=False):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    counts=Counter(p.stem.casefold() for p in videos); jobs=[]
    inputs={Path(p).resolve() for p in videos}|{Path(calibration_path).resolve()}
    names=['analysis.mp4','predictions.csv','full_predictions.csv','far_predictions.csv','fusion_audit.csv',
           'candidates.json','bounce_scores.csv','rackets.json','events.json','run.json','status.json','court_region.png','recovery.json','speeds.json']
    for video in videos:
        name=video.stem
        if counts[name.casefold()]>1: name+='_'+hashlib.sha256(str(video.resolve()).encode()).hexdigest()[:8]
        folder=output/name
        targets=[folder/n for n in names]
        if any(p.resolve() in inputs for p in targets): raise ValueError('Output would overwrite an input file.')
        if not overwrite and any(p.exists() for p in targets): raise FileExistsError(f'Analysis exists: {folder}; use --overwrite or another output directory.')
        jobs.append((video,folder))
    return jobs


async def analyze_videos(args, resources=None):
    from .engine import Engine
    loop=asyncio.get_running_loop()
    gpu=resources['gpu'] if resources else ThreadPoolExecutor(max_workers=1,thread_name_prefix='rally-cuda')
    cpu=resources['cpu'] if resources else ThreadPoolExecutor(max_workers=args.cpu_workers,thread_name_prefix='rally-cpu')
    semaphore=asyncio.Semaphore(args.jobs); stop=resources['stop'] if resources else threading.Event()
    started=time.perf_counter(); engine=None
    jobs=prepare_jobs(args.videos,args.output_dir,args.court_calibration,args.overwrite)
    async def on_gpu(function,*items,**kwargs):
        return await loop.run_in_executor(gpu,partial(function,*items,**kwargs))
    async def on_cpu(function,*items,**kwargs):
        return await loop.run_in_executor(cpu,partial(function,*items,**kwargs))
    try:
        engine=resources.get('engine') if resources else None
        if engine is None:
            engine=await on_gpu(Engine,args.device,args.amp,args.prefetch)
            if resources is not None: resources['engine']=engine
        engine.stop=stop
        async def job(video,folder):
            async with semaphore:
                folder.mkdir(parents=True,exist_ok=True); begin=time.perf_counter(); stages=[]
                def stage(name):
                    stages.append(dict(name=name,seconds_from_start=time.perf_counter()-begin))
                    write_json(folder/'status.json',dict(video=str(video),stage=name,state='running'))
                    print(f'[{video.name}] {name}',flush=True)
                try:
                    stage('tracking')
                    info=await on_cpu(metadata,video)
                    calibration=await on_cpu(load_calibration,args.court_calibration,info['width'],info['height'])
                    rois=[calibration['tracking_roi_xywh']]
                    if not args.single_window:
                        rois.append(far_window(calibration,info['width'],info['height'],args.far_roi))
                    gate=CourtGate(calibration,info['width'],info['height'],getattr(args,'court_margin_m',1.),getattr(args,'ball_lift',.25))
                    if not cv2.imwrite(str(folder/'court_region.png'),gate.mask): raise RuntimeError('Cannot save court region.')
                    tracked,timing=await on_gpu(engine.track,video,info,rois,args.batch_size,args.threshold,args.median_samples,args.max_frames,gate)
                    stage('fusion_and_candidates')
                    rows,audit=await on_cpu(fuse_target,tracked[0],tracked[1] if len(tracked)==2 else None,gate,info['fps'])
                    recovery=dict(seconds=0,attempts=[],recovered_frames=[],enabled=False)
                    if not getattr(args,'no_local_recovery',False):
                        stage('local_recovery')
                        rows,recovery=await on_gpu(engine.recover_gaps,video,info,rows,gate,args.batch_size,args.threshold)
                        recovery['enabled']=True
                        for i in recovery['recovered_frames']:
                            audit[i].update(SelectedWindow='local',Reason='verified_local_recovery')
                    events,scores,model_meta=await on_cpu(analyze,rows,info['fps'],info['width'],info['height'],
                                                       threshold=args.bounce_threshold,model_path=args.bounce_model)
                    stage('racket_evidence')
                    rackets=await on_gpu(engine.rackets,video,info,rows,events)
                    reviewed=await on_cpu(review,events,rows,rackets['frames'],info['width'],info['height'])
                    stage('ball_speed')
                    speed_report=await on_cpu(estimate_speeds,rows,reviewed,calibration,info['fps'])
                    summary=dict(display_bounce_frames=[e['bounce_frame'] for e in speed_report['events'] if 'landing_court_m' in e],
                                 supported_bounce_frames=[e['frame'] for e in reviewed if e['status']=='supported_bounce_candidate'],
                                 likely_hit_frames=[e['frame'] for e in reviewed if e['status']=='likely_racket_hit'],
                                 uncertain_frames=[e['frame'] for e in reviewed if e['status'].startswith('uncertain')])
                    def save():
                        write_csv(folder/'predictions.csv',rows); write_csv(folder/'full_predictions.csv',tracked[0])
                        if len(tracked)==2:
                            write_csv(folder/'far_predictions.csv',tracked[1])
                        write_csv(folder/'fusion_audit.csv',audit)
                        write_csv(folder/'bounce_scores.csv',scores)
                        write_json(folder/'candidates.json',dict(metadata=model_meta,events=events))
                        write_json(folder/'rackets.json',rackets)
                        write_json(folder/'recovery.json',recovery)
                        write_json(folder/'speeds.json',speed_report)
                        write_json(folder/'events.json',dict(summary=summary,events=reviewed,
                                   note='Evidence-based rules, not confirmed ground truth or calibrated probability.'))
                    await on_cpu(save)
                    stage('rendering' if not args.no_video else 'saving')
                    if not args.no_video: await on_cpu(render,video,folder/'analysis.mp4',rows,reviewed,rackets,stop,speed_report)
                    result=dict(video=str(video),output=str(folder),state='complete',frames=len(rows),
                                detected_frames=sum(int(r['Visibility']) for r in rows),summary=summary,
                                seconds=time.perf_counter()-begin,tracking=timing,racket_seconds=rackets['seconds'],
                                speed_estimates=sum(e.get('best_prediction') is not None for e in speed_report['events']),
                                calibration=str(args.court_calibration),coordinate_space='original_video_pixels',stages=stages,recovery=recovery)
                    write_json(folder/'run.json',result); write_json(folder/'status.json',dict(state='complete',video=str(video)))
                    print(f'[{video.name}] complete: {len(rows)} frames in {result["seconds"]:.1f}s',flush=True)
                    return result
                except asyncio.CancelledError:
                    stop.set()
                    write_json(folder/'status.json',dict(video=str(video),state='cancelled',
                               stage=stages[-1]['name'] if stages else 'queued'))
                    raise
                except Exception as error:
                    result=dict(video=str(video),output=str(folder),state='failed',error=str(error))
                    write_json(folder/'status.json',result)
                    (folder/'error.log').write_text(traceback.format_exc(),encoding='utf8')
                    print(f'[{video.name}] FAILED: {error}',flush=True)
                    return result
        results=await asyncio.gather(*(job(*item) for item in jobs))
        summary=dict(results=results,seconds=time.perf_counter()-started,
                     cuda_device=args.device,max_inflight=args.jobs,cpu_workers=args.cpu_workers,
                     single_gpu_owner=True,queue_capacity_batches=args.prefetch,model_load_counts=engine.load_counts)
        write_json(args.output_dir/'summary.json',summary)
        return 1 if any(r['state']=='failed' for r in results) else 0
    finally:
        if resources is None:
            stop.set()
            gpu.shutdown(wait=True,cancel_futures=True); cpu.shutdown(wait=True,cancel_futures=True)
