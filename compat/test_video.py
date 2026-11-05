"""Stream an MP4 through RacketVision BallTrack; write annotated MP4 + CSV."""
import argparse
import csv
import json
import sys
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('video', type=Path)
    p.add_argument('--checkpoint', type=Path, default=ROOT / 'RacketVision/source/BallTrack/checkpoints/balltrack_best.pth')
    p.add_argument('--output-dir', type=Path, default=ROOT / 'outputs')
    p.add_argument('--court-calibration', type=Path, help='Court JSON: crop tracking ROI for inference; retain original output coordinates')
    p.add_argument('--device', default='cuda', help='cuda (default), cuda:0, cpu, or auto')
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--median-samples', type=int, default=100)
    p.add_argument('--median', type=Path, help='Optional background image or NPZ with median key')
    p.add_argument('--max-frames', type=int, default=0, help='0 processes the whole video')
    p.add_argument('--trail', type=int, default=12)
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--racket-evidence', type=Path, help='Optional cached RacketVision pose JSON for evidence-based bounce review')
    p.add_argument('--skip-bounce', action='store_true', help='Run ball tracking only')
    p.add_argument('--bounce-model', type=Path, default=ROOT/'models/bounce/ctb_regr_bounce.cbm')
    p.add_argument('--bounce-threshold', type=float, default=0.55)
    p.add_argument('--bounce-max-gap', type=int, default=2)
    p.add_argument('--bounce-min-separation', type=float, default=0.12)
    a = p.parse_args()
    if not 0 < a.threshold < 1 or min(a.batch_size, a.median_samples) < 1 or min(a.max_frames, a.trail) < 0:
        p.error('Invalid threshold, batch size, sample count, frame limit, or trail length')
    if not 0 < a.bounce_threshold < 1 or a.bounce_max_gap < 0 or a.bounce_min_separation < 0:
        p.error('Invalid bounce threshold, gap limit, or event separation')
    return a

def crop_frame(frame, roi):
    x,y,width,height=roi
    return frame[y:y+height,x:x+width]


def model_to_image(x, y, roi):
    left,top,width,height=roi
    return (left+min(width-1,max(0,int(x*width/512))),
            top+min(height-1,max(0,int(y*height/288))))


def main():
    args = arguments()
    if not args.skip_bounce:
        if not args.bounce_model.is_file():
            raise FileNotFoundError('Bounce checkpoint missing. Run python download_bounce_model.py, or use --skip-bounce.')
        import catboost  # Fail before tracking if the bounce dependency is missing.
        bounce_paths = [args.output_dir/f'{args.video.stem}_{suffix}' for suffix in
                        ('bounces.json', 'bounce_scores.csv', 'bounces.mp4')]
        if not args.overwrite and any(p.exists() for p in bounce_paths):
            raise FileExistsError('Bounce outputs exist; use --overwrite or another --output-dir.')
    import cv2
    import numpy as np
    import torch
    sys.path.insert(0, str(ROOT / 'RacketVision/source/BallTrack'))
    from model.tracknet_v3 import TrackNet
    if not args.video.is_file() or not args.checkpoint.is_file():
        raise FileNotFoundError('Video or checkpoint missing. Run download_assets.py first.')
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError(
            f'CUDA requested but unavailable (PyTorch {torch.__version__}, '
            f'CUDA build: {torch.version.cuda}). Install CUDA-enabled PyTorch in '
            'this Python environment, or explicitly use --device cpu.'
        )
    if device.type == 'cuda':
        device = torch.device('cuda', device.index if device.index is not None else 0)
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True
        print(f'Using CUDA: {torch.cuda.get_device_name(device)} '
              f'(PyTorch {torch.__version__}, CUDA {torch.version.cuda})', flush=True)
    else:
        print('Using CPU', flush=True)
    # Avoid oversubscribing the CPU on machines with many logical processors.
    if device.type == 'cpu':
        torch.set_num_threads(min(8, torch.get_num_threads()))
    cap = cv2.VideoCapture(str(args.video.resolve()))
    writer = None
    try:
        if not cap.isOpened():
            raise RuntimeError(f'Cannot open video: {args.video}')
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            raise RuntimeError('Video has no valid FPS metadata')
        width, height = int(cap.get(3)), int(cap.get(4))
        roi = [0,0,width,height]
        calibration = None
        if args.court_calibration:
            from .court_geometry import load_calibration
            calibration = load_calibration(args.court_calibration,width,height)
            roi = calibration['tracking_roi_xywh']
            print(f'Court calibration: {args.court_calibration} | tracking ROI: {roi}',flush=True)
        limit = min(total, args.max_frames) if args.max_frames and total > 0 else (args.max_frames or total)
        if args.median:
            if args.median.suffix.lower() == '.npz':
                with np.load(args.median, allow_pickle=False) as archive:
                    median = archive['median']
            else:
                median = cv2.imread(str(args.median))
            if median is None:
                raise RuntimeError('Cannot read median background')
            if args.court_calibration:
                if median.shape != (height,width,3):
                    raise ValueError('--median with --court-calibration must be a full-resolution original-frame BGR image, not a pre-cropped/resized background.')
                median = crop_frame(median,roi)
            median = cv2.resize(median, (512, 288))
        else:
            if limit <= 0:
                raise RuntimeError('Frame count unknown; supply --max-frames or --median')
            print('Sampling background...', flush=True)
            samples = []
            for index in np.unique(np.linspace(0, limit - 1, min(args.median_samples, limit), dtype=int)):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                ok, frame = cap.read()
                if ok:
                    samples.append(cv2.resize(crop_frame(frame,roi), (512, 288)))
            if not samples:
                raise RuntimeError('No decodable background frames')
            median = np.median(np.stack(samples), axis=0).astype(np.uint8)
        cap.release()
        cap = cv2.VideoCapture(str(args.video.resolve()))
        # Preserve OpenCV BGR ordering, as used by the training dataset.
        bg = np.moveaxis(median.astype(np.float32) / 255, -1, 0)
        model = TrackNet(in_dim=15, out_dim=4, last_only=True)
        # Upstream MMEngine checkpoint includes HistoryBuffer metadata.
        # Load only checkpoints from a source you trust.
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        state = checkpoint.get('state_dict', checkpoint)
        state = {k.removeprefix('module.'): v for k, v in state.items()}
        model.load_state_dict(state, strict=True)
        model.to(device).eval()
        args.output_dir.mkdir(parents=True, exist_ok=True)
        stem = args.video.stem
        video_out = args.output_dir / f'{stem}_tracked.mp4'
        csv_out = args.output_dir / f'{stem}_predictions.csv'
        meta_out = args.output_dir / f'{stem}_run.json'
        if not args.overwrite and any(p.exists() for p in (video_out, csv_out, meta_out)):
            raise FileExistsError('Outputs exist; choose another --output-dir or pass --overwrite')
        if video_out.resolve() == args.video.resolve():
            raise ValueError('Output must differ from input')
        writer = cv2.VideoWriter(str(video_out), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError('MP4 encoder could not be opened')
        history, trail = deque(maxlen=4), deque(maxlen=max(1, args.trail))
        count = visible_count = 0
        start = time.perf_counter()
        with csv_out.open('w', newline='') as f, torch.inference_mode():
            csv_writer = csv.writer(f)
            csv_writer.writerow(['Frame', 'TimeSeconds', 'Visibility', 'X', 'Y', 'Confidence'])
            done = False
            while not done:
                raw, inputs = [], []
                for _ in range(args.batch_size):
                    if args.max_frames and count + len(raw) >= args.max_frames:
                        done = True
                        break
                    ok, frame = cap.read()
                    if not ok:
                        done = True
                        break
                    resized = np.moveaxis(cv2.resize(crop_frame(frame,roi), (512, 288)).astype(np.float32) / 255, -1, 0)
                    if not history:
                        history.extend([resized] * 3)
                    history.append(resized)
                    # Current frame is included, matching training labels (upstream inference lags by one).
                    inputs.append(np.concatenate([bg, *history], axis=0))
                    raw.append(frame)
                if not raw:
                    break
                tensor = torch.from_numpy(np.stack(inputs)).to(device)
                predictions, _, _ = model(frames=tensor)
                for frame, heatmap in zip(raw, predictions[:, 0].cpu().numpy()):
                    contours, _ = cv2.findContours((heatmap > args.threshold).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    x = y = 0
                    confidence = 0.0
                    visible = int(bool(contours))
                    if visible:
                        bx, by, bw, bh = max((cv2.boundingRect(c) for c in contours), key=lambda r:r[2]*r[3])
                        x,y = model_to_image(bx+bw/2,by+bh/2,roi)
                        confidence = float(heatmap[by:by+bh, bx:bx+bw].mean())
                        trail.append((x,y))
                        visible_count += 1
                    else:
                        trail.clear()
                    if args.trail:
                        points = list(trail)
                        for a,b in zip(points,points[1:]):
                            cv2.line(frame,a,b,(0,200,255),2)
                    if visible:
                        cv2.circle(frame,(x,y),7,(0,255,0),2)
                    cv2.putText(frame,f'Frame {count} | ball {confidence:.2f}',(15,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
                    writer.write(frame)
                    csv_writer.writerow([count, f'{count/fps:.6f}', visible, x, y, f'{confidence:.6f}'])
                    count += 1
                if count % 100 < args.batch_size:
                    print(f'Processed {count} frames', flush=True)
        if count == 0:
            raise RuntimeError('Video decoded no frames')
        meta_out.write_text(json.dumps(dict(video=str(args.video.resolve()), checkpoint=str(args.checkpoint.resolve()), device=str(device), threshold=args.threshold, frames=count, detected_frames=visible_count, source_fps=fps, court_calibration=str(args.court_calibration.resolve()) if args.court_calibration else None, camera_id=calibration['camera_id'] if calibration else None, tracking_roi_xywh=roi, coordinate_space='original_video_pixels', input_size=[512,288], elapsed_seconds=time.perf_counter()-start),indent=2))
        print(f'Done: {count} frames on {device}.\nVideo: {video_out}\nCSV: {csv_out}')
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    if not args.skip_bounce:
        from .bounce_detection import process_csv
        bounce_paths = process_csv(csv_out, video_out, args.output_dir, stem=stem,
                    overwrite=args.overwrite, model_path=args.bounce_model,
                    threshold=args.bounce_threshold, max_gap=args.bounce_max_gap,
                    min_separation_seconds=args.bounce_min_separation)
        if args.racket_evidence:
            import subprocess
            command=[sys.executable,str(ROOT/'run.py'),'_legacy','review_bounce_candidates',
                     '--csv',str(csv_out),'--candidates',str(bounce_paths['events']),
                     '--rackets',str(args.racket_evidence),'--video',str(video_out),
                     '--output-dir',str(args.output_dir/'review')]
            if args.overwrite: command.append('--overwrite')
            subprocess.run(command,check=True)

if __name__ == '__main__':
    main()
