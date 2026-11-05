"""Offline CatBoost bounce detection from RacketVision tracking CSVs.

Feature definitions follow jack16888/RacketVision's 12 displacement features.
This module is independently implemented; upstream source is GPL-3.0.
Scores are regression outputs, not calibrated probabilities. No court/IN-OUT logic.
"""
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

DEFAULT_MODEL = Path(__file__).resolve().parents[1] / 'models/bounce/ctb_regr_bounce.cbm'
FEATURE_NAMES = ([f'x_diff_{i}' for i in (1, 2)] +
                 [f'x_diff_inv_{i}' for i in (1, 2)] +
                 [f'x_div_{i}' for i in (1, 2)] +
                 [f'y_diff_{i}' for i in (1, 2)] +
                 [f'y_diff_inv_{i}' for i in (1, 2)] +
                 [f'y_div_{i}' for i in (1, 2)])


def prepare_trajectory(rows, fps, width, height, model_fps=30.0,
                       reference_width=1280, reference_height=720, max_gap=2):
    """Resample supported track segments; never extrapolate or fill long gaps."""
    n = len(rows)
    values = np.full((n, 2), np.nan)
    for i, row in enumerate(rows):
        if int(row['Frame']) != i:
            raise ValueError('CSV Frame must be consecutive, zero-based; do not remove missing frames.')
        if int(row['Visibility']):
            xy = np.array([float(row['X']), float(row['Y'])])
            if np.isfinite(xy).all() and 0 <= xy[0] < width and 0 <= xy[1] < height:
                values[i] = xy
    # Fill only short gaps bounded by observed positions. Keep provenance.
    observed = np.isfinite(values).all(axis=1)
    filled = np.zeros(n, dtype=bool)
    indices = np.flatnonzero(observed)
    for left, right in zip(indices, indices[1:]):
        if 0 < right-left-1 <= max_gap:
            for j in range(left+1, right):
                weight = (j-left)/(right-left)
                values[j] = values[left]*(1-weight) + values[right]*weight
                filled[j] = True
    times = np.arange(n)/fps
    grid = np.arange(int(np.floor(times[-1]*model_fps + 1e-8))+1)/model_fps if n else np.array([])
    track = np.full((len(grid), 2), np.nan)
    interpolated = np.zeros(len(grid), dtype=bool)
    supported = np.flatnonzero(np.isfinite(values).all(axis=1))
    segments = np.split(supported, np.flatnonzero(np.diff(supported) > 1)+1)
    for segment in segments:
        if len(segment) == 0:
            continue
        inside = (grid >= times[segment[0]]-1e-9) & (grid <= times[segment[-1]]+1e-9)
        for axis in (0, 1):
            track[inside, axis] = np.interp(grid[inside], times[segment], values[segment, axis])
        # Flag resampled positions that rely on any filled source endpoint.
        fractional = grid[inside]*fps
        lo = np.clip(np.floor(fractional+1e-9).astype(int), 0, n-1)
        hi = np.clip(np.ceil(fractional-1e-9).astype(int), 0, n-1)
        interpolated[inside] = filled[lo] | filled[hi]
    track *= np.array([reference_width/width, reference_height/height])
    return grid, track, interpolated


def features_for_track(track):
    """Return complete five-frame windows and ordered reference features."""
    frames, features = [], []
    for t in range(2, len(track)-2):
        if not np.isfinite(track[t-2:t+3]).all():
            continue
        x, y = track[t]
        dx = [abs(track[t-i, 0]-x) for i in (1, 2)]
        dx_next = [abs(track[t+i, 0]-x) for i in (1, 2)]
        dy = [track[t-i, 1]-y for i in (1, 2)]
        dy_next = [track[t+i, 1]-y for i in (1, 2)]
        features.append(dx + dx_next + [dx[i]/(dx_next[i]+1e-15) for i in (0, 1)] +
                        dy + dy_next + [dy[i]/(dy_next[i]+1e-15) for i in (0, 1)])
        frames.append(t)
    return np.array(frames, dtype=int), pd.DataFrame(features, columns=FEATURE_NAMES, dtype=float)


def select_peaks(scores, threshold, min_separation):
    """One global maximum per contiguous candidate run, then temporal NMS."""
    candidates = np.flatnonzero(np.isfinite(scores) & (scores > threshold))
    groups = np.split(candidates, np.flatnonzero(np.diff(candidates) > 1)+1)
    peaks = [int(g[np.argmax(scores[g])]) for g in groups if len(g)]
    kept = []
    for index in sorted(peaks, key=lambda t: (-scores[t], t)):
        if all(abs(index-other) > min_separation for other in kept):
            kept.append(index)
    return sorted(kept)


def supplemental_peaks(scores, track, interpolated, grid, rows, fps, high_peaks,
                       threshold, separation, floor=.40):
    """Conservative low-score proposals; these still require racket review."""
    accepted = {}
    # Use the same contiguous-run maximum/NMS, keeping high-score events intact.
    for t in select_peaks(scores, np.nextafter(floor, -np.inf), separation):
        if not floor <= scores[t] <= threshold or any(abs(t-h) <= separation for h in high_peaks):
            continue
        if t < 3 or t+3 >= len(track):
            continue
        window = track[t-3:t+4]
        if not np.isfinite(window).all() or interpolated[t-3:t+4].any():
            continue
        lo = int(np.floor(grid[t-3]*fps))
        hi = min(len(rows)-1, int(np.ceil(grid[t+3]*fps)))
        if any(not int(rows[i]['Visibility']) for i in range(lo, hi+1)):
            continue
        # Coordinates are in the 1280x720 reference space by default.
        dy = np.diff(window[:, 1])
        incoming = float(np.mean(dy[:3])); outgoing = float(np.mean(dy[3:]))
        dx = np.diff(window[:, 0])
        horizontal_path = float(np.abs(dx).sum())
        coherence = float(abs(dx.sum())/horizontal_path) if horizontal_path > 1e-6 else 1.
        steps = np.linalg.norm(np.diff(window, axis=0), axis=1)
        # Require a genuine local low point with nontrivial vertical reversal,
        # tolerant of a two-frame flat bottom, and reject jumps/sideways reversal.
        if (incoming < 1. or outgoing > -.75 or
                (dy[:3] < -.75).any() or (dy[3:] > .75).any() or
                window[3, 1] < window[:, 1].max()-.75 or
                coherence < .85 or steps.max() > 3*max(float(np.median(steps)), 1.)):
            continue
        accepted[t] = dict(incoming_y_per_frame=incoming, outgoing_y_per_frame=outgoing,
                           horizontal_coherence=coherence, fully_observed=True,
                           model_window_frames=[int(t-3), int(t+3)])
    return accepted


def analyze(rows, fps, width, height, model_path=DEFAULT_MODEL, threshold=0.55,
            model_fps=30.0, reference_width=1280, reference_height=720,
            max_gap=2, min_separation_seconds=0.12):
    from catboost import CatBoostRegressor
    if not np.isfinite([fps, width, height, model_fps, reference_width, reference_height]).all() or min(fps, width, height, model_fps, reference_width, reference_height) <= 0:
        raise ValueError('FPS and image/reference dimensions must be positive and finite.')
    if not 0 < threshold < 1 or max_gap < 0 or min_separation_seconds < 0:
        raise ValueError('Invalid bounce threshold, gap limit, or separation.')
    if not rows:
        raise ValueError('Tracking CSV is empty.')
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f'Bounce model missing: {model_path}. Run download_bounce_model.py.')
    model = CatBoostRegressor()
    model.load_model(str(model_path))
    if list(model.feature_names_) != FEATURE_NAMES:
        raise ValueError('Bounce checkpoint must accept the reference 12-feature schema.')
    grid, track, interpolated = prepare_trajectory(rows, fps, width, height, model_fps,
                                                  reference_width, reference_height, max_gap)
    indices, features = features_for_track(track)
    scores = np.full(len(grid), np.nan)
    if len(indices):
        scores[indices] = model.predict(features,thread_count=1)
    separation = int(round(min_separation_seconds*model_fps))
    peaks = select_peaks(scores, threshold, separation)
    supplemental = supplemental_peaks(scores, track, interpolated, grid, rows, fps,
                                       peaks, threshold, separation)
    peaks = sorted(peaks + list(supplemental))
    events = []
    for t in peaks:
        source_frame = min(len(rows)-1, int(round(grid[t]*fps)))
        events.append(dict(frame=source_frame, time_seconds=source_frame/fps,
                           model_time_seconds=float(grid[t]),
                           x=float(track[t, 0]*width/reference_width),
                           y=float(track[t, 1]*height/reference_height),
                           score=float(scores[t]),
                           window_uses_interpolation=bool(interpolated[t-2:t+3].any()),
                           source_position_observed=bool(int(rows[source_frame]['Visibility']))))
        events[-1]['candidate_path'] = 'trajectory_supplement' if t in supplemental else 'catboost_high_score'
        if t in supplemental:
            events[-1]['supplemental_evidence'] = supplemental[t]
    diagnostics = []
    event_frames = {e['frame'] for e in events}
    for row in rows:
        frame = int(row['Frame'])
        t = int(round(frame/fps*model_fps))
        valid = 0 <= t < len(scores) and np.isfinite(scores[t])
        diagnostics.append(dict(Frame=frame, TimeSeconds=frame/fps,
                                Score=float(scores[t]) if valid else '',
                                Evaluated=int(valid), Bounce=int(frame in event_frames),
                                WindowUsesInterpolation=int(interpolated[max(0,t-2):t+3].any()) if valid else ''))
    metadata = dict(model=str(model_path.resolve()), model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
                    threshold=threshold, source_fps=fps, model_fps=model_fps,
                    reference_width=reference_width, reference_height=reference_height,
                    max_gap_frames=max_gap, min_separation_seconds=min_separation_seconds,
                    source_frames=len(rows), evaluated_model_frames=len(indices),
                    score_type='CatBoost regression score; not a probability',
                    lookahead_model_frames=2, bounce_count=len(events),
                    supplemental_path=dict(enabled=True, score_floor=.40, score_ceiling=threshold,
                                           lookahead_model_frames=3, count=len(supplemental),
                                           requires='fully observed, vertical reversal, horizontal coherence'))
    return events, diagnostics, metadata


def render(video_path, target, events, max_frames, hold_seconds=0.4):
    cap = cv2.VideoCapture(str(video_path))
    writer = None
    try:
        if not cap.isOpened():
            raise RuntimeError(f'Cannot open {video_path}')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width, height = int(cap.get(3)), int(cap.get(4))
        writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width,height))
        if not writer.isOpened():
            raise RuntimeError('Cannot open bounce video writer.')
        count = 0
        for index in range(max_frames):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError('Video has fewer frames than the tracking CSV.')
            for number, event in enumerate(events, 1):
                if 0 <= index-event['frame'] <= int(hold_seconds*fps):
                    point = (int(round(event['x'])), int(round(event['y'])))
                    cv2.circle(frame, point, 15, (0,165,255), 3)
                    cv2.putText(frame, f'BOUNCE CANDIDATE {number} | frame {event["frame"]} | score {event["score"]:.2f}',
                                (15,65), cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,165,255),2)
            writer.write(frame)
            count += 1
        return count
    finally:
        cap.release()
        if writer is not None:
            writer.release()


def process_csv(csv_path, video_path, output_dir, stem=None, overwrite=False,
                render_video=True, **options):
    csv_path, video_path, output_dir = Path(csv_path), Path(video_path), Path(output_dir)
    with csv_path.open(newline='') as f:
        rows = list(csv.DictReader(f))
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f'Cannot open {video_path}')
        fps, width, height = cap.get(cv2.CAP_PROP_FPS), int(cap.get(3)), int(cap.get(4))
    finally:
        cap.release()
    stem = stem or (csv_path.stem.removesuffix('_predictions'))
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = dict(events=output_dir/f'{stem}_bounces.json',
                 scores=output_dir/f'{stem}_bounce_scores.csv')
    if render_video:
        paths['video'] = output_dir/f'{stem}_bounces.mp4'
    if any(p.resolve() in (video_path.resolve(), csv_path.resolve()) for p in paths.values()):
        raise ValueError('Bounce outputs must differ from inputs.')
    if not overwrite and any(p.exists() for p in paths.values()):
        raise FileExistsError('Bounce outputs exist; use --overwrite or a new output directory.')
    events, diagnostics, metadata = analyze(rows, fps, width, height, **options)
    paths['events'].write_text(json.dumps(dict(metadata=metadata, events=events),indent=2),encoding='utf8')
    with paths['scores'].open('w',newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(diagnostics[0]))
        writer.writeheader()
        writer.writerows(diagnostics)
    if render_video:
        render(video_path, paths['video'], events, len(rows))
    print(f'Bounce detection: {len(events)} candidate events, {metadata["evaluated_model_frames"]} evaluated model frames.',flush=True)
    for name, path in paths.items():
        print(f'{name}: {path}')
    return paths
