"""Re-use saved tracking/review outputs to validate speed and clean overlay on source video."""
import csv,json,threading
from pathlib import Path
from rally_analysis.pipeline import render,write_json
from rally_analysis.ball_speed import estimate_speeds
from rally_analysis.video_io import metadata
from rally_analysis.compat.court_geometry import load_calibration
from rally_analysis.paths import ROOT
import cv2

def main():
    for name in ('point_0003','point_0013','point_0021'):
        source=ROOT/'outputs'/name
        run=json.loads((source/'run.json').read_text())
        video=Path(run['video']); info=metadata(video)
        with (source/'predictions.csv').open() as f: rows=list(csv.DictReader(f))
        events=json.loads((source/'events.json').read_text())['events']
        calibration=load_calibration(run['calibration'],info['width'],info['height'])
        report=estimate_speeds(rows,events,calibration,info['fps'])
        out=ROOT/'outputs/speed_revision'/name; out.mkdir(parents=True,exist_ok=True)
        write_json(out/'speeds.json',report)
        write_json(out/'events.json',dict(events=events,summary=dict(display_bounce_frames=[e['bounce_frame'] for e in report['events'] if 'landing_court_m' in e])))
        print(name,[(e['bounce_frame'],e['reason'],None if not e['best_prediction'] else round(e['best_prediction']['speed_kmh'],1)) for e in report['events']],flush=True)
        render(video,out/'analysis.mp4',rows,events,{},threading.Event(),report)
        cap=cv2.VideoCapture(str(out/'analysis.mp4')); count=0; preview=max([e['available_frame'] for e in report['events'] if e.get('best_prediction')]+[len(rows)-1])
        while True:
            ok,image=cap.read()
            if not ok: break
            if count==preview: cv2.imwrite(str(out/'preview.jpg'),image)
            if name=='point_0003' and count==147: cv2.imwrite(str(out/'bounce_4s.jpg'),image)
            count+=1
        cap.release()
        assert count==len(rows),(count,len(rows))
        print('decoded',count,'frames',flush=True)
if __name__=='__main__': main()
