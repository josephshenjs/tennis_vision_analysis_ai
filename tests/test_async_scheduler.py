import asyncio
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from rally_analysis.pipeline import analyze_videos


class FakeEngine:
    def __init__(self,*args): self.load_counts={'ball':1,'racket':0}
    def recover_gaps(self,video,info,rows,*args): return rows,dict(seconds=0,attempts=[],recovered_frames=[])
    def track(self,*args):
        rows=[dict(Frame=i,TimeSeconds=i/30,Visibility=0,X=0,Y=0,Confidence=0) for i in range(5)]
        return [rows],dict(frames=5)
    def rackets(self,*args): return dict(frames={},seconds=0)


class AsyncSchedulerTests(unittest.TestCase):
    def test_cancellation_stops_worker_and_records_cancelled_status(self):
        import json
        import threading
        started=threading.Event(); exited=threading.Event()
        class WaitingEngine(FakeEngine):
            def track(self,*args):
                started.set()
                if not self.stop.wait(3):
                    raise RuntimeError('Worker did not receive cancellation')
                exited.set()
                raise InterruptedError('Analysis cancelled.')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            args=SimpleNamespace(videos=[root/'clip.mp4'],output_dir=root/'out',
                 court_calibration=root/'court.json',overwrite=False,jobs=1,cpu_workers=1,
                 device='cuda:0',amp=False,prefetch=1,single_window=True,far_roi=None,
                 batch_size=1,threshold=.5,median_samples=1,max_frames=5)
            async def cancel_run():
                task=asyncio.create_task(analyze_videos(args))
                for _ in range(200):
                    if started.is_set(): break
                    await asyncio.sleep(.01)
                self.assertTrue(started.is_set())
                task.cancel()
                with self.assertRaises(asyncio.CancelledError): await task
            with patch('rally_analysis.engine.Engine',WaitingEngine), \
                 patch('rally_analysis.pipeline.metadata',return_value=dict(width=100,height=100,fps=30)), \
                 patch('rally_analysis.pipeline.load_calibration',return_value={'tracking_roi_xywh':[0,0,100,100], 'H_court_m_to_image':[[5,0,20],[0,3,10],[0,0,1]]}):
                asyncio.run(cancel_run())
            self.assertTrue(exited.is_set())
            self.assertEqual(json.loads((root/'out/clip/status.json').read_text())['state'],'cancelled')

    def test_failed_clip_does_not_cancel_other_clip(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            args=SimpleNamespace(videos=[root/'bad.mp4',root/'good.mp4'],output_dir=root/'out',
                 court_calibration=root/'court.json',overwrite=False,jobs=2,cpu_workers=2,
                 device='cuda:0',amp=False,prefetch=1,single_window=True,far_roi=None,
                 batch_size=1,threshold=.5,median_samples=1,max_frames=5,bounce_threshold=.55,
                 bounce_model=root/'bounce.cbm',no_video=True)
            def meta(path):
                if path.name=='bad.mp4': raise RuntimeError('invalid video')
                return dict(width=100,height=100,fps=30,declared_frames=5)
            with patch('rally_analysis.engine.Engine',FakeEngine), \
                 patch('rally_analysis.pipeline.metadata',meta), \
                 patch('rally_analysis.pipeline.load_calibration',return_value={'tracking_roi_xywh':[0,0,100,100], 'H_court_m_to_image':[[5,0,20],[0,3,10],[0,0,1]]}), \
                 patch('rally_analysis.pipeline.analyze',return_value=([],[],{})):
                self.assertEqual(asyncio.run(analyze_videos(args)),1)
            import json
            results=json.loads((root/'out/summary.json').read_text())['results']
            self.assertEqual([r['state'] for r in results],['failed','complete'])
            self.assertTrue((root/'out/good/events.json').is_file())

if __name__=='__main__': unittest.main()
