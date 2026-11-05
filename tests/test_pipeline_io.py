import tempfile
import unittest
from pathlib import Path
import threading
import time
from rally_analysis.video_io import prefetch
from rally_analysis.pipeline import prepare_jobs,far_window
from rally_analysis.compat.court_geometry import build_calibration

class PipelineIOTests(unittest.TestCase):
    def test_prefetch_preserves_order_and_propagates_producer_error(self):
        with prefetch(iter(range(20)),2) as values:
            self.assertEqual(list(values),list(range(20)))
        def broken():
            yield 1
            raise RuntimeError('decode failed')
        with self.assertRaisesRegex(RuntimeError,'decode failed'):
            with prefetch(broken(),1) as values: list(values)

    def test_early_consumer_exit_stops_bounded_producer(self):
        stopped=threading.Event()
        def source():
            try:
                for i in range(100000): yield i
            finally: stopped.set()
        with prefetch(source(),1) as values: self.assertEqual(next(values),0)
        self.assertTrue(stopped.wait(1))

    def test_same_basename_jobs_do_not_collide(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            jobs=prepare_jobs([root/'a/clip.mp4',root/'b/clip.mp4'],root/'out',root/'court.json')
            self.assertNotEqual(jobs[0][1],jobs[1][1])
            folder=jobs[0][1]; folder.mkdir(); (folder/'events.json').write_text('{}')
            with self.assertRaises(FileExistsError):
                prepare_jobs([root/'a/clip.mp4',root/'b/clip.mp4'],root/'out',root/'court.json')

    def test_explicit_and_derived_far_roi_within_image(self):
        calibration=build_calibration([[700,300],[1100,300],[1500,900],[200,900]],1920,1080,'test')
        self.assertEqual(far_window(calibration,1920,1080,[650,250,650,320]),[650,250,650,320])
        x,y,w,h=far_window(calibration,1920,1080)
        self.assertTrue(0<=x<x+w<=1920 and 0<=y<y+h<=1080)

if __name__=='__main__': unittest.main()
