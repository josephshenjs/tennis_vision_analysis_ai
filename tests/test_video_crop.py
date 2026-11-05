import unittest
import tempfile
import json
from pathlib import Path
import numpy as np
from rally_analysis.compat.test_video import crop_frame,model_to_image
from rally_analysis.compat.court_geometry import build_calibration,load_calibration

class VideoCropTests(unittest.TestCase):
    def test_native_crop_and_prediction_offset(self):
        frame=np.arange(120*160*3).reshape(120,160,3)
        roi=[20,30,80,60]
        np.testing.assert_array_equal(crop_frame(frame,roi),frame[30:90,20:100])
        self.assertEqual(model_to_image(256,144,roi),(60,60))
        self.assertEqual(model_to_image(0,0,roi),(20,30))
        self.assertEqual(model_to_image(512,288,roi),(99,89))

    def test_full_frame_mapping_preserves_previous_coordinates(self):
        self.assertEqual(model_to_image(256,144,[0,0,1920,1080]),(960,540))

    def test_calibration_rejects_mismatched_video_resolution(self):
        c=build_calibration([[50,20],[100,20],[140,100],[10,100]],160,120,'test')
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'court.json'; p.write_text(json.dumps(c))
            self.assertEqual(load_calibration(p,160,120)['camera_id'],'test')
            with self.assertRaisesRegex(ValueError,'size differs'):
                load_calibration(p,320,240)

if __name__=='__main__': unittest.main()
