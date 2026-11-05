import importlib.util
from pathlib import Path
import unittest

import numpy as np
from rally_analysis.compat.bounce_detection import features_for_track, prepare_trajectory, select_peaks, supplemental_peaks


def rows(points):
    return [dict(Frame=i, Visibility=int(p is not None), X=0 if p is None else p[0],
                 Y=0 if p is None else p[1]) for i,p in enumerate(points)]


class BounceTests(unittest.TestCase):
    def supplement(self, points, missing=False, high=()):
        track=np.array(points,float)
        source=rows(points)
        if missing: source[2]['Visibility']=0
        scores=np.array([0.,0.,0.,.53,0.,0.,0.])
        return supplemental_peaks(scores,track,np.zeros(7,bool),np.arange(7)/30,
                                  source,30,list(high),.55,4)

    def test_low_score_flat_bottom_reversal_enters_review(self):
        points=[(i*6,y) for i,y in enumerate([320,333,346,346,343,341,340])]
        self.assertIn(3,self.supplement(points))
        self.assertFalse(self.supplement(points,missing=True))
        self.assertFalse(self.supplement(points,high=[4]))

    def test_low_score_monotonic_motion_and_horizontal_jump_rejected(self):
        self.assertFalse(self.supplement([(i*6,300+i*5) for i in range(7)]))
        points=[(i*6,y) for i,y in enumerate([320,333,346,346,343,341,340])]
        points[4]=(300,343)
        self.assertFalse(self.supplement(points))

    def test_features_match_reference(self):
        # Check against the reference implementation, not a second copy of our formula.
        path=Path(__file__).resolve().parents[1]/'reference_RacketVision/detectors/bounce_detector.py'
        if not path.exists():
            self.skipTest('Reference checkout unavailable')
        spec=importlib.util.spec_from_file_location('reference_bounce',path)
        reference=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference)
        track=np.array([[100+i*i,200+3*i-i*i/3] for i in range(15)])
        idx,actual=features_for_track(track)
        expected,frames=reference.BounceDetector().prepare_features(track[:,0],track[:,1])
        np.testing.assert_array_equal(idx,frames)
        np.testing.assert_allclose(actual.to_numpy(),expected.to_numpy())
        self.assertEqual(list(actual.columns),list(expected.columns))

    def test_short_gap_and_no_long_gap_or_endpoint_extrapolation(self):
        source=rows([None,(10,20),None,(30,40),None,None,None,(70,80),None])
        _,track,filled=prepare_trajectory(source,30,1280,720,max_gap=1)
        np.testing.assert_allclose(track[2],[20,30])
        self.assertTrue(filled[2])
        self.assertTrue(np.isnan(track[[0,4,5,6,8]]).all())
        idx,_=features_for_track(track)
        self.assertEqual(len(idx),0)

    def test_resampling_preserves_times_and_scaled_coordinates(self):
        source=rows([(2*i,3*i) for i in range(13)])
        grid,track,_=prepare_trajectory(source,60,2560,1440)
        np.testing.assert_allclose(grid,np.arange(7)/30)
        np.testing.assert_allclose(track,[[2*i,3*i] for i in range(7)])

    def test_peak_is_global_max_not_last_local_rise(self):
        scores=np.array([np.nan,0.9,0.6,0.8,0.1,0.7,np.nan,0.95])
        self.assertEqual(select_peaks(scores,0.55,0),[1,5,7])
        self.assertEqual(select_peaks(scores,0.55,2),[1,7])

    def test_missing_frames_must_not_be_removed(self):
        source=rows([(10,20)]*6)
        source[3]['Frame']=4
        with self.assertRaisesRegex(ValueError,'consecutive'):
            prepare_trajectory(source,30,1280,720)

    def test_border_coordinate_zero_is_valid_when_visible(self):
        source=rows([(0,10)]*5)
        _,track,_=prepare_trajectory(source,30,1280,720)
        idx,_=features_for_track(track)
        self.assertEqual(idx.tolist(),[2])

    def test_short_clip_has_no_complete_window(self):
        idx,features=features_for_track(np.ones((4,2)))
        self.assertEqual(len(idx),0)
        self.assertEqual(features.shape,(0,12))


if __name__=='__main__':
    unittest.main()
