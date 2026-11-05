import unittest
import numpy as np
from rally_analysis.court_tracking import CourtGate, fuse_target

class TargetCourtTests(unittest.TestCase):
    def gate(self):
        return CourtGate({'H_court_m_to_image':[[5,0,50],[0,2,30],[0,0,1]]},150,100,margin_m=0,lift_fraction=.2)
    def test_flight_corridor_keeps_high_ball_and_rejects_neighbor(self):
        gate=self.gate()
        self.assertTrue(gate.allows(75,15))
        self.assertFalse(gate.allows(20,40))
        self.assertFalse(gate.allows(75,0))
        self.assertEqual(gate.model_mask([0,0,150,100]).shape,(288,512))
    def rows(self,points):
        return [dict(Frame=i,TimeSeconds=i/30,Visibility=1,X=x,Y=y,Confidence=.9) for i,(x,y) in enumerate(points)]
    def test_far_view_recovers_target_when_full_switches_courts(self):
        full=self.rows([(70,40),(71,40),(72,40),(20,40),(21,40)])
        far=self.rows([(70,40),(71,40),(72,40),(73,40),(74,40)])
        output,audit=fuse_target(full,far,self.gate())
        self.assertEqual(output[3]['X'],73)
        self.assertEqual(audit[3]['SpatialRejected'],1)
        self.assertEqual(audit[3]['SelectedWindow'],'far')
    def test_isolated_detection_does_not_seed_trajectory(self):
        rows=self.rows([(70,40)])
        self.assertEqual(fuse_target(rows,None,self.gate())[0][0]['Visibility'],0)
    def test_misaligned_windows_fail(self):
        rows=self.rows([(70,40)]*3)
        with self.assertRaises(ValueError): fuse_target(rows,rows[:2],self.gate())

if __name__=='__main__': unittest.main()
