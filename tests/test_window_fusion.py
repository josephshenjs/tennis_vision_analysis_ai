import unittest
from rally_analysis.compat.fuse_tracking_windows import fuse


def row(i,x=None,y=50,confidence=.6):
    return dict(Frame=str(i),TimeSeconds=f'{i/30:.6f}',Visibility=str(int(x is not None)),
                X=str(x or 0),Y=str(y if x is not None else 0),Confidence=str(confidence))

class FusionTests(unittest.TestCase):
    def test_supported_far_track_bridges_full_gap(self):
        full=[row(i,100+2*i if i in (0,6) else None) for i in range(7)]
        far=[row(i,100+2*i) for i in range(7)]
        fused,audit=fuse(full,far,[50,0,200,100])
        self.assertEqual(audit[3]['Reason'],'far_bridges_full_gap')
        self.assertEqual(fused[3]['Visibility'],'1')

    def test_high_confidence_conflict_does_not_override_anchor(self):
        full=[row(i,100+i) for i in range(7)]
        far=[row(i,250+i,confidence=.99) for i in range(7)]
        fused,audit=fuse(full,far,[0,0,400,100])
        self.assertEqual(fused[3]['X'],full[3]['X'])
        self.assertEqual(audit[3]['Reason'],'full_kept_window_disagreement')

    def test_unanchored_or_isolated_far_detections_do_not_create_track(self):
        full=[row(i) for i in range(7)]
        far=[row(i,100+i) for i in range(7)]
        fused,_=fuse(full,far,[0,0,400,100])
        self.assertTrue(all(r['Visibility']=='0' for r in fused))
        full=[row(i,100+i) for i in range(7)]; full[3]=row(3)
        far=[row(i,103 if i==3 else None) for i in range(7)]
        fused,audit=fuse(full,far,[0,0,400,100])
        self.assertEqual(fused[3]['Visibility'],'0')
        self.assertEqual(audit[3]['Reason'],'far_rejected_isolated')

    def test_frame_misalignment_rejected(self):
        full=[row(i,100+i) for i in range(7)]; far=[row(i,100+i) for i in range(7)]
        far[3]['Frame']='4'
        with self.assertRaises(ValueError): fuse(full,far,[0,0,400,100])

if __name__=='__main__': unittest.main()
