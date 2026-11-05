import unittest
from rally_analysis.gap_recovery import plan_recovery,merge_recovery

def rows():
    return [dict(Frame=i,Visibility=int(i not in (6,7,8)),X=100+i*2,Y=100,Confidence=.6) for i in range(16)]

class Gate:
    def allows(self,x,y): return x<200

class RecoveryTests(unittest.TestCase):
    def test_bounded_crop_and_context(self):
        plans=plan_recovery(rows(),640,360,30)
        self.assertEqual(len(plans),1)
        self.assertEqual((plans[0]['start'],plans[0]['end']),(6,9))
        x,y,w,h=plans[0]['roi']
        self.assertTrue(0<=x<=640-w and 0<=y<=360-h)

    def test_observed_recovery_not_interpolation_and_no_overwrite(self):
        source=rows(); plan=plan_recovery(source,640,360,30)[0]
        local=rows()
        for r in local: r['Visibility']=1
        accepted,reason=merge_recovery(source,local,plan,Gate(),30)
        self.assertEqual(accepted,[6,7,8])
        self.assertEqual(reason,'recovered')
        self.assertEqual(source[7]['X'],114)

    def test_unanchored_and_outside_court_detections_rejected(self):
        source=rows(); plan=plan_recovery(source,640,360,30)[0]
        local=rows()
        for r in local: r.update(Visibility=1,X=300)
        self.assertEqual(merge_recovery(source,local,plan,Gate(),30)[0],[])
        for r in local: r.update(X=180)
        self.assertEqual(merge_recovery(source,local,plan,Gate(),30)[0],[])

    def test_long_gaps_and_cold_start_not_searched(self):
        source=rows()
        for r in source: r['Visibility']=0
        self.assertEqual(plan_recovery(source,640,360,30),[])

if __name__=='__main__': unittest.main()
