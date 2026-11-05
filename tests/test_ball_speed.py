import unittest
import numpy as np
from rally_analysis.ball_speed import camera_projection,estimate_speeds
from rally_analysis.shot_overlay import visible_shots,draw_shot_overlay


def fixture(fps=60):
    k=np.array([[1000.,0,640],[0,1000.,360],[0,0,1]])
    centre=np.array([5.485,35.,10.]); forward=np.array([5.485,10.,0])-centre
    forward/=np.linalg.norm(forward); right=np.cross(forward,[0,0,1]); right/=np.linalg.norm(right)
    down=np.cross(forward,right); r=np.array([right,down,forward]); p=k@np.column_stack([r,-r@centre])
    h=p[:,[0,1,3]]
    calibration=dict(image_size=dict(width=1280,height=720),H_court_m_to_image=h.tolist(),H_image_to_court_m=np.linalg.inv(h).tolist())
    rows=[]; bounce=round(.7*fps)
    for i in range(bounce+10):
        t=(i-bounce)/fps
        xyz=np.array([5+3*t,6-22*t,-5*t-.5*9.81*t*t])
        q=p@np.r_[xyz,1]; xy=q[:2]/q[2]
        rows.append(dict(Frame=i,Visibility=1,X=xy[0],Y=xy[1],Confidence=.9))
    event=dict(frame=bounce,x=rows[bounce]['X'],y=rows[bounce]['Y'],status='supported_bounce_candidate')
    return calibration,rows,event,p


class SpeedTests(unittest.TestCase):
    def test_known_3d_speed_and_fps_invariance(self):
        for fps in (30,60,120):
            c,rows,event,p=fixture(fps)
            np.testing.assert_allclose(camera_projection(c),p/p[2,3]*np.asarray(c['H_court_m_to_image'])[2,2],rtol=1e-7,atol=1e-7)
            result=estimate_speeds(rows,[event],c,fps)['events'][0]
            self.assertEqual(result['status'],'estimated')
            best=result['best_prediction']; t=(best['speed_frame']-event['frame'])/fps
            expected=np.linalg.norm([3,-22,-5-9.81*t])*3.6
            self.assertAlmostEqual(best['speed_kmh'],expected,places=5)
            self.assertEqual(best,min(result['candidates'],key=lambda x:x['selection_cost']))
            self.assertEqual(visible_shots({'events':[result]},event['frame']),[])
            self.assertEqual(len(visible_shots({'events':[result]},event['frame']+3)),1)
            self.assertEqual(result['available_frame'],event['frame']+6)

    def test_short_missing_gap_uses_observations_only(self):
        c,rows,e,_=fixture()
        rows[e['frame']-4]['Visibility']=0
        result=estimate_speeds(rows,[e],c,60)['events'][0]
        self.assertEqual(result['status'],'estimated')
        self.assertLess(result['best_prediction']['fit_rmse_px'],1e-5)

    def test_invalid_camera_still_keeps_ground_landing(self):
        c,rows,e,_=fixture(); c['H_court_m_to_image']=np.eye(3).tolist()
        result=estimate_speeds(rows,[e],c,60)['events'][0]
        self.assertEqual(result['status'],'estimated_low_confidence')
        self.assertIn('landing_court_m',result)
        self.assertEqual(result['best_prediction']['method'],'court_plane_motion_proxy')

    def test_uncertain_bounce_is_recorded_and_gets_best_available_speed(self):
        c,rows,e,_=fixture(); e['status']='uncertain_racket_evidence'
        result=estimate_speeds(rows,[e],c,60)['events'][0]
        self.assertEqual(result['status'],'estimated_low_confidence')
        self.assertIsNotNone(result['best_prediction'])
        self.assertIn('landing_court_m',result)
        self.assertEqual(len(visible_shots({'events':[result]},e['frame']+3)),1)

    def test_sparse_trajectory_still_provides_best_available_number(self):
        c,rows,e,_=fixture()
        for i in range(e['frame']):
            if i<e['frame']-2: rows[i]['Visibility']=0
        result=estimate_speeds(rows,[e],c,60)['events'][0]
        self.assertEqual(result['status'],'estimated_low_confidence')
        self.assertIsNotNone(result['best_prediction'])
        self.assertGreater(result['best_prediction']['speed_kmh'],0)

    def test_outlier_rejection(self):
        c,rows,e,_=fixture(); rows[25]['X']+=150; rows[25]['Y']-=100
        result=estimate_speeds(rows,[e],c,60)['events'][0]
        self.assertEqual(result['status'],'estimated')
        self.assertLess(result['best_prediction']['fit_rmse_px'],1e-5)

    def test_missing_and_uncertain_do_not_invent_speed(self):
        c,rows,e,_=fixture()
        for row in rows: row['Visibility']=0
        result=estimate_speeds(rows,[e],c,60)['events'][0]
        self.assertIsNone(result['best_prediction'])
        e['status']='uncertain_trajectory'
        result=estimate_speeds(rows,[e],c,60)['events'][0]
        self.assertEqual(result['reason'],'no_measurable_trajectory')
        self.assertIn('landing_court_m',result)
        e['status']='likely_racket_hit'
        self.assertEqual(estimate_speeds(rows,[e],c,60)['events'],[])

    def test_overlay_delays_speed_until_after_landing_and_retains_last_valid(self):
        import copy
        c,rows,e,_=fixture(); report=estimate_speeds(rows,[e],c,60)
        event=report['events'][0]; hidden=copy.deepcopy(report)
        hidden['events'][0]['best_prediction']=None
        def painted(value,frame):
            return draw_shot_overlay(np.zeros((546,1143,3),np.uint8),value,frame)
        landing=e['frame']+3; speed=e['frame']+6
        np.testing.assert_array_equal(painted(report,landing),painted(hidden,landing))
        self.assertFalse(np.array_equal(painted(report,speed),painted(hidden,speed)))
        later=copy.deepcopy(event); later.update(best_prediction=None,available_frame=100,landing_available_frame=97)
        report['events'].append(later)
        # Missing a later estimate must not erase the retained valid speed panel.
        np.testing.assert_array_equal(painted(report,101)[430:,:],painted({'events':[event]},101)[430:,:])

    def test_speed_retained_and_overlay_handles_small_video(self):
        c,rows,e,_=fixture(); report=estimate_speeds(rows,[e],c,60)
        self.assertEqual(len(visible_shots(report,1000)),1)
        for h,w in [(546,1143),(144,256)]:
            image=np.full((h,w,3),128,np.uint8)
            before=image.copy(); draw_shot_overlay(image,report,1000)
            self.assertFalse(np.array_equal(before,image))
            np.testing.assert_array_equal(image[:,:w//2],before[:,:w//2])

if __name__=='__main__': unittest.main()
