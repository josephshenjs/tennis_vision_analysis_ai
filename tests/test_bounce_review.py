import unittest
from rally_analysis.compat.review_bounce_candidates import review,head_distance


def trajectory(turn=True):
    return [dict(Frame=i,Visibility=1,X=100+10*(abs(i-5) if turn else i),Y=100,Confidence=.6) for i in range(11)]

def pose(x=100,y=100):
    return dict(bbox_score=.9,keypoints=[[x,y-10],[x,y+10],[x,y+25],[x-7,y],[x+7,y]],keypoint_scores=[.9]*5)

class BounceReviewTests(unittest.TestCase):
    event=dict(frame=5,time_seconds=5/30,x=100,y=100,score=.6)
    def test_contact_and_reversal_jointly_mark_hit(self):
        rows=trajectory(); rackets={str(i):dict(evaluated=True,predictions=[pose()]) for i in range(2,9)}
        result=review([self.event],rows,rackets,1920,1080)[0]
        self.assertEqual(result['status'],'likely_racket_hit')
        self.assertEqual(result['score'],.6)

    def test_proximity_alone_does_not_reject_bounce(self):
        rows=trajectory(False); rackets={'5':dict(evaluated=True,predictions=[pose(150,100)])}
        result=review([self.event],rows,rackets,1920,1080)[0]
        self.assertEqual(result['status'],'uncertain_contact')

    def test_missing_pose_is_unknown_not_absence(self):
        result=review([self.event],trajectory(),{},1920,1080)[0]
        self.assertEqual(result['status'],'uncertain_racket_evidence')

    def test_distant_in_time_pose_does_not_establish_absence_at_event(self):
        result=review([self.event],trajectory(),{'2':dict(evaluated=True,predictions=[pose(350,100)])},1920,1080)[0]
        self.assertEqual(result['status'],'uncertain_racket_evidence')

    def test_two_frame_gap_does_not_automatically_reject(self):
        rows=trajectory(); rows[6]['Visibility']=rows[7]['Visibility']=0
        rackets={str(i):dict(evaluated=True,predictions=[pose(350,100)]) for i in range(2,9)}
        result=review([self.event],rows,rackets,1920,1080)[0]
        self.assertEqual(result['status'],'supported_bounce_candidate')
        self.assertEqual(result['evidence']['max_missing_run'],2)

    def test_large_jump_is_uncertain(self):
        rows=trajectory(); rows[4]['X']=1800
        result=review([self.event],rows,{},1920,1080)[0]
        self.assertEqual(result['status'],'uncertain_trajectory')

    def test_real_frame166_occluded_contact_regression(self):
        import json
        from pathlib import Path
        fixture=json.loads((Path(__file__).parent/'fixtures/contact166.json').read_text())
        result=review([fixture['event']],fixture['rows'],fixture['racket_frames'],1920,1080)[0]
        self.assertEqual(result['status'],'likely_racket_hit')
        contact=result['evidence']['nearest_gap_racket_contact']
        self.assertEqual(contact['frame'],8)
        self.assertEqual(contact['proposal_verified_by_frame'],10)
        self.assertLess(contact['distance_head_lengths'],.1)
        self.assertEqual(fixture['rows'][8]['Visibility'],0)

    def test_short_gap_turn_with_offset_contact_is_a_hit(self):
        rows=[]
        for i in range(13):
            x=100+10*abs(i-7)
            rows.append(dict(Frame=i,Visibility=int(i not in (6,7)),X=x,Y=100,Confidence=.8))
        event=dict(self.event,frame=5)
        pred=pose(100,100)
        pred.update(bbox_score=0.,proposal_method='trajectory_guided_pose',proposal_verified_by_frame=9)
        rackets={'7':dict(evaluated=True,predictions=[pred])}
        result=review([event],rows,rackets,1920,1080)[0]
        self.assertEqual(result['status'],'likely_racket_hit')
        self.assertEqual(result['evidence']['nearest_racket_contact']['frame'],7)
        self.assertEqual(result['evidence']['nearest_gap_racket_contact']['missing_frames'],[6,7])
        self.assertEqual(rows[7]['Visibility'],0)

    def test_uncorroborated_pose_proposal_cannot_veto_bounce(self):
        pred=pose(); pred.update(bbox_score=0.,proposal_method='trajectory_guided_pose')
        self.assertIsNone(head_distance([100,100],pred))

    def test_long_gap_cannot_create_contact_evidence(self):
        from rally_analysis.compat.review_bounce_candidates import gap_contact
        rows=trajectory()
        for i in (4,5,6): rows[i]['Visibility']=0
        self.assertIsNone(gap_contact(rows,5,pose()))

    def test_low_quality_head_not_contact_evidence(self):
        pred=pose(); pred['keypoint_scores']=[.1]*5
        self.assertIsNone(head_distance([100,100],pred))

if __name__=='__main__': unittest.main()
