import unittest
import numpy as np
from rally_analysis.compat.court_geometry import build_calibration, validate_corners, transform, CORNERS_M, validate_roi

class CourtGeometryTests(unittest.TestCase):
    corners=[[700,300],[1100,300],[1500,900],[200,900]]

    def test_homography_and_inverse_roundtrip(self):
        c=build_calibration(self.corners,1920,1080,'test')
        np.testing.assert_allclose(transform(self.corners,c['H_image_to_court_m']),CORNERS_M,atol=1e-5)
        points=np.array([[800,400],[900,700]])
        np.testing.assert_allclose(transform(transform(points,c['H_image_to_court_m']),c['H_court_m_to_image']),points,atol=1e-6)
        self.assertEqual(len(c['keypoints_px']),14)

    def test_derived_points_follow_standard_court_dimensions(self):
        c=build_calibration(self.corners,1920,1080,'test')
        world=transform(c['keypoints_px'],c['H_image_to_court_m'])
        self.assertAlmostEqual(world[9,0]-world[8,0],8.23,places=5)
        self.assertAlmostEqual(23.77/2-world[8,1],6.40,places=5)
        self.assertAlmostEqual(world[10,1]-23.77/2,6.40,places=5)
        self.assertAlmostEqual(world[12,0],10.97/2,places=5)
        np.testing.assert_allclose(np.array(c['keypoints_px'])[[0,1,3,2]],self.corners,atol=1e-4)

    def test_bad_order_and_collinear_rejected(self):
        for bad in [[[1,1],[2,2],[3,3],[4,4]], [self.corners[i] for i in [0,2,1,3]],list(reversed(self.corners))]:
            with self.assertRaises(ValueError): validate_corners(bad,1920,1080)

    def test_crop_clamped_to_image_and_contains_court(self):
        c=build_calibration(self.corners,1920,1080,'test',padding=10)
        self.assertEqual(c['tracking_roi_xywh'],[0,0,1920,1080])
        with self.assertRaises(ValueError): build_calibration(self.corners,1920,1080,'test',[300,400,100,100])

    def test_invalid_coordinates_and_roi(self):
        for bad in [[-1,0,50,50],[0,0,2000,10],[0,0,0,5]]:
            with self.assertRaises(ValueError): validate_roi(bad,1920,1080)
        p=np.array(self.corners,float); p[0,0]=np.nan
        with self.assertRaises(ValueError): validate_corners(p,1920,1080)

    def test_crop_offset_preserves_original_pixel_coordinates(self):
        c=build_calibration(self.corners,1920,1080,'test',[100,200,1500,800])
        original=np.array([800,500]); offset=np.array(c['crop_to_image_offset_xy'])
        cropped=original-offset
        np.testing.assert_array_equal(cropped+offset,original)

if __name__=='__main__': unittest.main()
