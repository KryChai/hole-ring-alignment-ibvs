import unittest

import numpy as np

from alignment import ServiceError, estimate_image_jacobian, fit_ring, ring_error
from demo import holes, simulate


class AlignmentTests(unittest.TestCase):
    def test_different_radii_share_a_zero_error_alignment(self):
        source = fit_ring(holes((320, 240), 52, yaw=4))
        target = fit_ring(holes((320, 240), 84, yaw=4)[::-1])
        error, _ = ring_error(source, target)
        np.testing.assert_allclose(error, np.zeros(4), atol=1e-9)

    def test_ambiguity_stops_cross_frame_association(self):
        source = fit_ring(holes((320, 240), 52))
        target = fit_ring(holes((320, 240), 84))
        _, order = ring_error(source, target)
        with self.assertRaisesRegex(ServiceError, 'ambiguous'):
            ring_error(source, fit_ring(holes((370, 240), 84)), previous=order)

    def test_dependent_probes_are_rejected(self):
        with self.assertRaisesRegex(ServiceError, 'independent'):
            estimate_image_jacobian(np.zeros(4), np.zeros((3, 4)), np.ones((3, 3)))

    def test_both_formulations_converge(self):
        for references in (1, 2):
            with self.subTest(references=references):
                result = simulate(references)
                self.assertEqual(result['stable_observations'], 3)
                self.assertLessEqual(result['final_center_error_px'], 1.0)
                self.assertLessEqual(result['final_angle_error_deg'], 0.2)
