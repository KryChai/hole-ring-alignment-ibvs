import unittest

import numpy as np
import pytest

from pickup import PickUVPolicy, fit_pixel_ring

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


def circle(center=(400., 300.), radius=100.):
    angles = np.arange(12) * np.pi / 6
    return np.array(center) + radius * np.column_stack([np.cos(angles), np.sin(angles)])


def test_complete_pixel_circle_is_fit_invariant_to_detection_order():
    points = circle((405., 290.))[np.random.default_rng(2).permutation(12)]
    ring = fit_pixel_ring(points, 12, 800, 600, PickUVPolicy())
    np.testing.assert_allclose(ring.center, [405., 290.], atol=1e-8)
    assert ring.radius == pytest.approx(100.) and ring.residual < 1e-8


@pytest.mark.parametrize("kind", ["missing", "extra", "duplicate", "nan", "outside", "collinear", "radial", "spacing"])
def test_bad_ring_observations_are_rejected(kind):
    points = circle()
    if kind == "missing": points = points[:-1]
    if kind == "extra": points = np.vstack([points, [420, 320]])
    if kind == "duplicate": points[1] = points[0]
    if kind == "nan": points[1, 0] = np.nan
    if kind == "outside": points[1, 0] = 801
    if kind == "collinear": points = np.column_stack([np.arange(12) * 10 + 200, np.ones(12) * 300])
    if kind == "radial": points[1] += [30, 0]
    if kind == "spacing":
        angles = np.radians([0, 10, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330])
        points = [400, 300] + 100 * np.column_stack([np.cos(angles), np.sin(angles)])
    with pytest.raises(ServiceError):
        fit_pixel_ring(points, 12, 800, 600, PickUVPolicy())
