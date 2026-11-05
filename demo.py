"""Synthetic four- or six-point XY/yaw visual servoing with unequal ring radii."""

import argparse
import json

import numpy as np

from alignment import (converged, estimate_image_jacobian, fit_ring, ring_error,
                       solve_step, update_image_jacobian)


def holes(center, radius, yaw=0.0, count=12):
    angles = np.arange(count) * (2 * np.pi / count) + np.deg2rad(yaw)
    return np.asarray(center) + radius * np.column_stack((np.cos(angles), np.sin(angles)))


def simulate(reference_count=2):
    source = fit_ring(holes((320, 240), 52))

    def observe(pose, previous=None):
        x, y, yaw = pose
        center = np.array([329 + 2.2*x - .25*y, 234 + .15*x + 1.8*y])
        target = fit_ring(holes(center, 84, 7 + .9*yaw))
        return ring_error(source, target, previous=previous, reference_count=reference_count)

    pose = np.zeros(3)
    error, order = observe(pose)
    initial = error.copy()
    probes = .5 * np.eye(3)
    jacobian = estimate_image_jacobian(error, [observe(pose + step, order)[0] for step in probes], probes)
    stable = 0
    for iteration in range(200):
        if converged(error):
            stable += 1
            if stable == 3:
                return {"features": 2 + reference_count, "iterations": iteration,
                        "initial_center_error_px": float(np.linalg.norm(initial[:2])),
                        "final_center_error_px": float(np.linalg.norm(error[:2])),
                        "final_angle_error_deg": float(np.max(np.abs(np.degrees(error[2:])))),
                        "stable_observations": stable}
            error, order = observe(pose, order)
            continue
        stable = 0
        action = solve_step(jacobian, error)
        pose += action
        next_error, order = observe(pose, order)
        jacobian = update_image_jacobian(jacobian, error, next_error, action)
        error = next_error
    raise RuntimeError("synthetic alignment did not converge")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--references', type=int, choices=(1, 2), default=2)
    print(json.dumps(simulate(parser.parse_args().references), indent=2))
