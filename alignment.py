"""Pixel-space hole-ring alignment extracted from the four- and six-point controllers."""

from dataclasses import dataclass
import math

import numpy as np


class ServiceError(ValueError):
    """An observation or controller state is invalid for alignment."""


@dataclass
class UVRing:
    """图像像素坐标中的圆环，不包含任何工作平面坐标。"""

    center: np.ndarray
    holes: np.ndarray
    radius: float
    residual: float


def _wrap_array(values: np.ndarray) -> np.ndarray:
    return (values + math.pi) % (2.0 * math.pi) - math.pi


def _fit_circle_uv(points: np.ndarray) -> tuple[np.ndarray, float, float]:
    """仅在 UV 图像坐标中做代数圆拟合，返回中心、半径和最大径向残差。"""
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3 or not np.isfinite(pts).all():
        raise ServiceError("图像孔位不足或包含无效 UV 坐标，不能拟合圆心")
    mean = pts.mean(axis=0)
    shifted = pts - mean
    design = np.column_stack((2.0 * shifted, np.ones(len(pts))))
    coeff, _, rank, _ = np.linalg.lstsq(design, np.sum(shifted * shifted, axis=1), rcond=None)
    if rank < 3:
        raise ServiceError("图像孔位近似共线，不能拟合 UV 圆心")
    squared_radius = float(np.dot(coeff[:2], coeff[:2]) + coeff[2])
    if not math.isfinite(squared_radius) or squared_radius <= 0:
        raise ServiceError("图像 UV 圆拟合半径无效")
    center = mean + coeff[:2]
    radius = math.sqrt(squared_radius)
    residual = float(np.max(np.abs(np.linalg.norm(pts - center, axis=1) - radius)))
    return center, radius, residual


@dataclass(frozen=True)
class ServoPolicy:
    position_px: float = 1.0
    angle_deg: float = 0.2
    max_step_mm: float = 1.0
    max_step_deg: float = 1.0
    gain_min: float = 0.08
    gain_max: float = 0.30
    gain_error_scale: float = 15.0
    damping: float = 0.01
    condition_max: float = 1e6
    association_fraction: float = 0.5
    min_update_norm2: float = 1e-8

    def action_units(self):
        return np.array([self.max_step_mm, self.max_step_mm, self.max_step_deg])


def fit_ring(points):
    """Fit the pixel center and put detected holes in cyclic order."""
    holes = np.asarray(points, dtype=float)
    center, radius, residual = _fit_circle_uv(holes)
    if len(np.unique(holes, axis=0)) != len(holes):
        raise ServiceError("duplicate holes in the ring")
    vectors = holes - center
    order = np.argsort(np.arctan2(vectors[:, 1], vectors[:, 0]))
    return UVRing(center, holes[order], radius, residual)


def ring_error(source, target, *, previous=None, reference_count=2, policy=ServoPolicy()):
    """Return center and wrapped direction errors, plus target holes in correspondence order."""
    count = len(source.holes)
    if len(target.holes) != count or count < 3 or reference_count not in (1, 2):
        raise ServiceError("two complete rings with the same hole count are required")
    indices = [0] if reference_count == 1 else [0, count // 2]
    source_directions = source.holes[indices] - source.center
    candidates = [np.roll(target.holes, -shift, axis=0) for shift in range(count)]

    def direction_errors(candidate):
        target_directions = candidate[indices] - target.center
        if (np.any(np.linalg.norm(source_directions, axis=1) < 1e-8)
                or np.any(np.linalg.norm(target_directions, axis=1) < 1e-8)):
            raise ServiceError("reference hole overlaps the fitted center")
        return _wrap_array(np.arctan2(target_directions[:, 1], target_directions[:, 0])
                           - np.arctan2(source_directions[:, 1], source_directions[:, 0]))

    if previous is None:
        ordered = min(candidates, key=lambda candidate: np.linalg.norm(direction_errors(candidate)))
    else:
        previous = np.asarray(previous, dtype=float)
        if previous.shape != target.holes.shape or not np.isfinite(previous).all():
            raise ServiceError("invalid previous target correspondence")
        ordered = min(candidates, key=lambda candidate: np.linalg.norm(candidate - previous, axis=1).sum())
        spacing = np.min(np.linalg.norm(target.holes - np.roll(target.holes, -1, axis=0), axis=1))
        if np.max(np.linalg.norm(ordered - previous, axis=1)) >= policy.association_fraction * spacing:
            raise ServiceError("ambiguous cross-frame hole correspondence")
    return np.r_[source.center - target.center, direction_errors(ordered)], ordered


def normalize(error, policy):
    error = np.asarray(error, dtype=float)
    scale = [policy.position_px] * 2 + [math.radians(policy.angle_deg)] * (len(error) - 2)
    return error / scale


def feature_delta(after, before):
    delta = np.asarray(after, dtype=float) - before
    delta[2:] = _wrap_array(delta[2:])
    return delta


def check_jacobian(jacobian, policy):
    if (jacobian.shape not in ((3, 3), (4, 3)) or not np.isfinite(jacobian).all()
            or np.linalg.matrix_rank(jacobian) < 3 or np.linalg.cond(jacobian) > policy.condition_max):
        raise ServiceError("degenerate image Jacobian")


def estimate_image_jacobian(baseline, probe_errors, measured_actions, policy=ServoPolicy()):
    """Estimate the normalized 3- or 4-by-3 Jacobian from three independent probes.

    Each row of measured_actions contains the actual X, Y and yaw change in mm/mm/deg.
    Each feature sample is compared with the same restored baseline pose.
    """
    actions = np.asarray(measured_actions, dtype=float) / policy.action_units()
    if actions.shape != (3, 3) or not np.isfinite(actions).all() or np.linalg.matrix_rank(actions) < 3:
        raise ServiceError("three independent planar probes are required")
    changes = np.array([normalize(feature_delta(error, baseline), policy) for error in probe_errors])
    if len(changes) != 3:
        raise ServiceError("one feature observation is required per probe")
    jacobian = np.linalg.solve(actions, changes).T
    check_jacobian(jacobian, policy)
    return jacobian


def solve_step(jacobian, error, policy=ServoPolicy()):
    """Calculate bounded X/Y/yaw motion with damped least squares and scheduled gain."""
    check_jacobian(jacobian, policy)
    normalized = normalize(error, policy)
    if not np.isfinite(normalized).all():
        raise ServiceError("nonfinite feature error")
    norm = np.linalg.norm(normalized)
    gain = policy.gain_min + (policy.gain_max - policy.gain_min) * norm / (norm + policy.gain_error_scale)
    normal_matrix = jacobian.T @ jacobian + policy.damping * np.eye(3)
    step = -gain * np.linalg.solve(normal_matrix, jacobian.T @ normalized) * policy.action_units()
    xy_norm = np.linalg.norm(step[:2])
    if xy_norm > policy.max_step_mm:
        step[:2] *= policy.max_step_mm / xy_norm
    step[2] = np.clip(step[2], -policy.max_step_deg, policy.max_step_deg)
    return step


def update_image_jacobian(jacobian, before, after, measured_action, policy=ServoPolicy()):
    """Update from an observed planar action and its wrapped feature change."""
    action = np.asarray(measured_action, dtype=float) / policy.action_units()
    if action.shape != (3,) or not np.isfinite(action).all():
        raise ServiceError("invalid measured planar action")
    denominator = float(action @ action)
    if denominator <= policy.min_update_norm2:
        return jacobian
    change = normalize(feature_delta(after, before), policy)
    candidate = jacobian + np.outer(change - jacobian @ action, action) / denominator
    check_jacobian(candidate, policy)
    return candidate


def converged(error, policy=ServoPolicy()):
    return (np.linalg.norm(error[:2]) <= policy.position_px
            and np.max(np.abs(np.degrees(error[2:]))) <= policy.angle_deg)
