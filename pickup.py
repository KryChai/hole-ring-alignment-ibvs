"""Uncalibrated pickup: pixel ring checks and online XY Jacobian estimation."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import numpy as np

from alignment import ServiceError


class ConfigError(ValueError):
    """Invalid pickup policy."""


@dataclass
class PixelRing:
    center: np.ndarray
    holes: np.ndarray
    radius: float
    residual: float


def image_center(width, height):
    # Keep the same W/2, H/2 convention in control, export and downstream code.
    return np.array([width / 2.0, height / 2.0])


def validate_pick_uv(policy):
    for name in ("condition_max", "residual_max_px", "spacing_tolerance_deg",
                 "min_separation_px", "min_radius_px", "stability_px", "gain_error_scale"):
        value = getattr(policy, name)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ConfigError(f"第一阶段像素参数 {name} 必须是有限正数")
    learning = policy.jacobian_learning
    for name in ("min_move_mm", "min_shift_px", "max_relative_step"):
        value = getattr(learning, name)
        if not math.isfinite(value) or value <= 0:
            raise ConfigError(f"第一阶段雅可比学习参数 {name} 必须是有限正数")
    if learning.max_relative_step >= 1:
        raise ConfigError("第一阶段 JACOBIAN_UPDATE_MAX_RELATIVE_STEP 必须小于 1")
    matrix = np.asarray(policy.initial_jacobian, dtype=float)
    if (matrix.shape != (2, 2) or not np.isfinite(matrix).all()
            or np.linalg.cond(matrix) > policy.condition_max):
        raise ConfigError("第一阶段 INITIAL_JACOBIAN 必须是有限、非奇异的 2×2 矩阵")
    return matrix


def fit_pixel_ring(points, count, width, height, policy):
    """Fit the complete ring locally; never substitute the backend's centre."""
    pts = np.asarray(points, dtype=float)
    if count < 3 or pts.shape != (count, 2) or not np.isfinite(pts).all():
        raise ServiceError(f"圆环必须包含 {count} 个有效孔位，实际形状 {pts.shape}")
    if (np.any(pts < 0) or np.any(pts[:, 0] >= width)
            or np.any(pts[:, 1] >= height)):
        raise ServiceError("圆环孔位超出图像范围")
    distances = np.linalg.norm(pts[:, None] - pts[None, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    if np.min(distances) < policy.min_separation_px:
        raise ServiceError("圆环存在重复或距离过近的孔位")
    mean = pts.mean(axis=0)
    centered = pts - mean
    design = np.column_stack([2 * centered, np.ones(count)])
    coefficients, _, rank, _ = np.linalg.lstsq(design, np.sum(centered**2, axis=1), rcond=None)
    if rank < 3:
        raise ServiceError("孔位退化，不能拟合像素圆环")
    center = mean + coefficients[:2]
    radius_squared = float(coefficients[:2] @ coefficients[:2] + coefficients[2])
    if radius_squared <= 0 or not math.isfinite(radius_squared):
        raise ServiceError("像素圆环半径无效")
    radius = math.sqrt(radius_squared)
    if radius < policy.min_radius_px or not (0 <= center[0] < width and 0 <= center[1] < height):
        raise ServiceError("像素圆环过小或圆心超出图像范围")
    residual = float(np.max(np.abs(np.linalg.norm(pts - center, axis=1) - radius)))
    if residual > policy.residual_max_px:
        raise ServiceError(f"像素圆环拟合残差过大: {residual:.3f} px")
    angles = np.mod(np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0]), 2 * math.pi)
    order = np.argsort(angles)
    gaps = np.diff(np.r_[angles[order], angles[order][0] + 2 * math.pi])
    if np.max(np.abs(np.degrees(gaps) - 360.0 / count)) > policy.spacing_tolerance_deg:
        raise ServiceError("孔位不符合完整圆环的均匀角度分布")
    return PixelRing(center, pts[order], radius, residual)


def pick_control_step(error, jacobian, policy, tuning, *, diagnostics=None):
    matrix = np.asarray(jacobian, dtype=float) / policy.position_px
    normalized = np.asarray(error, dtype=float) / policy.position_px
    norm = float(np.linalg.norm(normalized))
    gain = policy.gain_min + (policy.gain_max - policy.gain_min) * norm / (norm + tuning.gain_error_scale)
    delta = -gain * np.linalg.solve(matrix.T @ matrix + policy.damping * np.eye(2),
                                    matrix.T @ normalized)
    length = float(np.linalg.norm(delta))
    if not np.isfinite(delta).all():
        raise ServiceError("第一阶段 XY 修正无效")
    if length > policy.max_step_mm:
        delta *= policy.max_step_mm / length
    if diagnostics is not None:
        diagnostics.update(effective_gain=float(gain),requested_step_mm=length,
                           step_limited=length>policy.max_step_mm)
    return delta


def update_pick_jacobian(jacobian, delta_xy, delta_uv, policy):
    """Apply one bounded Broyden update from a measured XY/UV pair."""
    learning = policy.jacobian_learning
    move_mm = float(np.linalg.norm(delta_xy))
    if move_mm < learning.min_move_mm:
        return jacobian, "small_motion"
    if np.linalg.norm(delta_uv) < learning.min_shift_px:
        return jacobian, "small_pixel_shift"

    # The rank-one correction reduces the current secant residual Δuv - J Δxy.
    correction = np.outer(delta_uv - jacobian @ delta_xy, delta_xy / move_mm) / move_mm
    correction_norm = float(np.linalg.norm(correction))
    if not math.isfinite(correction_norm):
        return jacobian, "invalid_update"
    # A correction smaller than σ_min cannot cross a singular matrix in this step.
    limit = learning.max_relative_step * np.linalg.svd(jacobian, compute_uv=False)[-1]
    limited = correction_norm > limit
    if limited:
        correction *= limit / correction_norm
    candidate = jacobian + correction
    if not np.isfinite(candidate).all() or np.linalg.cond(candidate) > policy.condition_max:
        return jacobian, "invalid_update"
    return candidate, "limited" if limited else "accepted"



@dataclass
class PickJacobianLearning:
    enabled: bool = False
    min_move_mm: float = 0.05
    min_shift_px: float = 0.1
    max_relative_step: float = 0.25


@dataclass
class PickUVPolicy:
    initial_jacobian: list[list[float]] = field(default_factory=lambda: [[-2., 0.], [0., 2.]])
    condition_max: float = 1e6
    residual_max_px: float = 2.0
    spacing_tolerance_deg: float = 8.0
    min_separation_px: float = 3.0
    min_radius_px: float = 5.0
    stability_px: float = 0.5
    gain_error_scale: float = 15.0
    jacobian_learning: PickJacobianLearning = field(default_factory=PickJacobianLearning)
