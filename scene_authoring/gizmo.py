"""Translation gizmo geometry and ray interaction."""

import numpy as np

from scene_authoring import picking


AXIS_COLORS = {
    "x": (1.0, 0.2, 0.2, 1.0),
    "y": (0.2, 1.0, 0.2, 1.0),
    "z": (0.2, 0.5, 1.0, 1.0),
}


def axis_vector(axis: str):
  direction = np.zeros(3, dtype=np.float64)
  direction["xyz".index(axis)] = 1.0
  return direction


def axis_segments(origin, length: float = 0.75, basis=None):
  origin = np.asarray(origin, dtype=np.float64)
  basis = np.eye(3, dtype=np.float64) if basis is None else np.asarray(basis)
  return {
      axis: (origin, origin + basis @ direction * length)
      for axis, direction in (
          ("x", np.array([1.0, 0.0, 0.0])),
          ("y", np.array([0.0, 1.0, 0.0])),
          ("z", np.array([0.0, 0.0, 1.0])),
      )
  }


def closest_axis(
    ray_origin, ray_direction, origin, length=0.75, threshold=0.12, basis=None
):
  best_axis = None
  best_distance = threshold
  for axis, (start, end) in axis_segments(origin, length, basis=basis).items():
    distance, _ = picking.ray_segment_distance(
        ray_origin, ray_direction, start, end
    )
    if distance < best_distance:
      best_axis = axis
      best_distance = distance
  return best_axis


def closest_rotation_axis(
    ray_origin, ray_direction, origin, radius=0.58, threshold=0.10, basis=None
):
  best_axis = None
  best_distance = threshold
  for axis in ("x", "y", "z"):
    for start, end in rotation_ring_segments(
        origin, axis, radius=radius, segments=96, basis=basis
    ):
      distance, _ = picking.ray_segment_distance(
          ray_origin, ray_direction, start, end
      )
      if distance < best_distance:
        best_axis = axis
        best_distance = distance
  return best_axis


def drag_position(
    start_position, axis: str, start_ray, current_ray, basis=None
):
  result = np.asarray(start_position, dtype=np.float64).copy()
  direction = axis_vector(axis)
  if basis is not None:
    direction = np.asarray(basis) @ direction
  start_parameter = picking.ray_line_parameter(
      start_ray[0], start_ray[1], result, direction
  )
  current_parameter = picking.ray_line_parameter(
      current_ray[0], current_ray[1], result, direction
  )
  result += direction * (current_parameter - start_parameter)
  return tuple(float(v) for v in result)


def drag_delta(axis: str, start_ray, current_ray, origin, basis=None):
  direction = axis_vector(axis)
  if basis is not None:
    direction = np.asarray(basis) @ direction
  start_parameter = picking.ray_line_parameter(
      start_ray[0], start_ray[1], origin, direction
  )
  current_parameter = picking.ray_line_parameter(
      current_ray[0], current_ray[1], origin, direction
  )
  return float(current_parameter - start_parameter)


def scale_factor(
    axis, start_ray, current_ray, origin, reference, basis=None
):
  delta = drag_delta(axis, start_ray, current_ray, origin, basis=basis)
  return max(0.01, 1.0 + delta / max(float(reference), 0.05))


def rotation_angle(start_mouse, current_mouse):
  dx = float(current_mouse[0] - start_mouse[0])
  dy = float(current_mouse[1] - start_mouse[1])
  return (dx - dy) * 2.0 * np.pi


def rotation_angle_from_rays(
    start_ray, current_ray, origin, axis, fallback_mouse=None, basis=None
):
  normal = axis_vector(axis)
  if basis is not None:
    normal = np.asarray(basis) @ normal
  normal /= max(np.linalg.norm(normal), 1e-12)

  def plane_vector(ray):
    ray_origin, ray_direction = ray
    denominator = float(np.dot(ray_direction, normal))
    if abs(denominator) <= 1e-8:
      return None
    distance = float(np.dot(np.asarray(origin) - ray_origin, normal)) / denominator
    point = np.asarray(ray_origin) + distance * np.asarray(ray_direction)
    vector = point - np.asarray(origin)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-8 else None

  start_vector = plane_vector(start_ray)
  current_vector = plane_vector(current_ray)
  if start_vector is None or current_vector is None:
    if fallback_mouse is None:
      return 0.0
    return rotation_angle(fallback_mouse[0], fallback_mouse[1])
  sine = float(np.dot(normal, np.cross(start_vector, current_vector)))
  cosine = float(np.clip(np.dot(start_vector, current_vector), -1.0, 1.0))
  return float(np.arctan2(sine, cosine))


def rotation_ring_segments(origin, axis, radius=0.75, segments=96, basis=None):
  origin = np.asarray(origin, dtype=np.float64)
  basis = np.eye(3, dtype=np.float64) if basis is None else np.asarray(basis)
  local_origin = np.zeros(3, dtype=np.float64)
  normal = axis_vector(axis)
  helper = np.array([0.0, 0.0, 1.0])
  if abs(float(np.dot(normal, helper))) > 0.9:
    helper = np.array([0.0, 1.0, 0.0])
  first = np.cross(normal, helper)
  first /= max(np.linalg.norm(first), 1e-12)
  second = np.cross(normal, first)
  segments_out = []
  step = 2.0 * np.pi / segments
  overlap = step * 0.35
  for index in range(segments):
    a0 = step * index - overlap
    a1 = step * (index + 1) + overlap
    p0 = origin + basis @ (
        local_origin + radius * (np.cos(a0) * first + np.sin(a0) * second)
    )
    p1 = origin + basis @ (
        local_origin + radius * (np.cos(a1) * first + np.sin(a1) * second)
    )
    segments_out.append((p0, p1))
  return segments_out
