"""Viewport ray and placement helpers."""

import mujoco
import numpy as np
from mujoco.experimental.studio import ux
from mujoco.experimental.dear_imgui import dear_imgui as imgui


def viewport_coordinates() -> tuple[float, float, float] | None:
  io = imgui.GetIO()
  if io.DisplaySize.x <= 0 or io.DisplaySize.y <= 0:
    return None
  return (
      float(io.MousePos.x / io.DisplaySize.x),
      float(io.MousePos.y / io.DisplaySize.y),
      float(io.DisplaySize.x / io.DisplaySize.y),
  )


def pick_point(viewer, x: float, y: float, aspect_ratio: float):
  """Returns a world point from geometry or the world Z=0 plane."""
  picked = ux.Pick(
      viewer.model, viewer.data, viewer.camera, x, y, aspect_ratio,
      viewer.vis_options,
  )
  if picked.dist >= 0:
    return tuple(float(v) for v in picked.point), picked
  origin, direction = make_ray(viewer, x, y, aspect_ratio)
  if abs(direction[2]) < 1e-9:
    return None, picked
  distance = -origin[2] / direction[2]
  if distance < 0:
    return None, picked
  point = origin + distance * direction
  return tuple(float(v) for v in point), picked


def make_ray(viewer, x: float, y: float, aspect_ratio: float):
  forward = np.zeros(3, dtype=np.float64)
  up = np.zeros(3, dtype=np.float64)
  right = np.zeros(3, dtype=np.float64)
  origin = np.zeros(3, dtype=np.float64)
  mujoco.mjv_cameraFrame(origin, forward, up, right, viewer.data, viewer.camera)
  zver = np.zeros(2, dtype=np.float32)
  zhor = np.zeros(2, dtype=np.float32)
  zclip = np.zeros(2, dtype=np.float32)
  mujoco.mjv_cameraFrustum(zver, zhor, zclip, viewer.model, viewer.camera)
  half_width = 0.5 * aspect_ratio * float(zver[0] + zver[1])
  frustum_center = float((zhor[1] - zhor[0]) / 2.0)
  d_up = -float(zver[0]) + (1.0 - y) * float(zver[0] + zver[1])
  d_right = frustum_center + (2.0 * x - 1.0) * half_width
  if viewer.camera.orthographic:
    ray = forward.copy()
    origin = origin + up * d_up + right * d_right
  else:
    ray = forward * float(zclip[0]) + up * d_up + right * d_right
    ray /= max(np.linalg.norm(ray), 1e-12)
  return origin, ray


def ray_segment_distance(ray_origin, ray_direction, segment_start, segment_end):
  ro = np.asarray(ray_origin, dtype=np.float64)
  rd = np.asarray(ray_direction, dtype=np.float64)
  a = np.asarray(segment_start, dtype=np.float64)
  b = np.asarray(segment_end, dtype=np.float64)
  u = rd / max(np.linalg.norm(rd), 1e-12)
  v = b - a
  w = ro - a
  uu = float(np.dot(u, u))
  uv = float(np.dot(u, v))
  vv = float(np.dot(v, v))
  uw = float(np.dot(u, w))
  vw = float(np.dot(v, w))
  denominator = uu * vv - uv * uv
  if vv <= 1e-12:
    s = max(0.0, -uw / uu)
    return float(np.linalg.norm(ro + s * u - a)), s
  if abs(denominator) <= 1e-12:
    s = max(0.0, -uw / uu)
    t = min(1.0, max(0.0, vw / vv))
  else:
    s = (uv * vw - vv * uw) / denominator
    t = (uu * vw - uv * uw) / denominator
    if s < 0.0:
      s = 0.0
      t = min(1.0, max(0.0, vw / vv))
    else:
      t = min(1.0, max(0.0, t))
  return float(np.linalg.norm(ro + s * u - (a + t * v))), s


def ray_point_distance(ray_origin, ray_direction, point):
  ro = np.asarray(ray_origin, dtype=np.float64)
  rd = np.asarray(ray_direction, dtype=np.float64)
  target = np.asarray(point, dtype=np.float64)
  rd /= max(np.linalg.norm(rd), 1e-12)
  parameter = float(np.dot(target - ro, rd))
  closest = ro + max(parameter, 0.0) * rd
  return float(np.linalg.norm(target - closest)), parameter


def ray_line_parameter(ray_origin, ray_direction, line_origin, line_direction):
  ro = np.asarray(ray_origin, dtype=np.float64)
  rd = np.asarray(ray_direction, dtype=np.float64)
  lo = np.asarray(line_origin, dtype=np.float64)
  ld = np.asarray(line_direction, dtype=np.float64)
  rd /= max(np.linalg.norm(rd), 1e-12)
  ld /= max(np.linalg.norm(ld), 1e-12)
  w = ro - lo
  a = float(np.dot(rd, rd))
  b = float(np.dot(rd, ld))
  c = float(np.dot(ld, ld))
  d = float(np.dot(rd, w))
  e = float(np.dot(ld, w))
  denominator = a * c - b * b
  if abs(denominator) <= 1e-12:
    return float(e / c)
  return float((a * e - b * d) / denominator)
