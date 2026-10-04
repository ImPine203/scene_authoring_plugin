"""MjSpec helpers for authored primitive objects."""

from dataclasses import dataclass

import mujoco


@dataclass(frozen=True)
class PrimitiveDescription:
  name: str
  primitive_type: str
  position: tuple[float, float, float]
  size: tuple[float, float, float]
  mass: float
  dynamic: bool


def next_name(spec: mujoco.MjSpec, primitive_type: str) -> str:
  prefix = f"authored_{primitive_type}_"
  index = 1
  while spec.body(f"{prefix}{index:03d}") is not None:
    index += 1
  return f"{prefix}{index:03d}"


def add_primitive(spec: mujoco.MjSpec, primitive: PrimitiveDescription) -> None:
  body = spec.worldbody.add_body(
      name=primitive.name, pos=list(primitive.position)
  )
  if primitive.dynamic:
    body.add_freejoint()
  if primitive.primitive_type == "box":
    geom_type = mujoco.mjtGeom.mjGEOM_BOX
    geom_size = list(primitive.size)
  elif primitive.primitive_type == "sphere":
    geom_type = mujoco.mjtGeom.mjGEOM_SPHERE
    geom_size = [primitive.size[0], 0.0, 0.0]
  else:
    raise ValueError(f"Unsupported primitive type: {primitive.primitive_type}")
  body.add_geom(
      type=geom_type,
      size=geom_size,
      mass=max(0.001, float(primitive.mass)),
      name=f"{primitive.name}_geom",
      rgba=[0.35, 0.65, 1.0, 1.0],
  )


def find_authored_body(spec: mujoco.MjSpec, name: str):
  body = spec.body(name)
  if body is None or not name.startswith("authored_"):
    return None
  return body
