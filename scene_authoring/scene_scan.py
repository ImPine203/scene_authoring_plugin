"""Read-only snapshots of the currently loaded MjSpec body tree."""

from __future__ import annotations

import math
from typing import Any

import mujoco

from scene_authoring.messages import SceneNodeSnapshot
from scene_authoring.properties import read_extra_attributes


def _vec(value: Any, length: int, default: tuple[float, ...]) -> tuple[float, ...]:
  if value is None:
    return default
  try:
    values = tuple(float(item) for item in value)
  except (TypeError, ValueError):
    return default
  if len(values) != length:
    return default
  return values


def _optional_float(value: Any) -> float | None:
  try:
    result = float(value)
  except (TypeError, ValueError):
    return None
  return result if math.isfinite(result) else None


def _enum_name(value: Any) -> str:
  text = str(value)
  return text.rsplit(".", 1)[-1]


def _display_name(kind: str, name: str, index: int) -> str:
  return f"{kind.capitalize()}: {name or f'<unnamed #{index}>'}"


def scan_spec(spec: mujoco.MjSpec) -> tuple[SceneNodeSnapshot, ...]:
  """Returns a stable, read-only snapshot of the spec's body tree.

  The traversal starts at worldbody and follows direct child bodies, joints,
  geoms and sites. Each node has a local transform and the
  type-specific fields that are meaningful for that MJCF element.
  """
  nodes: list[SceneNodeSnapshot] = []
  counter = 0

  def visit_body(body: Any, parent_key: str | None, index: int, body_path: tuple[int, ...]) -> str:
    nonlocal counter
    counter += 1
    key = f"body:{counter}"
    name = str(getattr(body, "name", "") or "")
    child_keys: list[str] = []

    body_node_index = len(nodes)
    nodes.append(SceneNodeSnapshot(
        key=key,
        kind="body",
        name=name,
        label=_display_name("body", name, index),
        parent_key=parent_key,
        target_path=body_path,
        position=_vec(getattr(body, "pos", None), 3, (0.0, 0.0, 0.0)),
        quaternion=_vec(
            getattr(body, "quat", None), 4, (1.0, 0.0, 0.0, 0.0)
        ),
        mass=_optional_float(getattr(body, "mass", None)),
        attributes=read_extra_attributes("body", body),
    ))

    for child_index, child in enumerate(getattr(body, "bodies", ())):
      child_keys.append(visit_body(child, key, child_index, body_path + (child_index,)))

    for joint_index, joint in enumerate(getattr(body, "joints", ())):
      counter += 1
      joint_key = f"joint:{counter}"
      joint_name = str(getattr(joint, "name", "") or "")
      child_keys.append(joint_key)
      nodes.append(SceneNodeSnapshot(
          key=joint_key,
          kind="joint",
          name=joint_name,
          label=_display_name("joint", joint_name, joint_index),
          parent_key=key,
          target_path=body_path,
          element_index=joint_index,
          position=_vec(getattr(joint, "pos", None), 3, (0.0, 0.0, 0.0)),
          type_name=_enum_name(getattr(joint, "type", "")),
          type_id=int(getattr(joint, "type", -1)),
          axis=_vec(getattr(joint, "axis", None), 3, (0.0, 0.0, 1.0)),
          range=_vec(getattr(joint, "range", None), 2, (0.0, 0.0)),
          limited=_enum_name(getattr(joint, "limited", "")),
          damping=_vec(
              getattr(joint, "damping", None), 3, (0.0, 0.0, 0.0)
          ),
          attributes=read_extra_attributes("joint", joint),
      ))

    for geom_index, geom in enumerate(getattr(body, "geoms", ())):
      counter += 1
      geom_key = f"geom:{counter}"
      geom_name = str(getattr(geom, "name", "") or "")
      child_keys.append(geom_key)
      nodes.append(SceneNodeSnapshot(
          key=geom_key,
          kind="geom",
          name=geom_name,
          label=_display_name("geom", geom_name, geom_index),
          parent_key=key,
          target_path=body_path,
          element_index=geom_index,
          position=_vec(getattr(geom, "pos", None), 3, (0.0, 0.0, 0.0)),
          quaternion=_vec(
              getattr(geom, "quat", None), 4, (1.0, 0.0, 0.0, 0.0)
          ),
          size=_vec(getattr(geom, "size", None), 3, (0.0, 0.0, 0.0)),
          type_name=_enum_name(getattr(geom, "type", "")),
          type_id=int(getattr(geom, "type", -1)),
          mass=_optional_float(getattr(geom, "mass", None)),
          rgba=_vec(
              getattr(geom, "rgba", None), 4, (0.5, 0.5, 0.5, 1.0)
          ),
          attributes=read_extra_attributes("geom", geom),
      ))

    for site_index, site in enumerate(getattr(body, "sites", ())):
      counter += 1
      site_key = f"site:{counter}"
      site_name = str(getattr(site, "name", "") or "")
      child_keys.append(site_key)
      nodes.append(SceneNodeSnapshot(
          key=site_key,
          kind="site",
          name=site_name,
          label=_display_name("site", site_name, site_index),
          parent_key=key,
          target_path=body_path,
          element_index=site_index,
          position=_vec(getattr(site, "pos", None), 3, (0.0, 0.0, 0.0)),
          quaternion=_vec(
              getattr(site, "quat", None), 4, (1.0, 0.0, 0.0, 0.0)
          ),
          size=_vec(getattr(site, "size", None), 3, (0.0, 0.0, 0.0)),
          type_name=_enum_name(getattr(site, "type", "")),
          type_id=int(getattr(site, "type", -1)),
          rgba=_vec(
              getattr(site, "rgba", None), 4, (0.5, 0.5, 0.5, 1.0)
          ),
          attributes=read_extra_attributes("site", site),
      ))

    nodes[body_node_index] = SceneNodeSnapshot(
        **{
            **nodes[body_node_index].__dict__,
            "child_keys": tuple(child_keys),
        }
    )
    return key

  visit_body(spec.worldbody, None, 0, ())
  return tuple(nodes)
