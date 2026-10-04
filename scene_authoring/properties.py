"""MJCF property mapping shared by scene scan, editing, and the UI."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


# Keys are the MJCF-facing names shown in the Property panel. Values are the
# corresponding MjSpec attribute names. Pose/type/size/color/mass fields are
# kept in SceneNodeSnapshot's existing first-class fields.
EXTRA_ATTRIBUTE_SOURCES = {
    "body": {
        "class": "__classname__",
        "childclass": "childclass",
        "mocap": "mocap",
        "gravcomp": "gravcomp",
        "sleep": "sleep",
        "simple": "simple",
        "explicitinertial": "explicitinertial",
        "ipos": "ipos",
        "iquat": "iquat",
        "inertia": "inertia",
        "fullinertia": "fullinertia",
        "userdata": "userdata",
    },
    "geom": {
        "class": "__classname__",
        "contype": "contype",
        "conaffinity": "conaffinity",
        "condim": "condim",
        "group": "group",
        "priority": "priority",
        "material": "material",
        "mesh": "meshname",
        "hfield": "hfieldname",
        "friction": "friction",
        "density": "density",
        "solmix": "solmix",
        "solref": "solref",
        "solimp": "solimp",
        "margin": "margin",
        "gap": "gap",
        "surfacevel": "surfacevel",
        "adhesion": "adhesion",
        "fluidshape": "fluid_ellipsoid",
        "fluidcoef": "fluid_coefs",
        "fitscale": "fitscale",
        "typeinertia": "typeinertia",
        "fromto": "fromto",
        "userdata": "userdata",
    },
    "site": {
        "class": "__classname__",
        "group": "group",
        "material": "material",
        "mesh": "meshname",
        "fromto": "fromto",
        "userdata": "userdata",
    },
    "joint": {
        "class": "__classname__",
        "group": "group",
        "springdamper": "springdamper",
        "actfrclimited": "actfrclimited",
        "actfrcrange": "actfrcrange",
        "actgravcomp": "actgravcomp",
        "align": "align",
        "limited": "limited",
        "solref_limit": "solref_limit",
        "solimp_limit": "solimp_limit",
        "solref_friction": "solref_friction",
        "solimp_friction": "solimp_friction",
        "stiffness": "stiffness",
        "margin": "margin",
        "ref": "ref",
        "springref": "springref",
        "armature": "armature",
        "damping": "damping",
        "frictionloss": "frictionloss",
        "userdata": "userdata",
    },
}


def _plain_value(value: Any):
  if isinstance(value, (str, bool, int, float)):
    if isinstance(value, float) and not math.isfinite(value):
      return None
    return value
  if isinstance(value, np.generic):
    return _plain_value(value.item())
  try:
    iter(value)
  except TypeError:
    try:
      return int(value)
    except (TypeError, ValueError):
      return None
  try:
    values = tuple(value)
  except TypeError:
    return None
  result = []
  for item in values:
    item = _plain_value(item)
    if item is None:
      return None
    result.append(item)
  return tuple(result)


def read_extra_attributes(kind: str, node: Any) -> dict:
  result = {}
  for key, source in EXTRA_ATTRIBUTE_SOURCES.get(kind, {}).items():
    try:
      if source == "__classname__":
        value = _plain_value(node.classname.name)
      else:
        value = _plain_value(getattr(node, source))
    except (AttributeError, TypeError, ValueError):
      continue
    if value is not None:
      result[key] = value
  return result


def source_name(kind: str, key: str) -> str | None:
  return EXTRA_ATTRIBUTE_SOURCES.get(kind, {}).get(key)


def display_label(key: str) -> str:
  labels = {
      "childclass": "Child class",
      "conaffinity": "Conaffinity",
      "contype": "Contype",
      "explicitinertial": "Explicit inertial",
      "fluidcoef": "Fluid coef",
      "fluidshape": "Fluid shape",
      "fullinertia": "Full inertia",
      "fitscale": "Fit scale",
      "fromto": "From-to",
      "ipos": "Inertial pos",
      "iquat": "Inertial quat",
      "mesh": "Mesh",
      "hfield": "Heightfield",
      "mocap": "Mocap",
      "solimp": "Solimp",
      "solmix": "Solmix",
      "solref": "Solref",
      "solimp_limit": "Solimp limit",
      "solref_limit": "Solref limit",
      "solimp_friction": "Solimp friction",
      "solref_friction": "Solref friction",
      "springdamper": "Spring damper",
      "springref": "Spring ref",
      "surfacevel": "Surface velocity",
      "typeinertia": "Type inertia",
      "userdata": "User data",
  }
  return labels.get(key, key.replace("_", " ").capitalize())
