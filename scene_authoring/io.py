"""MJCF loading and save helpers."""

from pathlib import Path

import mujoco


def load_spec(path: str):
  source = Path(path)
  if source.suffix.lower() not in (".xml", ".mjcf"):
    raise ValueError("Scene Authoring currently accepts MJCF XML files only")
  spec = mujoco.MjSpec.from_file(str(source))
  model = spec.compile()
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  return spec, model, data


def save_spec(spec: mujoco.MjSpec, source_path: str, target_path: str | None = None):
  # Saving without an explicit target updates the MJCF currently being
  # edited. A target supplied by the Save File dialog remains available for
  # users who want to write a separate copy.
  path = Path(target_path or source_path)
  spec.to_file(str(path))
  return str(path)
