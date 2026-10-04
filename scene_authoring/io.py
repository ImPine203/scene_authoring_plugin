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


def edited_path(source_path: str) -> str:
  source = Path(source_path)
  return str(source.with_name(f"{source.stem}_edited{source.suffix or '.xml'}"))


def save_spec(spec: mujoco.MjSpec, source_path: str, target_path: str | None = None):
  path = Path(target_path or edited_path(source_path))
  if path.exists() and target_path is None:
    index = 2
    while True:
      candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
      if not candidate.exists():
        path = candidate
        break
      index += 1
  spec.to_file(str(path))
  return str(path)
