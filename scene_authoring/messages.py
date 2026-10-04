"""Reliable messages used by the Scene Authoring plugins."""

from dataclasses import dataclass, field
from typing import Literal

from mujoco.experimental.studio import messages as studio_messages

PrimitiveType = Literal["box", "sphere"]


@dataclass(frozen=True)
class SceneNodeSnapshot:
  """Serializable snapshot of one MJCF element in the body tree."""

  key: str
  kind: str
  name: str
  label: str
  parent_key: str | None
  child_keys: tuple[str, ...] = ()
  target_path: tuple[int, ...] = ()
  element_index: int = -1
  position: tuple[float, float, float] = (0.0, 0.0, 0.0)
  quaternion: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
  size: tuple[float, float, float] = (0.0, 0.0, 0.0)
  type_name: str = ""
  type_id: int = -1
  mass: float | None = None
  rgba: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0)
  axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
  range: tuple[float, float] = (0.0, 0.0)
  limited: str = ""
  damping: tuple[float, float, float] = (0.0, 0.0, 0.0)
  attributes: dict = field(default_factory=dict)


@dataclass(frozen=True)
class SceneScanEvent(studio_messages.Event):
  """Full scene property snapshot sent from the sim plugin to the viewer."""

  source_path: str
  nodes: tuple[SceneNodeSnapshot, ...]


@dataclass(frozen=True)
class CreateNodeEvent(studio_messages.Event):
  request_id: int
  kind: Literal["body", "geom", "site"]
  parent_path: tuple[int, ...]
  name: str
  values: dict


@dataclass(frozen=True)
class EditNodeEvent(studio_messages.Event):
  request_id: int
  kind: Literal["body", "geom", "site", "joint"]
  target_path: tuple[int, ...]
  element_index: int
  values: dict
  target_name: str = ""


@dataclass(frozen=True)
class CreatePrimitiveEvent(studio_messages.Event):
  request_id: int
  primitive_type: PrimitiveType
  name: str
  position: tuple[float, float, float]
  size: tuple[float, float, float]
  mass: float
  dynamic: bool


@dataclass(frozen=True)
class TransformObjectEvent(studio_messages.Event):
  request_id: int
  object_name: str
  position: tuple[float, float, float]
  quaternion: tuple[float, float, float, float]


@dataclass(frozen=True)
class DeleteObjectEvent(studio_messages.Event):
  request_id: int
  object_name: str


@dataclass(frozen=True)
class UndoEvent(studio_messages.Event):
  request_id: int


@dataclass(frozen=True)
class RedoEvent(studio_messages.Event):
  request_id: int


@dataclass(frozen=True)
class SaveSceneEvent(studio_messages.Event):
  request_id: int
  path: str


@dataclass(frozen=True)
class AuthoringResultEvent(studio_messages.Event):
  request_id: int
  operation: str
  success: bool
  object_name: str = ""
  message: str = ""
  model_changed: bool = False
