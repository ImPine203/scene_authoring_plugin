from dataclasses import dataclass, field
from enum import Enum, auto


class AuthoringMode(Enum):
  SELECT = auto()
  PLACE_BOX = auto()
  PLACE_SPHERE = auto()


@dataclass
class AuthoringState:
  mode: AuthoringMode = AuthoringMode.SELECT
  dynamic: bool = False
  size: list[float] = field(default_factory=lambda: [0.25, 0.25, 0.25])
  mass: float = 1.0
  selected_name: str | None = None
  selected_body_id: int = -1
  inspected_node_key: str | None = None
  scene_source_path: str = ""
  preview_position: tuple[float, float, float] | None = None
  preview_quaternion: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
  hovered_axis: str | None = None
  dragging_axis: str | None = None
  drag_start_position: tuple[float, float, float] | None = None
  pending_request_id: int | None = None
  paused_for_authoring: bool = False
  status: str = "Ready"

  @property
  def placing(self) -> bool:
    return self.mode in (AuthoringMode.PLACE_BOX, AuthoringMode.PLACE_SPHERE)
