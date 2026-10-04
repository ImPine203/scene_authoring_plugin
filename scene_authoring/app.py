"""External ViewerApp wrapper used for mouse arbitration."""

from mujoco.experimental.studio import messages
from mujoco.experimental.studio import viewer_app


class SceneAuthoringViewerApp(viewer_app.ViewerApp):
  """Runs the standard app while allowing authoring to own active drags."""

  def __init__(self, authoring_plugin, config=None):
    super().__init__(config)
    self.authoring_plugin = authoring_plugin

  def handle_mouse_events(self) -> None:
    if self.authoring_plugin.handle_mouse_input():
      return
    super().handle_mouse_events()

  def handle_keyboard_events(self) -> None:
    if self.authoring_plugin.handle_keyboard_input():
      return
    super().handle_keyboard_events()

  @messages.handler(priority=messages.Priority.CRITICAL)
  def _on_update(self, _: messages.UpdateEvent) -> None:
    self.update()
