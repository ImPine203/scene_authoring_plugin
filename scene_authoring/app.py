"""External ViewerApp wrapper used for mouse arbitration."""

from mujoco.experimental.studio import messages
from mujoco.experimental.studio import sim
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

  def refresh_native_renderer_for_preview(self) -> bool:
    """Rebuilds native render resources after an in-place preview edit.

    Studio's Filament renderer uploads model meshes when it is initialized.
    The public Viewer API updates transforms every frame, but it does not
    expose a mesh-buffer update operation. Scene Authoring uses this hook for
    scale previews; it is a no-op for non-native viewers.
    """
    viewer = self.authoring_plugin.viewer
    if viewer is None:
      return False
    renderer = getattr(viewer, "_viewer", None)
    init_renderer = getattr(renderer, "InitRenderer", None)
    if init_renderer is None:
      return False
    init_renderer(viewer.model)
    return True

  @messages.handler(priority=messages.Priority.CRITICAL)
  def _on_model(self, event: messages.ModelEvent) -> bool:
    # ViewerApp._on_model resets StepControl to UNPAUSED. That default is
    # unsafe while Scene Authoring owns the viewport: recompiling a mesh can
    # otherwise advance a free body for one frame before the authoring panel
    # restores pause, making every geom on that body appear to move.
    keep_authoring_pause = self.authoring_plugin._authoring_pause_lock
    result = super()._on_model(event)
    if keep_authoring_pause:
      self.step_control_state.set_pause_state(
          sim.PauseState.NORMAL_PAUSED
      )
    return result

  @messages.handler(priority=messages.Priority.CRITICAL)
  def _on_update(self, _: messages.UpdateEvent) -> None:
    self.update()
