"""Launch MuJoCo Studio with the external Scene Authoring plugin."""

import argparse

from mujoco.experimental.studio import launch_passive
from mujoco.experimental.studio import messages as studio_messages
from mujoco.experimental.studio import step_control
from mujoco.experimental.studio import viewer_protocol

from scene_authoring.app import SceneAuthoringViewerApp
from scene_authoring.io import load_spec
from scene_authoring.sim_plugin import SceneAuthoringSimPlugin
from scene_authoring.viewer_plugin import SceneAuthoringViewerPlugin


def main(argv=None):
  parser = argparse.ArgumentParser()
  parser.add_argument("model", help="Path to an MJCF XML file")
  parser.add_argument("--gfx", default="", choices=("", "opengl", "webgl", "web"))
  parser.add_argument("--width", type=int, default=1400)
  parser.add_argument("--height", type=int, default=900)
  parser.add_argument("--port", type=int, default=0)
  args = parser.parse_args(argv)

  spec, model, data = load_spec(args.model)
  viewer_plugin = SceneAuthoringViewerPlugin()
  app = SceneAuthoringViewerApp(viewer_plugin)
  sim_plugin = SceneAuthoringSimPlugin(spec, model, data, args.model)
  config = viewer_protocol.ViewerConfig(
      title="MuJoCo Studio - Scene Authoring",
      width=args.width,
      height=args.height,
      gfx=args.gfx,
      http_port=args.port,
  )

  with launch_passive.launch_passive(
      config,
      viewer_plugins=[app, viewer_plugin],
      sim_plugins=[sim_plugin, step_control.StepControl()],
  ) as handle:
    sim_plugin.attach_handle(handle)
    handle.send_to_viewer(
        studio_messages.ModelEvent(model=model, path=args.model)
    )
    sim_plugin.publish_scene_scan()
    while handle.is_running():
      model, data = handle.sync(model, data)


if __name__ == "__main__":
  main()
