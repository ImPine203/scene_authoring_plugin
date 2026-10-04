# MuJoCo Studio API audit

This package targets the Python native Studio APIs in the MuJoCo source
checkout used during implementation. The checkout was kept as a sibling
working directory (`../mujoco`) rather than being modified by this package.

## Observed environment

- Source commit observed during implementation: `5ceb72b1`.
- The system MuJoCo package available before the source build was 3.6.0 and did
  not expose `mujoco.experimental.studio`.
- The source checkout exposed the native Studio C++ sources and Python bindings
  under `python/mujoco/experimental/studio`.

## APIs used

- `launch_passive.launch_passive(viewer_plugins=..., sim_plugins=...)`.
- `messages.Event` for reliable ordered commands and results.
- `ViewerInitEvent`, `ViewerAppInitEvent`, `ModelEvent`, `BuildGuiEvent` and the
  Studio update loop.
- `Viewer.model`, `Viewer.data`, `Viewer.camera`, `Viewer.vis_options` and
  `Viewer.extra_geoms`.
- `ux.Pick` for geometry picking.
- `mjv_cameraFrame` and `mjv_cameraFrustum` for fallback viewport rays and
  camera-scaled gizmos.
- `MjSpec.from_file`, `MjSpec.recompile(model, data)`, `MjSpec.delete` and
  `MjSpec.to_file`.

## Ownership and synchronization

`SceneAuthoringSimPlugin` owns the authoritative `MjSpec`, model, data and
history. Viewer plugins send reliable edit events and render temporary
geometry. After a structural edit, the simulation plugin recompiles with the
existing model and data, updates the `ViewerHandle`, sends a `ModelEvent`, and
publishes a fresh scene scan.

Runtime body, geom and site IDs are resolved again from stable names after
each model update. The plugin does not persist runtime IDs across recompiles.

## Known constraints

- The implementation targets the Python native Studio viewer.
- The package accepts MJCF XML paths. `.mjz` and binary model workflows are not
  enabled by the package loader.
- The external Python plugin registry does not expose the C++ Studio plugin
  registration API. This package therefore uses Studio's Python event/plugin
  layer and an external `ViewerApp` subclass for mouse arbitration.
- No automated test suite is included by request. Verification is manual.

## Build and smoke observations

- The native MuJoCo Studio target was built in a temporary directory with
  `MUJOCO_BUILD_STUDIO=ON`; the MuJoCo checkout itself was not edited.
- Python extensions were built from the source checkout into a wheel for
  MuJoCo 3.13.1. Imports of `mujoco.experimental.studio`, Dear ImGui and this
  package succeeded.
- The native launcher opened with the OpenGL backend and initialized the
  available graphics renderer without a startup exception.
- Interactive placement, gizmo dragging, undo/redo and save require a person
  to operate the visible window. No automated test suite was created or run.
