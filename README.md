# MuJoCo Scene Authoring

An external Python plugin for the native Python MuJoCo Studio viewer. The
plugin adds scene authoring tools without modifying the MuJoCo source tree or
rebuilding Studio with plugin code.

## Requirements

- Python 3.10 or newer.
- `uv`.
- MuJoCo 3.13.1 or newer with the Python Studio bindings.
- A native desktop session with an OpenGL-capable graphics driver.

The project declares the MuJoCo and NumPy dependencies in `pyproject.toml`.
`uv sync` creates a local `.venv`, resolves the lockfile, installs the package
and its dependencies, and installs the `mujoco-scene-authoring` command.

## Install with uv

From the package root:

```sh
uv sync
```

Run the launcher through the managed environment:

```sh
uv run mujoco-scene-authoring path/to/model.xml
```

`uv run` is recommended because it automatically uses the environment created
by `uv sync`. The same command can also be run after activating `.venv`.

## Using a MuJoCo source checkout

The package is compatible with a source-built MuJoCo checkout when testing
Studio changes that are not available in the published wheel. Keep the
checkout outside this package and point the environment variables at it:

```sh
export MUJOCO_CHECKOUT=../mujoco
export PYTHONPATH="$MUJOCO_CHECKOUT/python${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$MUJOCO_CHECKOUT/build/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

uv sync
uv run mujoco-scene-authoring path/to/model.xml
```

The source checkout must have the Python Studio extensions built. The package
does not patch or modify that checkout.

## Features

- Box and sphere placement from the viewport.
- Mesh geom creation from a user-selected mesh file, added to the MJCF `<asset>` section.
- Camera creation as a child of any body from the Scene Tree.
- Static and dynamic primitives. Dynamic primitives receive a free joint.
- Geometry picking and fallback placement on the world `Z=0` plane.
- Scene Tree entries for bodies, geoms, sites, cameras and joints.
- Double-click a body, geom or site name in the Scene Tree to rename it.
- Double-click gizmo activation for body, geom and site nodes.
- XYZ translation, `R` rotation and `S` axis scaling for supported nodes.
- Body, primitive geom, site and mesh edits use a live viewer-model preview
  while moving, rotating or scaling.
- Property editing for MJCF attributes, including vector-valued `pos` and
  `quat` fields.
- Plugin-owned undo, redo and deletion.
- Save the edited scene with **Ctrl+S**. The Save File dialog defaults to the
  currently opened MJCF XML file, or can be used to choose another output path.

## Scene Authoring workflow

Selecting the **Scene Authoring** tab pauses the simulation. Scene Authoring
input and the gizmo are enabled only while the tab is active and the
simulation is paused. Press **Run** to resume simulation; press **Pause** to
enable authoring interaction again.

In the Scene Authoring panel:

- Right-click a body in the Scene Tree to open the Create menu.
- Create a child body, geom or site.
- Right-click a body, geom or site to open Scene Actions; choose **Create** or
  **Rename** when available.
- A new body receives a default spherical site with radius `0.01`.
- Geom and site creation dialogs expose the supported type and size fields.
- Mesh geom creation exposes a mesh file path and X/Y/Z mesh scale.
- Camera creation adds a fixed camera at the selected body's local origin,
  with editable pose, field of view and camera attributes in the Property
  panel.
- Select a node to inspect and edit its properties.
- Right-click a body, geom, site or camera and choose **Delete** to remove it.
- Select a node and press **Delete** to remove it. Use **Ctrl+Z** or **Ctrl+Y**
  to undo or redo the last authoring command.
- Pose and size edits on named bodies, geoms, sites and cameras update the
  paused runtime model with `mj_forward()` so the viewport responds without a
  full compile. The edit is written to `MjSpec` at commit time, and a full
  recompile is deferred until the user resumes the simulation. Structural
  edits and edits that cannot use this fast path still recompile immediately.

## Viewport gizmo

Double-click a body, geom or site in the viewport to activate the gizmo.
Double-click its name in the Scene Tree to rename it. Drag a local axis to
translate the selected node. Press `R` and drag an axis to
rotate its quaternion using the object's current orientation. Press `S` and
drag an axis to scale supported geom/site dimensions. Press `P` or `T` to
return to translation mode. Press `Esc` to cancel the current gizmo mode.

A body has no MJCF `size` attribute, so body scaling is not available. Body
translation and rotation are supported.

## Saving

Press **Ctrl+S** while the Scene Authoring tab is active to open the Save File
dialog. Its default target is the MJCF file currently being edited, so saving
overwrites that file. You can change the folder or filename in the dialog to
write a separate copy. The `.xml` suffix is added when omitted.

The current loader accepts MJCF XML paths. `.mjz`, binary model workflows,
and asset/include path relocation are outside the current scope. For a mesh
geom, enter a path to an OBJ, STL, PLY or other MuJoCo-supported mesh file in
the Create dialog. Relative paths are resolved against the directory of the
MJCF currently open in Studio. The mesh is added as a new `<asset><mesh>` and
the created geom references that asset.

## Architecture

The simulation-side plugin owns the authoritative `MjSpec`, `MjModel`,
`MjData` and command history. The viewer-side plugin owns UI state, input and
frame-local preview state. During a gizmo drag, the viewer temporarily updates
the live model arrays and calls `mj_forward()`; it restores those arrays when
the drag ends. Releasing the mouse sends one edit event to the simulation
plugin. Structural edits travel through MuJoCo Studio events and are applied
with `MjSpec.recompile(model, data)`. Pose and size edits use the faster
runtime path while paused and are recompiled once when Run is pressed.

After a successful recompile, the simulation plugin updates the viewer handle,
publishes a `ModelEvent`, and sends a fresh scene scan. Runtime IDs are
resolved again from stable names after each model update.

For scale previews, native Filament does not update uploaded mesh buffers from
`mesh_scale` alone. The viewer plugin therefore patches the selected mesh
vertices and normals around their visual center and refreshes native render
resources while the drag is active. The next authoritative compile still
rebuilds the final collision and asset data.

## Development commands

Compile-check the Python package without creating test files:

```sh
uv run python -m py_compile run_studio.py scene_authoring/*.py
```

Launch with custom native viewer settings:

```sh
uv run mujoco-scene-authoring path/to/model.xml --gfx opengl --width 1400 --height 900
```
