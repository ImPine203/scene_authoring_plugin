"""Simulation-side authoritative scene editing plugin."""

from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np
from mujoco.experimental.studio import messages as studio_messages
from mujoco.experimental.studio import sim

from scene_authoring import messages
from scene_authoring.history import Command, CommandHistory
from scene_authoring.io import load_spec, save_spec
from scene_authoring.properties import read_extra_attributes, source_name
from scene_authoring.scene_scan import scan_spec
from scene_authoring.primitives import (
    PrimitiveDescription,
    add_primitive,
    find_authored_body,
)


class SceneAuthoringSimPlugin:
  """Owns MjSpec and performs all structural edits."""

  def __init__(self, spec, model, data, source_path: str):
    self.spec = spec
    self.model = model
    self.data = data
    self.source_path = source_path
    self.handle = None
    self.history = CommandHistory()
    self.authored: dict[str, PrimitiveDescription] = {}
    self._model_dirty = False

  def attach_handle(self, handle):
    self.handle = handle

  def publish_scene_scan(self):
    if self.handle is None:
      return
    self.handle.send_to_viewer(messages.SceneScanEvent(
        source_path=self.source_path,
        nodes=scan_spec(self.spec),
    ))

  def _result(self, event, operation, success, **kwargs):
    if self.handle is not None:
      self.handle.send_to_viewer(messages.AuthoringResultEvent(
          request_id=event.request_id,
          operation=operation,
          success=success,
          **kwargs,
      ))

  def _publish_model(self):
    if self.handle is None:
      raise RuntimeError("SceneAuthoringSimPlugin is not attached to a handle")
    self.handle.model = self.model
    self.handle.data = self.data
    self.handle.send_to_viewer(
        studio_messages.ModelEvent(model=self.model, path=self.source_path)
    )
    self.publish_scene_scan()

  def _recompile(self):
    state_sig = mujoco.mjtState.mjSTATE_INTEGRATION
    state = None
    if self.model is not None and self.data is not None:
      state = np.empty(
          mujoco.mj_stateSize(self.model, state_sig), dtype=np.float64
      )
      mujoco.mj_getState(self.model, self.data, state, state_sig)
    self.model, self.data = self.spec.recompile(self.model, self.data)
    if state is not None:
      new_state_size = mujoco.mj_stateSize(self.model, state_sig)
      if len(state) == new_state_size:
        mujoco.mj_setState(self.model, self.data, state, state_sig)
    mujoco.mj_forward(self.model, self.data)
    self._model_dirty = False
    self._publish_model()

  def _can_fast_edit(self, kind, values):
    if kind not in ("body", "geom", "site", "camera"):
      return False
    if not values:
      return False
    return set(values).issubset({"position", "quaternion", "size"})

  def _model_object_id(self, kind, node):
    if not node.name:
      return -1
    object_types = {
        "body": mujoco.mjtObj.mjOBJ_BODY,
        "geom": mujoco.mjtObj.mjOBJ_GEOM,
        "site": mujoco.mjtObj.mjOBJ_SITE,
        "camera": mujoco.mjtObj.mjOBJ_CAMERA,
    }
    return int(mujoco.mj_name2id(
        self.model, object_types[kind], str(node.name)
    ))

  def _apply_mesh_runtime_pose(self, kind, node, object_id, values, before):
    """Maps an MJCF mesh pose to MuJoCo's compiled mesh pose."""
    model = self.model
    old_position = np.asarray(before["position"], dtype=np.float64)
    old_quaternion = np.asarray(before["quaternion"], dtype=np.float64)
    new_position = np.asarray(
        values.get("position", tuple(float(v) for v in node.pos)),
        dtype=np.float64,
    )
    new_quaternion = np.asarray(
        values.get("quaternion", tuple(float(v) for v in node.quat)),
        dtype=np.float64,
    )
    mujoco.mju_normalize4(new_quaternion)
    old_logical_matrix = np.empty((3, 3), dtype=np.float64)
    new_logical_matrix = np.empty((3, 3), dtype=np.float64)
    mujoco.mju_quat2Mat(old_logical_matrix.reshape(-1), old_quaternion)
    mujoco.mju_quat2Mat(new_logical_matrix.reshape(-1), new_quaternion)

    if kind == "geom":
      compiled_position = model.geom_pos[object_id].copy()
      compiled_quaternion = model.geom_quat[object_id].copy()
      mesh_id = int(model.geom_dataid[object_id])
    else:
      compiled_position = model.site_pos[object_id].copy()
      compiled_quaternion = model.site_quat[object_id].copy()
      mesh_id = int(model.site_dataid[object_id])
    compiled_matrix = np.empty((3, 3), dtype=np.float64)
    mujoco.mju_quat2Mat(compiled_matrix.reshape(-1), compiled_quaternion)

    asset_position = old_logical_matrix.T @ compiled_position - (
        old_logical_matrix.T @ old_position
    )
    asset_matrix = old_logical_matrix.T @ compiled_matrix
    # Scale is compensated through mesh.refpos below, so the visible mesh
    # keeps the current compiled pose while its dimensions change.
    new_compiled_position = new_position + new_logical_matrix @ asset_position
    new_compiled_matrix = new_logical_matrix @ asset_matrix
    new_compiled_quaternion = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(
        new_compiled_quaternion, new_compiled_matrix.reshape(-1)
    )

    if kind == "geom":
      model.geom_pos[object_id] = new_compiled_position
      model.geom_quat[object_id] = new_compiled_quaternion
    else:
      model.site_pos[object_id] = new_compiled_position
      model.site_quat[object_id] = new_compiled_quaternion
    if "size" in values:
      old_scale = np.asarray(model.mesh_scale[mesh_id], dtype=np.float64).copy()
      new_scale = np.asarray(values["size"], dtype=np.float64)
      self._apply_mesh_vertex_scale(mesh_id, old_scale, new_scale)
      model.mesh_scale[mesh_id] = values["size"]

  def _apply_mesh_vertex_scale(self, mesh_id, old_scale, new_scale):
    """Patches render-time mesh buffers until the next authoritative compile."""
    if old_scale.shape != (3,) or new_scale.shape != (3,):
      return
    ratio = np.ones(3, dtype=np.float64)
    nonzero = np.abs(old_scale) > 1e-9
    ratio[nonzero] = new_scale[nonzero] / old_scale[nonzero]
    vertex_adr = int(self.model.mesh_vertadr[mesh_id])
    vertex_count = int(self.model.mesh_vertnum[mesh_id])
    vertices = np.asarray(
        self.model.mesh_vert[vertex_adr:vertex_adr + vertex_count],
        dtype=np.float64,
    ).reshape(-1, 3)
    if not len(vertices):
      return
    pivot = vertices.mean(axis=0)
    scaled = pivot + (vertices - pivot) * ratio
    self.model.mesh_vert[
        vertex_adr:vertex_adr + vertex_count
    ] = scaled.astype(self.model.mesh_vert.dtype, copy=False)

    normal_adr = int(self.model.mesh_normaladr[mesh_id])
    normal_count = int(self.model.mesh_normalnum[mesh_id])
    if normal_count:
      normals = np.asarray(
          self.model.mesh_normal[
              normal_adr:normal_adr + normal_count
          ],
          dtype=np.float64,
      ).reshape(-1, 3)
      inv_ratio = np.divide(
          1.0, ratio, out=np.zeros(3), where=np.abs(ratio) > 1e-9
      )
      transformed = normals * inv_ratio
      lengths = np.linalg.norm(transformed, axis=1, keepdims=True)
      transformed = np.divide(
          transformed, lengths, out=np.zeros_like(transformed),
          where=lengths > 1e-9,
      )
      self.model.mesh_normal[
          normal_adr:normal_adr + normal_count
      ] = transformed.astype(self.model.mesh_normal.dtype, copy=False)

  def _apply_fast_edit(self, kind, node, values, before):
    """Updates the paused runtime model without rebuilding the full model."""
    object_id = self._model_object_id(kind, node)
    if object_id < 0:
      raise ValueError("Fast editing requires a named MJCF element")

    if kind == "body":
      if "position" in values:
        self.model.body_pos[object_id] = values["position"]
      if "quaternion" in values:
        self.model.body_quat[object_id] = values["quaternion"]
    elif kind == "geom":
      if int(node.type) == int(mujoco.mjtGeom.mjGEOM_MESH):
        self._apply_mesh_runtime_pose(
            kind, node, object_id, values, before
        )
      else:
        if "position" in values:
          self.model.geom_pos[object_id] = values["position"]
        if "quaternion" in values:
          self.model.geom_quat[object_id] = values["quaternion"]
        if "size" in values:
          self.model.geom_size[object_id] = values["size"]
    elif kind == "site":
      if int(node.type) == int(mujoco.mjtGeom.mjGEOM_MESH):
        self._apply_mesh_runtime_pose(
            kind, node, object_id, values, before
        )
      else:
        if "position" in values:
          self.model.site_pos[object_id] = values["position"]
        if "quaternion" in values:
          self.model.site_quat[object_id] = values["quaternion"]
        if "size" in values:
          self.model.site_size[object_id] = values["size"]
    elif kind == "camera":
      if "position" in values:
        self.model.cam_pos[object_id] = values["position"]
      if "quaternion" in values:
        self.model.cam_quat[object_id] = values["quaternion"]

    mujoco.mj_forward(self.model, self.data)
    self._model_dirty = True
    self._publish_model()

  @studio_messages.handler(priority=studio_messages.Priority.LIBRARY)
  def _on_step_control(self, event: studio_messages.StepControlSnapshot) -> bool:
    if event.pause_state == sim.PauseState.UNPAUSED and self._model_dirty:
      self._recompile()
    return False

  def _record(self, command: Command):
    self.history.push(command)

  @studio_messages.handler(priority=studio_messages.Priority.LIBRARY)
  def _on_model(self, event: studio_messages.ModelEvent) -> bool:
    if not event.path:
      print("Scene Authoring disabled: loaded model has no MJCF path")
      return False
    try:
      self.spec, self.model, self.data = load_spec(event.path)
      self.source_path = event.path
      self.authored.clear()
      self.history.clear()
      self._model_dirty = False
      self.publish_scene_scan()
    except Exception as exc:  # pylint: disable=broad-except
      print(f"Scene Authoring could not load {event.path}: {exc}")
    return False

  def _body_at_path(self, body_path):
    body = self.spec.worldbody
    for index in body_path:
      bodies = list(body.bodies)
      if index < 0 or index >= len(bodies):
        raise ValueError(f"Invalid body path: {body_path}")
      body = bodies[index]
    return body

  def _find_body_by_name(self, name):
    def visit(body):
      if str(body.name or "") == name:
        return body
      for child in body.bodies:
        found = visit(child)
        if found is not None:
          return found
      return None

    return visit(self.spec.worldbody)

  def _resolve_node(
      self, kind, target_path, element_index=-1, target_name=""
  ):
    if target_name:
      if kind == "body":
        named_body = self._find_body_by_name(target_name)
        if named_body is not None:
          return named_body
      else:
        def visit(body):
          for candidate in getattr(body, f"{kind}s"):
            if str(candidate.name or "") == target_name:
              return candidate
          for child in body.bodies:
            found = visit(child)
            if found is not None:
              return found
          return None

        named_node = visit(self.spec.worldbody)
        if named_node is not None:
          return named_node
    body = self._body_at_path(tuple(target_path))
    if kind == "body":
      return body
    collection = list(getattr(body, f"{kind}s"))
    if element_index < 0 or element_index >= len(collection):
      raise ValueError(f"Invalid {kind} index: {element_index}")
    return collection[element_index]

  def _unique_name(self, kind):
    prefix = f"authored_{kind}_"
    names = {
        node.name for node in scan_spec(self.spec)
        if node.kind == kind and node.name
    }
    index = 1
    while f"{prefix}{index:03d}" in names:
      index += 1
    return f"{prefix}{index:03d}"

  def _mesh_file_path(self, mesh_file):
    path = Path(str(mesh_file)).expanduser()
    if not path.is_absolute():
      path = Path(self.source_path).expanduser().parent / path
    return str(path)

  def _mesh_asset(self, name):
    for mesh in self.spec.meshes:
      if mesh.name == name:
        return mesh
    return None

  def _mesh_scale(self, node):
    mesh_name = str(getattr(node, "meshname", "") or "")
    mesh = self._mesh_asset(mesh_name)
    if mesh is None:
      return (1.0, 1.0, 1.0)
    return tuple(float(value) for value in mesh.scale)

  def _preserve_mesh_center(self, kind, node, values, before):
    """Adjusts mesh refpos so scaling does not move the visible mesh center."""
    if (
        "size" not in values
        or int(node.type) != int(mujoco.mjtGeom.mjGEOM_MESH)
    ):
      return
    mesh = self._mesh_asset(getattr(node, "meshname", ""))
    if mesh is None:
      return
    object_id = self._model_object_id(kind, node)
    if object_id < 0:
      return

    old_size = np.asarray(before.get("size", mesh.scale), dtype=np.float64)
    new_size = np.asarray(values["size"], dtype=np.float64)
    if old_size.shape != (3,) or new_size.shape != (3,):
      return

    if kind == "geom":
      compiled_position = self.model.geom_pos[object_id]
      mesh_id = int(self.model.geom_dataid[object_id])
    else:
      compiled_position = self.model.site_pos[object_id]
      mesh_id = int(self.model.site_dataid[object_id])
    old_quaternion = np.asarray(before["quaternion"], dtype=np.float64)
    old_position = np.asarray(before["position"], dtype=np.float64)
    old_matrix = np.empty((3, 3), dtype=np.float64)
    mujoco.mju_quat2Mat(old_matrix.reshape(-1), old_quaternion)
    desired_asset_position = old_matrix.T @ (
        np.asarray(compiled_position) - old_position
    )

    # Recover the unscaled mesh center from the current compiled asset data,
    # then choose a refpos that retains the old asset offset at the new scale.
    current_scale = np.asarray(self.model.mesh_scale[mesh_id], dtype=np.float64)
    current_refpos = np.asarray(mesh.refpos, dtype=np.float64)
    raw_center = current_refpos.copy()
    nonzero = np.abs(current_scale) > 1e-9
    raw_center[nonzero] += (
        np.asarray(self.model.mesh_pos[mesh_id], dtype=np.float64)[nonzero]
        / current_scale[nonzero]
    )
    new_refpos = current_refpos.copy()
    new_nonzero = np.abs(new_size) > 1e-9
    new_refpos[new_nonzero] = (
        raw_center[new_nonzero]
        - desired_asset_position[new_nonzero] / new_size[new_nonzero]
    )
    mesh.refpos = list(float(value) for value in new_refpos)

  def _add_node(self, kind, parent_path, name, values):
    parent = self._body_at_path(tuple(parent_path))
    position = list(values.get("position", (0.0, 0.0, 0.0)))
    quaternion = list(values.get("quaternion", (1.0, 0.0, 0.0, 0.0)))
    if kind == "body":
      node = parent.add_body(name=name, pos=position, quat=quaternion)
      if values.get("dynamic", False):
        node.add_freejoint()
      if values.get("with_default_site", False):
        site_name = values.get("default_site_name") or f"{name}_site"
        node.add_site(
            name=site_name,
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.01, 0.0, 0.0],
            rgba=[1.0, 0.3, 0.2, 1.0],
        )
      return node
    if kind == "geom":
      type_id = int(values.get("type", int(mujoco.mjtGeom.mjGEOM_BOX)))
      meshname = values.get("meshname")
      mesh_file = values.get("mesh_file")
      if type_id == int(mujoco.mjtGeom.mjGEOM_MESH):
        if not meshname:
          raise ValueError("A mesh geom requires a mesh asset name")
        if mesh_file and self._mesh_asset(meshname) is None:
          self.spec.add_mesh(
              name=str(meshname), file=self._mesh_file_path(mesh_file)
          )
        if self._mesh_asset(meshname) is None:
          raise ValueError(f"Mesh asset not found: {meshname}")
        mesh = self._mesh_asset(meshname)
        if "size" in values:
          mesh.scale = list(values["size"])
      return parent.add_geom(
          name=name,
          type=type_id,
          pos=position,
          quat=quaternion,
          size=list(values.get("size", (0.1, 0.1, 0.1))),
          mass=float(values.get("mass", 1.0)),
          rgba=list(values.get("rgba", (0.35, 0.65, 1.0, 1.0))),
          meshname=str(meshname) if meshname else None,
      )
    if kind == "site":
      return parent.add_site(
          name=name,
          type=int(values.get("type", int(mujoco.mjtGeom.mjGEOM_SPHERE))),
          pos=position,
          quat=quaternion,
          size=list(values.get("size", (0.05, 0.05, 0.05))),
          rgba=list(values.get("rgba", (1.0, 0.3, 0.2, 1.0))),
      )
    if kind == "camera":
      return parent.add_camera(
          name=name,
          pos=position,
          quat=quaternion,
          fovy=float(values.get("fovy", 45.0)),
      )
    raise ValueError(f"Unsupported node kind: {kind}")

  def _node_values(self, kind, node):
    values = {
        "position": tuple(float(v) for v in node.pos),
        "name": str(node.name or ""),
    }
    if kind in ("body", "geom", "site", "camera"):
      values["quaternion"] = tuple(float(v) for v in node.quat)
    if kind in ("geom", "site", "joint"):
      values["type_id"] = int(node.type)
    if kind == "body":
      values["mass"] = float(node.mass)
    elif kind in ("geom", "site"):
      values["size"] = (
          self._mesh_scale(node)
          if int(node.type) == int(mujoco.mjtGeom.mjGEOM_MESH)
          else tuple(float(v) for v in node.size)
      )
      values["rgba"] = tuple(float(v) for v in node.rgba)
      if kind == "geom":
        values["mass"] = float(node.mass)
    elif kind == "joint":
      values["axis"] = tuple(float(v) for v in node.axis)
      values["range"] = tuple(float(v) for v in node.range)
      values["damping"] = tuple(float(v) for v in node.damping)
    values.update(read_extra_attributes(kind, node))
    return values

  def _apply_node_values(self, kind, node, values):
    if "name" in values and not (kind == "body" and node.parent is None):
      node.name = str(values["name"])
    if "type_id" in values and kind in ("geom", "site", "joint"):
      node.type = int(values["type_id"])
    if "position" in values:
      node.pos = list(values["position"])
    if "quaternion" in values and kind in ("body", "geom", "site", "camera"):
      node.quat = list(values["quaternion"])
    if kind == "body" and "mass" in values:
      node.mass = float(values["mass"])
    elif kind in ("geom", "site"):
      if "size" in values:
        if int(node.type) == int(mujoco.mjtGeom.mjGEOM_MESH):
          mesh = self._mesh_asset(getattr(node, "meshname", ""))
          if mesh is None:
            raise ValueError("Mesh geom has no valid mesh asset")
          mesh.scale = list(values["size"])
        else:
          node.size = list(values["size"])
      if "rgba" in values:
        node.rgba = list(values["rgba"])
      if kind == "geom" and "mass" in values:
        node.mass = float(values["mass"])
    elif kind == "joint":
      if "axis" in values:
        node.axis = list(values["axis"])
      if "range" in values:
        node.range = list(values["range"])
      if "damping" in values:
        node.damping = list(values["damping"])

    for key, value in values.items():
      source = source_name(kind, key)
      if source is None:
        continue
      if key == "userdata":
        node.userdata[:] = list(value)
        continue
      if source == "__classname__":
        node.classname.name = str(value)
        continue
      current = getattr(node, source)
      if isinstance(current, bool):
        value = bool(value)
      elif isinstance(value, (list, tuple)):
        value = list(value)
      elif key in (
          "sleep", "typeinertia", "actfrclimited", "align", "limited",
      ):
        value = int(value)
      setattr(node, source, value)

  def _delete_node(self, kind, target_path, element_index):
    if kind == "body" and not target_path:
      raise ValueError("The world body cannot be deleted")
    node = self._resolve_node(kind, target_path, element_index)
    self.spec.delete(node)

  @studio_messages.handler
  def _on_create_node(self, event: messages.CreateNodeEvent) -> bool:
    try:
      parent_path = tuple(event.parent_path)
      parent = self._body_at_path(parent_path)
      element_index = (
          len(parent.bodies) if event.kind == "body"
          else len(getattr(parent, f"{event.kind}s"))
      )
      values = dict(event.values)
      self._add_node(event.kind, parent_path, event.name, values)
      self._recompile()
      self._record(Command("add_node", {
          "kind": event.kind,
          "parent_path": parent_path,
          "element_index": element_index,
          "name": event.name,
          "values": values,
      }))
      self._result(
          event, "create", True, object_name=event.name, model_changed=True
      )
    except Exception as exc:  # pylint: disable=broad-except
      self._result(event, "create", False, message=str(exc))
    return True

  def _refresh_authored(self):
    for name, primitive in list(self.authored.items()):
      body = find_authored_body(self.spec, name)
      if body is None:
        continue
      geoms = list(body.geoms)
      geom = geoms[0] if geoms else None
      size = (
          tuple(float(v) for v in geom.size)
          if geom is not None else primitive.size
      )
      mass = (
          float(geom.mass)
          if geom is not None and geom.mass == geom.mass
          else primitive.mass
      )
      self.authored[name] = PrimitiveDescription(
          name=name,
          primitive_type=primitive.primitive_type,
          position=tuple(float(v) for v in body.pos),
          size=size,
          mass=mass,
          dynamic=primitive.dynamic,
      )

  @studio_messages.handler
  def _on_edit_node(self, event: messages.EditNodeEvent) -> bool:
    before_spec = None
    before_authored = None
    try:
      node = self._resolve_node(
          event.kind, event.target_path, event.element_index, event.target_name
      )
      before_spec = self.spec.copy()
      before_authored = dict(self.authored)
      before = self._node_values(event.kind, node)
      scale_only = (
          event.kind in ("geom", "site")
          and "size" in event.values
          and "position" not in event.values
          and "quaternion" not in event.values
      )
      preserved_position = tuple(float(v) for v in node.pos)
      preserved_quaternion = tuple(float(v) for v in node.quat)
      self._apply_node_values(event.kind, node, event.values)
      self._preserve_mesh_center(event.kind, node, event.values, before)
      if scale_only:
        # Scaling changes the shape only. Keep the local pose explicit even if
        # a future MjSpec implementation normalizes geometry attributes.
        node.pos = list(preserved_position)
        node.quat = list(preserved_quaternion)
      after = self._node_values(event.kind, node)
      self._refresh_authored()
      if self._can_fast_edit(event.kind, event.values):
        self._apply_fast_edit(event.kind, node, event.values, before)
      else:
        self._recompile()
      if event.kind == "body":
        body_name = str(event.values.get("name", node.name or ""))
        if body_name:
          self._set_body_pose(
              body_name,
              event.values.get("position", tuple(node.pos)),
              event.values.get("quaternion", tuple(node.quat)),
          )
      before_pose = None
      after_pose = None
      if event.kind == "body":
        before_pose = {
            "name": str(before.get("name", "")),
            "position": tuple(before["position"]),
            "quaternion": tuple(before["quaternion"]),
        }
        after_pose = {
            "name": str(after.get("name", "")),
            "position": tuple(after["position"]),
            "quaternion": tuple(after["quaternion"]),
        }
      self._record(Command("edit_spec", {
          "kind": event.kind,
          "before_spec": before_spec,
          "after_spec": self.spec.copy(),
          "before_authored": before_authored,
          "after_authored": dict(self.authored),
          "before_pose": before_pose,
          "after_pose": after_pose,
      }))
      self._result(event, "edit", True, model_changed=True)
    except Exception as exc:  # pylint: disable=broad-except
      if before_spec is not None and before_authored is not None:
        self.spec = before_spec
        self.authored = before_authored
      self._result(event, "edit", False, message=str(exc))
    return True

  @studio_messages.handler
  def _on_create(self, event: messages.CreatePrimitiveEvent) -> bool:
    try:
      primitive = PrimitiveDescription(
          name=event.name,
          primitive_type=event.primitive_type,
          position=event.position,
          size=event.size,
          mass=event.mass,
          dynamic=event.dynamic,
      )
      add_primitive(self.spec, primitive)
      self._recompile()
      self.authored[event.name] = primitive
      self._record(Command("add", {"primitive": asdict(primitive)}))
      self._result(event, "create", True, object_name=event.name, model_changed=True)
    except Exception as exc:  # pylint: disable=broad-except
      self._result(event, "create", False, message=str(exc))
    return True

  def _body_qpos(self, body_name: str):
    body_id = mujoco.mj_name2id(
        self.model, mujoco.mjtObj.mjOBJ_BODY, body_name
    )
    if body_id < 0:
      return None
    for joint_id in range(self.model.njnt):
      if (
          self.model.jnt_bodyid[joint_id] == body_id
          and self.model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE
      ):
        return int(self.model.jnt_qposadr[joint_id])
    return None

  def _set_body_pose(self, name, position, quaternion):
    qpos_adr = self._body_qpos(name)
    if qpos_adr is not None:
      self.data.qpos[qpos_adr:qpos_adr + 3] = position
      self.data.qpos[qpos_adr + 3:qpos_adr + 7] = quaternion
      mujoco.mj_forward(self.model, self.data)

  @studio_messages.handler
  def _on_transform(self, event: messages.TransformObjectEvent) -> bool:
    primitive = self.authored.get(event.object_name)
    body = find_authored_body(self.spec, event.object_name)
    if primitive is None or body is None:
      self._result(event, "transform", False, message="Unknown authored object")
      return True
    old_position = tuple(float(v) for v in body.pos)
    old_quaternion = tuple(float(v) for v in body.quat)
    body.pos = list(event.position)
    body.quat = list(event.quaternion)
    try:
      self._recompile()
      self._set_body_pose(event.object_name, event.position, event.quaternion)
      updated = PrimitiveDescription(
          name=primitive.name,
          primitive_type=primitive.primitive_type,
          position=event.position,
          size=primitive.size,
          mass=primitive.mass,
          dynamic=primitive.dynamic,
      )
      self.authored[event.object_name] = updated
      self._record(Command("transform", {
          "name": event.object_name,
          "before": old_position,
          "after": event.position,
          "before_quaternion": old_quaternion,
          "quaternion": event.quaternion,
      }))
      self._result(event, "transform", True,
                   object_name=event.object_name, model_changed=True)
    except Exception as exc:  # pylint: disable=broad-except
      body.pos = list(old_position)
      body.quat = list(old_quaternion)
      self._result(event, "transform", False, message=str(exc))
    return True

  def _delete_name(self, name):
    body = find_authored_body(self.spec, name)
    if body is None:
      raise ValueError("Unknown authored object")
    primitive = self.authored.pop(name)
    self.spec.delete(body)
    return primitive

  @studio_messages.handler
  def _on_delete(self, event: messages.DeleteObjectEvent) -> bool:
    try:
      if event.kind:
        if event.kind not in ("body", "geom", "site", "camera"):
          raise ValueError(f"Unsupported delete kind: {event.kind}")
        if event.kind == "body" and not event.target_path:
          raise ValueError("The world body cannot be deleted")
        node = self._resolve_node(
            event.kind,
            event.target_path,
            event.element_index,
            event.object_name,
        )
        before_spec = self.spec.copy()
        before_authored = dict(self.authored)
        self.spec.delete(node)
        self.authored.pop(event.object_name, None)
        self._recompile()
        self._record(Command("delete_spec", {
            "before_spec": before_spec,
            "after_spec": self.spec.copy(),
            "before_authored": before_authored,
            "after_authored": dict(self.authored),
        }))
      else:
        primitive = self._delete_name(event.object_name)
        self._recompile()
        self._record(Command("delete", {"primitive": asdict(primitive)}))
      self._result(event, "delete", True, model_changed=True)
    except Exception as exc:  # pylint: disable=broad-except
      self._result(event, "delete", False, message=str(exc))
    return True

  def _apply_command(self, command: Command, forward: bool):
    payload = command.payload
    if command.kind == "add":
      primitive = PrimitiveDescription(**payload["primitive"])
      if forward:
        add_primitive(self.spec, primitive)
        self.authored[primitive.name] = primitive
      else:
        self._delete_name(primitive.name)
    elif command.kind == "delete":
      primitive = PrimitiveDescription(**payload["primitive"])
      if forward:
        self._delete_name(primitive.name)
      else:
        add_primitive(self.spec, primitive)
        self.authored[primitive.name] = primitive
    elif command.kind == "delete_spec":
      snapshot = payload["after_spec"] if forward else payload["before_spec"]
      authored = (
          payload["after_authored"] if forward
          else payload["before_authored"]
      )
      self.spec = snapshot.copy()
      self.authored = dict(authored)
    elif command.kind == "transform":
      name = payload["name"]
      body = find_authored_body(self.spec, name)
      if body is None:
        raise ValueError("Unknown authored object")
      position = payload["after"] if forward else payload["before"]
      quaternion = payload.get("quaternion", (1.0, 0.0, 0.0, 0.0))
      if not forward:
        quaternion = payload.get(
            "before_quaternion", (1.0, 0.0, 0.0, 0.0)
        )
      body.pos = list(position)
      body.quat = list(quaternion)
      primitive = self.authored[name]
      self.authored[name] = PrimitiveDescription(
          name=primitive.name,
          primitive_type=primitive.primitive_type,
          position=tuple(position), size=primitive.size,
          mass=primitive.mass, dynamic=primitive.dynamic,
      )
    elif command.kind == "edit_spec":
      snapshot = payload["after_spec"] if forward else payload["before_spec"]
      authored = (
          payload["after_authored"] if forward
          else payload["before_authored"]
      )
      self.spec = snapshot.copy()
      self.authored = dict(authored)
    elif command.kind == "add_node":
      if forward:
        self._add_node(
            payload["kind"], payload["parent_path"],
            payload["name"], payload["values"],
        )
      else:
        self._delete_node(
            payload["kind"], payload["parent_path"],
            payload["element_index"],
        )
    elif command.kind == "edit_node":
      node = self._resolve_node(
          payload["kind"], payload["target_path"], payload["element_index"]
      )
      values = payload["after"] if forward else payload["before"]
      self._apply_node_values(payload["kind"], node, values)

  def _history_event(self, event, forward):
    command = self.history.redo() if forward else self.history.undo()
    if command is None:
      self._result(event, "redo" if forward else "undo", True)
      return
    try:
      self._apply_command(command, forward=forward)
      self._recompile()
      if command.kind == "transform":
        payload = command.payload
        position = payload["after"] if forward else payload["before"]
        quaternion = payload.get("quaternion", (1.0, 0.0, 0.0, 0.0))
        if not forward:
          quaternion = payload.get(
              "before_quaternion", (1.0, 0.0, 0.0, 0.0)
          )
        self._set_body_pose(payload["name"], position, quaternion)
      elif command.kind == "edit_spec":
        pose = (
            command.payload["after_pose"] if forward
            else command.payload["before_pose"]
        )
        if pose is not None and pose["name"]:
          self._set_body_pose(
              pose["name"], pose["position"], pose["quaternion"]
          )
      self._result(event, "redo" if forward else "undo", True, model_changed=True)
    except Exception as exc:  # pylint: disable=broad-except
      self._result(event, "redo" if forward else "undo", False, message=str(exc))

  @studio_messages.handler
  def _on_undo(self, event: messages.UndoEvent) -> bool:
    self._history_event(event, forward=False)
    return True

  @studio_messages.handler
  def _on_redo(self, event: messages.RedoEvent) -> bool:
    self._history_event(event, forward=True)
    return True

  @studio_messages.handler
  def _on_save(self, event: messages.SaveSceneEvent) -> bool:
    try:
      path = save_spec(self.spec, self.source_path, event.path or None)
      self._result(event, "save", True, message=path)
    except Exception as exc:  # pylint: disable=broad-except
      self._result(event, "save", False, message=str(exc))
    return True
