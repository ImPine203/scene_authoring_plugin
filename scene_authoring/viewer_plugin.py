"""Viewer-side ImGui, picking and preview plugin."""

import itertools
import re
from pathlib import Path

import mujoco
import numpy as np
from mujoco.experimental.dear_imgui import dear_imgui as imgui
from mujoco.experimental.studio import messages as studio_messages
from mujoco.experimental.studio import sim
from mujoco.experimental.studio import viewer_app
from mujoco.experimental.studio import viewer_protocol

from scene_authoring import gizmo, messages, picking, properties
from scene_authoring.state import AuthoringMode, AuthoringState


_GEOM_TYPES = [
    (int(mujoco.mjtGeom.mjGEOM_PLANE), "plane"),
    (int(mujoco.mjtGeom.mjGEOM_SPHERE), "sphere"),
    (int(mujoco.mjtGeom.mjGEOM_CAPSULE), "capsule"),
    (int(mujoco.mjtGeom.mjGEOM_ELLIPSOID), "ellipsoid"),
    (int(mujoco.mjtGeom.mjGEOM_CYLINDER), "cylinder"),
    (int(mujoco.mjtGeom.mjGEOM_BOX), "box"),
    (int(mujoco.mjtGeom.mjGEOM_MESH), "mesh"),
    (int(mujoco.mjtGeom.mjGEOM_HFIELD), "hfield"),
]
_JOINT_TYPES = [
    (int(mujoco.mjtJoint.mjJNT_FREE), "free"),
    (int(mujoco.mjtJoint.mjJNT_BALL), "ball"),
    (int(mujoco.mjtJoint.mjJNT_SLIDE), "slide"),
    (int(mujoco.mjtJoint.mjJNT_HINGE), "hinge"),
]


_CREATE_GEOM_TYPES = [
    (int(mujoco.mjtGeom.mjGEOM_PLANE), "plane"),
    (int(mujoco.mjtGeom.mjGEOM_SPHERE), "sphere"),
    (int(mujoco.mjtGeom.mjGEOM_CAPSULE), "capsule"),
    (int(mujoco.mjtGeom.mjGEOM_ELLIPSOID), "ellipsoid"),
    (int(mujoco.mjtGeom.mjGEOM_CYLINDER), "cylinder"),
    (int(mujoco.mjtGeom.mjGEOM_BOX), "box"),
    (int(mujoco.mjtGeom.mjGEOM_MESH), "mesh"),
]
_CREATE_SITE_TYPES = [
    (int(mujoco.mjtGeom.mjGEOM_SPHERE), "sphere"),
    (int(mujoco.mjtGeom.mjGEOM_CAPSULE), "capsule"),
    (int(mujoco.mjtGeom.mjGEOM_ELLIPSOID), "ellipsoid"),
    (int(mujoco.mjtGeom.mjGEOM_CYLINDER), "cylinder"),
    (int(mujoco.mjtGeom.mjGEOM_BOX), "box"),
]


class SceneAuthoringViewerPlugin:
  def __init__(self):
    self.viewer = None
    self.app = None
    self.state = AuthoringState()
    self._request_ids = itertools.count(1)
    self._owned_geoms = []
    self._last_mouse_ray = None
    self._drag_start_ray = None
    self._scene_nodes = {}
    self._scene_root_keys = ()
    self._save_folder = ""
    self._save_filename = ""
    self._save_error = ""
    self._last_saved_path = ""
    self._save_popup_pending = False
    self._pending_create_name = None
    self._property_drafts = {}
    self._property_vector_buffers = {}
    self._property_height = 360.0
    self._create_error = ""
    self._create_dialog_pending = False
    self._create_dialog_kind = ""
    self._create_dialog_parent_path = ()
    self._create_dialog_type_id = int(mujoco.mjtGeom.mjGEOM_BOX)
    self._create_dialog_mesh_file = ""
    self._mesh_browser_pending = False
    self._mesh_browser_dir = ""
    self._mesh_browser_error = ""
    self._create_dialog_size = [0.1, 0.1, 0.1]
    self._create_dialog_mass = 1.0
    self._scene_authoring_active = False
    self._authoring_pause_lock = False
    self._pause_after_model_update = False
    self._gizmo_node_key = None
    self._gizmo_kind = None
    self._gizmo_target_path = ()
    self._gizmo_element_index = -1
    self._gizmo_mode = 'translate'
    self._gizmo_axis = None
    self._gizmo_start_ray = None
    self._gizmo_start_mouse = None
    self._gizmo_start_world_position = None
    self._gizmo_start_world_matrix = None
    self._gizmo_start_display_origin = None
    self._gizmo_parent_position = None
    self._gizmo_parent_matrix = None
    self._gizmo_start_local_position = None
    self._gizmo_start_quaternion = None
    self._gizmo_start_size = None
    self._gizmo_preview_world_position = None
    self._gizmo_preview_local_position = None
    self._gizmo_preview_quaternion = None
    self._gizmo_preview_size = None

  def _next_request(self):
    request_id = next(self._request_ids)
    self.state.pending_request_id = request_id
    return request_id

  def _pause(self):
    self._authoring_pause_lock = True
    if self.app is not None:
      self.app.step_control_state.set_pause_state(sim.PauseState.NORMAL_PAUSED)
      self.state.paused_for_authoring = True

  def _authoring_interaction_enabled(self):
    if not self._scene_authoring_active or self.app is None:
      return False
    return (
        self.app.step_control_state.get_pause_state()
        != sim.PauseState.UNPAUSED
    )

  def _sync_authoring_pause(self, panel_active):
    if self.app is None:
      return

    if not panel_active:
      # The ViewerApp calls input handlers before the next GUI build. Clear
      # transient authoring interaction as soon as the panel is known to be
      # inactive, so the standard Studio mouse/keyboard handlers own the
      # viewport in other tabs.
      if self._scene_authoring_active:
        self._clear_gizmo_drag()
        self.state.mode = AuthoringMode.SELECT
        self.state.preview_position = None
        self._gizmo_node_key = None
        self._gizmo_kind = None
        self._gizmo_target_path = ()
        self._gizmo_element_index = -1
      self._scene_authoring_active = False
      return

    # Re-entering the Scene Authoring tab is a new authoring session. Pause
    # immediately even if the user pressed Run while another Studio tab was
    # active and the previous authoring lock was cleared there.
    if not self._scene_authoring_active:
      self._scene_authoring_active = True
      self._pause()
      return

    pause_state = self.app.step_control_state.get_pause_state()
    if self._authoring_pause_lock and self._pause_after_model_update:
      # ViewerApp resets its StepControl when a recompiled ModelEvent arrives.
      # That reset is not a user request to run, so restore authoring pause.
      self._pause_after_model_update = False
      self._pause()
    elif self._authoring_pause_lock and pause_state == sim.PauseState.UNPAUSED:
      # The Studio Run button (or its normal keyboard equivalent) explicitly
      # released the authoring pause lock.
      self._authoring_pause_lock = False
      self.state.paused_for_authoring = False
    elif self._authoring_pause_lock:
      self._pause()

    # Run explicitly disables authoring interaction until the user pauses
    # again. Drop an in-progress drag so it cannot commit after Run.
    if self.app.step_control_state.get_pause_state() == sim.PauseState.UNPAUSED:
      self._clear_gizmo_drag()
      self.state.preview_position = None

    self._scene_authoring_active = panel_active

  def _send(self, event):
    self._pause()
    self.viewer.send_to_sim(event)

  @studio_messages.handler(priority=studio_messages.Priority.LIBRARY)
  def _on_viewer_init(self, event: viewer_protocol.ViewerInitEvent):
    self.viewer = event.viewer

  @studio_messages.handler(priority=studio_messages.Priority.LIBRARY)
  def _on_viewer_app_init(self, event: viewer_app.ViewerAppInitEvent):
    self.app = event.viewer_app

  @studio_messages.handler(priority=studio_messages.Priority.LIBRARY)
  def _on_model(self, event: studio_messages.ModelEvent) -> bool:
    del event
    self._pause_after_model_update = self._authoring_pause_lock
    self._remove_owned_geoms()
    self.state.selected_body_id = self._resolve_selected_body()
    if self.state.selected_body_id < 0:
      self.state.selected_name = None
    self.state.hovered_axis = None
    self.state.dragging_axis = None
    self.state.preview_position = None
    self._drag_start_ray = None
    return False

  @studio_messages.handler(priority=studio_messages.Priority.LIBRARY)
  def _on_scene_scan(self, event: messages.SceneScanEvent) -> bool:
    self._scene_nodes = {node.key: node for node in event.nodes}
    self._scene_root_keys = tuple(
        node.key for node in event.nodes if node.parent_key is None
    )
    if event.source_path != self.state.scene_source_path:
      source = Path(event.source_path)
      self._save_folder = str(source.parent)
      self._save_filename = (
          f"{source.stem}_edited{source.suffix or '.xml'}"
      )
      self._save_error = ""
    self.state.scene_source_path = event.source_path
    if self._gizmo_kind is not None:
      target = self._find_scene_node(
          self._gizmo_kind, self._gizmo_target_path,
          self._gizmo_element_index
      )
      self._gizmo_node_key = target.key if target is not None else None
    self._property_drafts = {
        node.key: self._draft_from_node(node) for node in event.nodes
    }
    self._property_vector_buffers.clear()
    if self._pending_create_name:
      matches = [
          node for node in event.nodes
          if node.name == self._pending_create_name
      ]
      if matches:
        self.state.inspected_node_key = matches[0].key
        self._activate_gizmo_node(matches[0])
        self._pending_create_name = None
    elif self.state.inspected_node_key not in self._scene_nodes:
      self.state.inspected_node_key = None
    self.state.status = f"Scanned {len(event.nodes)} scene elements"
    return False

  @studio_messages.handler(priority=studio_messages.Priority.LIBRARY)
  def _on_result(self, event: messages.AuthoringResultEvent) -> bool:
    if event.request_id != self.state.pending_request_id:
      return False
    self.state.pending_request_id = None
    self.state.status = event.message or (
        f"{event.operation} completed" if event.success else "Operation failed"
    )
    if event.operation == "save":
      if event.success:
        self._last_saved_path = event.message
        self._save_error = ""
      else:
        self._save_error = event.message or "Save failed"
    if event.success and event.operation == "create":
      self.state.selected_name = event.object_name
      self.state.mode = AuthoringMode.SELECT
    if event.success and event.operation in ("transform", "delete"):
      if event.operation == "delete":
        self.state.selected_name = None
      self.state.mode = AuthoringMode.SELECT
    if not event.success:
      self.state.preview_position = None
    return False

  def _find_scene_node(self, kind, target_path, element_index=-1):
    target_path = tuple(target_path)
    for node in self._scene_nodes.values():
      if node.kind != kind or tuple(node.target_path) != target_path:
        continue
      if kind == "body" or node.element_index == element_index:
        return node
    return None

  def _body_node_for_model_id(self, body_id):
    if body_id == 0:
      return self._find_scene_node("body", ())
    name = mujoco.mj_id2name(
        self.viewer.model, mujoco.mjtObj.mjOBJ_BODY, int(body_id)
    )
    if name:
      for node in self._scene_nodes.values():
        if node.kind == "body" and node.name == name:
          return node
    body_nodes = [
        node for node in self._scene_nodes.values() if node.kind == "body"
    ]
    return body_nodes[body_id] if body_id < len(body_nodes) else None

  def _model_id_for_node(self, node):
    if self.viewer is None:
      return -1
    if node.kind == "body":
      if not node.name and not node.target_path:
        return 0
      if node.name:
        return mujoco.mj_name2id(
            self.viewer.model, mujoco.mjtObj.mjOBJ_BODY, node.name
        )
      return -1
    if node.kind not in ("geom", "site"):
      return -1
    obj_type = (
        mujoco.mjtObj.mjOBJ_GEOM
        if node.kind == "geom" else mujoco.mjtObj.mjOBJ_SITE
    )
    if node.name:
      object_id = mujoco.mj_name2id(self.viewer.model, obj_type, node.name)
      if object_id >= 0:
        return object_id
    parent = self._find_scene_node("body", node.target_path)
    if parent is None:
      return -1
    parent_id = self._model_id_for_node(parent)
    if parent_id < 0:
      return -1
    if node.kind == "geom":
      body_ids = self.viewer.model.geom_bodyid
      candidates = [
          index for index, value in enumerate(body_ids)
          if int(value) == parent_id
      ]
    else:
      body_ids = self.viewer.model.site_bodyid
      candidates = [
          index for index, value in enumerate(body_ids)
          if int(value) == parent_id
      ]
    return candidates[node.element_index] if node.element_index < len(candidates) else -1

  def _node_for_model_object(self, kind, object_id):
    if object_id < 0:
      return None
    name = mujoco.mj_id2name(
        self.viewer.model,
        mujoco.mjtObj.mjOBJ_GEOM if kind == "geom" else mujoco.mjtObj.mjOBJ_SITE,
        int(object_id),
    )
    if name:
      for node in self._scene_nodes.values():
        if node.kind == kind and node.name == name:
          return node
    if kind == "geom":
      body_id = int(self.viewer.model.geom_bodyid[object_id])
      body_ids = self.viewer.model.geom_bodyid
    else:
      body_id = int(self.viewer.model.site_bodyid[object_id])
      body_ids = self.viewer.model.site_bodyid
    body = self._body_node_for_model_id(body_id)
    if body is None:
      return None
    local_index = sum(int(value) == body_id for value in body_ids[:object_id])
    return self._find_scene_node(kind, body.target_path, local_index)

  def _target_node(self):
    if self._gizmo_kind is None:
      return None
    node = self._scene_nodes.get(self._gizmo_node_key)
    if node is not None:
      return node
    return self._find_scene_node(
        self._gizmo_kind, self._gizmo_target_path, self._gizmo_element_index
    )

  def _world_pose(self, node):
    object_id = self._model_id_for_node(node)
    if object_id < 0:
      return None
    if node.kind == "body":
      position = self.viewer.data.xpos[object_id]
      matrix = self.viewer.data.xmat[object_id].reshape(3, 3)
    elif node.kind == "geom":
      position = self.viewer.data.geom_xpos[object_id]
      matrix = self.viewer.data.geom_xmat[object_id].reshape(3, 3)
    else:
      position = self.viewer.data.site_xpos[object_id]
      matrix = self.viewer.data.site_xmat[object_id].reshape(3, 3)
    return np.asarray(position, dtype=np.float64).copy(), np.asarray(matrix, dtype=np.float64).copy()

  def _camera_frame(self):
    """Returns the current camera position and orthonormal frame."""
    camera_position = np.zeros(3, dtype=np.float64)
    forward = np.zeros(3, dtype=np.float64)
    up = np.zeros(3, dtype=np.float64)
    right = np.zeros(3, dtype=np.float64)
    mujoco.mjv_cameraFrame(
        camera_position, forward, up, right, self.viewer.data, self.viewer.camera
    )
    return camera_position, forward, up, right

  def _gizmo_length(self, origin):
    """Chooses a gizmo size in world units for a stable screen size."""
    io = imgui.GetIO()
    height = max(float(io.DisplaySize.y), 1.0)
    zver = np.zeros(2, dtype=np.float32)
    zhor = np.zeros(2, dtype=np.float32)
    zclip = np.zeros(2, dtype=np.float32)
    mujoco.mjv_cameraFrustum(
        zver, zhor, zclip, self.viewer.model, self.viewer.camera
    )
    camera_position, forward, _, _ = self._camera_frame()
    depth = max(
        float(np.dot(np.asarray(origin) - camera_position, forward)),
        float(zclip[0]),
    )
    frustum_height = float(zver[0] + zver[1])
    if self.viewer.camera.orthographic:
      world_height = frustum_height
    else:
      world_height = frustum_height * depth / max(float(zclip[0]), 1e-6)
    # Keep the handle easy to grab without making its size depend on the
    # object's distance from the global origin.
    return max(0.02, world_height * 185.0 / height)

  def _node_bound_radius(self, node):
    """Returns a conservative radius for placing the gizmo outside a node."""
    if node.kind in ("geom", "site"):
      size_source = node.size
      if (
          node.key == self._gizmo_node_key
          and self._gizmo_preview_size is not None
      ):
        size_source = self._gizmo_preview_size
      size = np.abs(np.asarray(size_source, dtype=np.float64))
      type_id = int(node.type_id)
      if type_id == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        return max(float(size[0]), 0.02)
      if type_id in (
          int(mujoco.mjtGeom.mjGEOM_CAPSULE),
          int(mujoco.mjtGeom.mjGEOM_CYLINDER),
      ):
        return max(float(size[0] + size[1]), 0.02)
      if type_id == int(mujoco.mjtGeom.mjGEOM_PLANE):
        # A plane has no thickness. Its in-plane extent must not push the
        # gizmo behind the camera when the user zooms toward a large plane.
        return 0.02
      return max(float(np.linalg.norm(size)), 0.02)

    if node.kind != "body":
      return 0.1
    body_id = self._model_id_for_node(node)
    if body_id < 0:
      return 0.1
    body_position = np.asarray(self.viewer.data.xpos[body_id], dtype=np.float64)
    radius = 0.02
    for geom_id in range(self.viewer.model.ngeom):
      if int(self.viewer.model.geom_bodyid[geom_id]) != body_id:
        continue
      geom_position = np.asarray(
          self.viewer.data.geom_xpos[geom_id], dtype=np.float64
      )
      size = np.abs(
          np.asarray(self.viewer.model.geom_size[geom_id], dtype=np.float64)
      )
      type_id = int(self.viewer.model.geom_type[geom_id])
      if type_id == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        geom_radius = float(size[0])
      elif type_id in (
          int(mujoco.mjtGeom.mjGEOM_CAPSULE),
          int(mujoco.mjtGeom.mjGEOM_CYLINDER),
      ):
        geom_radius = float(size[0] + size[1])
      else:
        geom_radius = float(np.linalg.norm(size))
      radius = max(
          radius,
          float(np.linalg.norm(geom_position - body_position)) + geom_radius,
      )
    for site_id in range(self.viewer.model.nsite):
      if int(self.viewer.model.site_bodyid[site_id]) != body_id:
        continue
      site_position = np.asarray(
          self.viewer.data.site_xpos[site_id], dtype=np.float64
      )
      radius = max(
          radius,
          float(np.linalg.norm(site_position - body_position))
          + float(np.linalg.norm(self.viewer.model.site_size[site_id])),
      )
    return radius

  def _gizmo_display_origin(self, node, origin, length=None):
    """Places the visual handle on the camera-facing side of the selection."""
    if length is None:
      length = self._gizmo_length(origin)
    camera_position, forward, _, _ = self._camera_frame()
    origin = np.asarray(origin, dtype=np.float64)
    to_camera = camera_position - origin
    distance = float(np.linalg.norm(to_camera))
    if distance <= 1e-8:
      to_camera = -np.asarray(forward, dtype=np.float64)
      distance = 1.0
    else:
      to_camera /= distance

    zclip = np.zeros(2, dtype=np.float32)
    mujoco.mjv_cameraFrustum(
        np.zeros(2, dtype=np.float32),
        np.zeros(2, dtype=np.float32),
        zclip,
        self.viewer.model,
        self.viewer.camera,
    )
    depth = float(np.dot(origin - camera_position, forward))
    max_offset = max(0.0, depth - 4.0 * float(zclip[0]))
    # Move past the object's conservative front radius, and leave the whole
    # handle outside the surface when its size changes during scaling.
    offset = min(self._node_bound_radius(node) + length, max_offset)
    return origin + to_camera * offset

  def _parent_pose(self, node):
    parent_path = (
        tuple(node.target_path[:-1])
        if node.kind == "body" else tuple(node.target_path)
    )
    parent = self._find_scene_node("body", parent_path)
    if parent is None:
      return np.zeros(3, dtype=np.float64), np.eye(3, dtype=np.float64)
    pose = self._world_pose(parent)
    if pose is None:
      return np.zeros(3, dtype=np.float64), np.eye(3, dtype=np.float64)
    return pose

  def _activate_gizmo_node(self, node):
    if node.kind not in ("body", "geom", "site"):
      return False
    if node.kind == "body" and not node.target_path:
      return False
    self._pause()
    self.state.inspected_node_key = node.key
    self._gizmo_node_key = node.key
    self._gizmo_kind = node.kind
    self._gizmo_target_path = tuple(node.target_path)
    self._gizmo_element_index = node.element_index
    self._gizmo_mode = "translate"
    self._clear_gizmo_drag()
    self.state.status = f"Gizmo {node.kind}: {node.name or '(unnamed)'}"
    return True

  def _clear_gizmo_drag(self):
    self._gizmo_axis = None
    self._gizmo_start_ray = None
    self._gizmo_start_mouse = None
    self._gizmo_start_world_position = None
    self._gizmo_start_world_matrix = None
    self._gizmo_start_display_origin = None
    self._gizmo_parent_position = None
    self._gizmo_parent_matrix = None
    self._gizmo_start_local_position = None
    self._gizmo_start_quaternion = None
    self._gizmo_start_size = None
    self._gizmo_preview_world_position = None
    self._gizmo_preview_local_position = None
    self._gizmo_preview_quaternion = None
    self._gizmo_preview_size = None

  def _pick_site(self, ray_origin, ray_direction):
    best = None
    best_parameter = float("inf")
    for site_id in range(self.viewer.model.nsite):
      distance, parameter = picking.ray_point_distance(
          ray_origin, ray_direction, self.viewer.data.site_xpos[site_id]
      )
      radius = max(
          0.04, float(np.max(self.viewer.model.site_size[site_id])) * 1.5
      )
      if parameter >= 0.0 and distance <= radius and parameter < best_parameter:
        node = self._node_for_model_object("site", site_id)
        if node is not None:
          best = node
          best_parameter = parameter
    return best, best_parameter

  def _pick_gizmo_node(self, picked, ray):
    site, site_parameter = self._pick_site(ray[0], ray[1])
    if site is not None and (picked.dist < 0 or site_parameter <= picked.dist):
      return site
    if picked.geom >= 0:
      node = self._node_for_model_object("geom", picked.geom)
      if node is not None:
        return node
    if picked.body > 0:
      name = mujoco.mj_id2name(
          self.viewer.model, mujoco.mjtObj.mjOBJ_BODY, picked.body
      )
      for node in self._scene_nodes.values():
        if node.kind == "body" and node.name == name:
          return node
    return None

  def _start_gizmo_drag(self, axis, mouse, ray, node):
    pose = self._world_pose(node)
    if pose is None:
      return False
    self._pause()
    self._gizmo_axis = axis
    self._gizmo_start_ray = ray
    self._gizmo_start_mouse = (mouse[0], mouse[1])
    self._gizmo_start_world_position = pose[0]
    self._gizmo_start_world_matrix = pose[1]
    length = self._gizmo_length(pose[0])
    self._gizmo_start_display_origin = self._gizmo_display_origin(
        node, pose[0], length
    )
    self._gizmo_parent_position, self._gizmo_parent_matrix = self._parent_pose(node)
    self._gizmo_start_local_position = np.asarray(node.position, dtype=np.float64)
    self._gizmo_start_quaternion = np.asarray(node.quaternion, dtype=np.float64)
    self._gizmo_start_size = np.asarray(node.size, dtype=np.float64)
    self._gizmo_preview_world_position = pose[0].copy()
    self._gizmo_preview_local_position = self._gizmo_start_local_position.copy()
    self._gizmo_preview_quaternion = self._gizmo_start_quaternion.copy()
    self._gizmo_preview_size = self._gizmo_start_size.copy()
    return True

  def _scaled_size(self, node, factor, axis):
    size = self._gizmo_start_size.copy()
    if node.kind not in ("geom", "site"):
      return size
    type_id = int(node.type_id)
    if type_id == int(mujoco.mjtGeom.mjGEOM_SPHERE):
      size[0] = max(0.001, size[0] * factor)
    elif type_id in (
        int(mujoco.mjtGeom.mjGEOM_CAPSULE),
        int(mujoco.mjtGeom.mjGEOM_CYLINDER),
    ):
      size[1 if axis == "z" else 0] = max(
          0.001, size[1 if axis == "z" else 0] * factor
      )
    elif type_id == int(mujoco.mjtGeom.mjGEOM_PLANE):
      if axis in ("x", "y"):
        size["xyz".index(axis)] = max(
            0.001, size["xyz".index(axis)] * factor
        )
    else:
      index = "xyz".index(axis)
      size[index] = max(0.001, size[index] * factor)
    return size

  def _update_gizmo_drag(self, mouse, ray, node):
    if self._gizmo_mode == "translate":
      delta = gizmo.drag_delta(
          self._gizmo_axis, self._gizmo_start_ray, ray,
          self._gizmo_start_display_origin,
          basis=self._gizmo_start_world_matrix,
      )
      direction = self._gizmo_start_world_matrix @ gizmo.axis_vector(
          self._gizmo_axis
      )
      world_position = self._gizmo_start_world_position + direction * delta
      self._gizmo_preview_world_position = world_position
      self._gizmo_preview_local_position = (
          self._gizmo_parent_matrix.T
          @ (world_position - self._gizmo_parent_position)
      )
    elif self._gizmo_mode == "rotate":
      angle = gizmo.rotation_angle_from_rays(
          self._gizmo_start_ray, ray, self._gizmo_start_display_origin,
          self._gizmo_axis,
          fallback_mouse=(self._gizmo_start_mouse, (mouse[0], mouse[1])),
          basis=self._gizmo_start_world_matrix,
      )
      delta_quat = np.empty(4, dtype=np.float64)
      rotation_axis = self._gizmo_start_world_matrix @ gizmo.axis_vector(
          self._gizmo_axis
      )
      mujoco.mju_axisAngle2Quat(delta_quat, rotation_axis, angle)
      delta_matrix = np.empty((3, 3), dtype=np.float64)
      mujoco.mju_quat2Mat(delta_matrix.reshape(-1), delta_quat)
      world_matrix = delta_matrix @ self._gizmo_start_world_matrix
      local_matrix = self._gizmo_parent_matrix.T @ world_matrix
      quaternion = np.empty(4, dtype=np.float64)
      mujoco.mju_mat2Quat(quaternion, local_matrix.reshape(-1))
      self._gizmo_preview_quaternion = quaternion
    elif self._gizmo_mode == "scale":
      reference = max(float(np.max(np.abs(self._gizmo_start_size))), 0.1)
      factor = gizmo.scale_factor(
          self._gizmo_axis, self._gizmo_start_ray, ray,
          self._gizmo_start_display_origin, reference,
          basis=self._gizmo_start_world_matrix,
      )
      self._gizmo_preview_size = self._scaled_size(node, factor, self._gizmo_axis)

  def _commit_gizmo_drag(self, node):
    if node is None or self._gizmo_preview_local_position is None:
      self._clear_gizmo_drag()
      return
    if self._gizmo_mode == "scale" and node.kind in ("geom", "site"):
      # Scaling must be independent from pose. Sending a stale position or
      # quaternion here can move the geom when the model was recompiled.
      values = {
          "size": tuple(float(v) for v in self._gizmo_preview_size),
      }
    else:
      values = {
          "position": tuple(float(v) for v in self._gizmo_preview_local_position),
          "quaternion": tuple(float(v) for v in self._gizmo_preview_quaternion),
      }
    self._send(messages.EditNodeEvent(
        request_id=self._next_request(),
        kind=node.kind,
        target_path=tuple(node.target_path),
        element_index=node.element_index,
        values=values,
    ))
    self._clear_gizmo_drag()

  def _resolve_selected_body(self):
    if self.viewer is None or self.state.selected_name is None:
      return -1
    return mujoco.mj_name2id(
        self.viewer.model,
        mujoco.mjtObj.mjOBJ_BODY,
        self.state.selected_name,
    )

  def _remove_owned_geoms(self):
    if self.viewer is None:
      return
    owned_ids = {id(geom) for geom in self._owned_geoms}
    self.viewer.extra_geoms[:] = [
        geom for geom in self.viewer.extra_geoms if id(geom) not in owned_ids
    ]
    self._owned_geoms.clear()

  def _add_geom(self, geom):
    self.viewer.extra_geoms.append(geom)
    self._owned_geoms.append(geom)

  def _make_primitive_preview(self, position):
    if self.state.mode == AuthoringMode.PLACE_SPHERE:
      geom_type = mujoco.mjtGeom.mjGEOM_SPHERE
      size = [max(0.01, self.state.size[0]), 0.0, 0.0]
    else:
      geom_type = mujoco.mjtGeom.mjGEOM_BOX
      size = [max(0.01, value) for value in self.state.size]
    geom = mujoco.MjvGeom()
    mujoco.mjv_initGeom(
        geom, int(geom_type), size, position, np.eye(3).reshape(-1),
        [0.25, 0.65, 1.0, 0.38],
    )
    self._add_geom(geom)

  def _make_axis_geom(self, start, end, color, radius=0.025):
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    axis = end - start
    length = float(np.linalg.norm(axis))
    if length <= 1e-9:
      return
    axis /= length
    helper = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(axis, helper))) > 0.9:
      helper = np.array([0.0, 1.0, 0.0])
    side = np.cross(helper, axis)
    side /= max(np.linalg.norm(side), 1e-12)
    up = np.cross(axis, side)
    geom = mujoco.MjvGeom()
    mujoco.mjv_initGeom(
        geom, int(mujoco.mjtGeom.mjGEOM_CYLINDER),
        [radius, length / 2.0, 0.0], (start + end) / 2.0,
        np.column_stack((side, up, axis)).reshape(-1), color,
    )
    self._add_geom(geom)

  def _append_object_preview(self, geom_type, size, position, matrix, rgba,
                             objtype):
    preview_types = {
        int(mujoco.mjtGeom.mjGEOM_PLANE),
        int(mujoco.mjtGeom.mjGEOM_SPHERE),
        int(mujoco.mjtGeom.mjGEOM_CAPSULE),
        int(mujoco.mjtGeom.mjGEOM_ELLIPSOID),
        int(mujoco.mjtGeom.mjGEOM_CYLINDER),
        int(mujoco.mjtGeom.mjGEOM_BOX),
    }
    if int(geom_type) not in preview_types:
      return
    geom = mujoco.MjvGeom()
    color = list(float(value) for value in rgba)
    color[3] = min(color[3], 0.45)
    mujoco.mjv_initGeom(
        geom, int(geom_type), np.asarray(size, dtype=np.float64),
        np.asarray(position, dtype=np.float64),
        np.asarray(matrix, dtype=np.float64).reshape(-1), color,
    )
    geom.objtype = int(objtype)
    geom.objid = -1
    self._add_geom(geom)

  def _preview_world_matrix(self, node):
    if self._gizmo_preview_quaternion is None:
      return self._world_pose(node)[1]
    local_matrix = np.empty((3, 3), dtype=np.float64)
    mujoco.mju_quat2Mat(local_matrix.reshape(-1), self._gizmo_preview_quaternion)
    return self._gizmo_parent_matrix @ local_matrix

  def _draw_object_preview(self, node, position, matrix):
    if self._gizmo_axis is None:
      return
    if node.kind in ("geom", "site"):
      size = (
          self._gizmo_preview_size
          if self._gizmo_preview_size is not None else node.size
      )
      self._append_object_preview(
          node.type_id, size, position, matrix, node.rgba,
          mujoco.mjtObj.mjOBJ_GEOM if node.kind == "geom"
          else mujoco.mjtObj.mjOBJ_SITE,
      )
      return
    if node.kind != "body":
      return
    body_id = self._model_id_for_node(node)
    if body_id < 0:
      return
    for geom_id in range(self.viewer.model.ngeom):
      if int(self.viewer.model.geom_bodyid[geom_id]) != body_id:
        continue
      local_matrix = self.viewer.model.geom_quat[geom_id]
      geom_matrix = np.empty((3, 3), dtype=np.float64)
      mujoco.mju_quat2Mat(geom_matrix.reshape(-1), local_matrix)
      geom_position = position + matrix @ self.viewer.model.geom_pos[geom_id]
      self._append_object_preview(
          self.viewer.model.geom_type[geom_id],
          self.viewer.model.geom_size[geom_id],
          geom_position, matrix @ geom_matrix,
          self.viewer.model.geom_rgba[geom_id], mujoco.mjtObj.mjOBJ_GEOM,
      )
    for site_id in range(self.viewer.model.nsite):
      if int(self.viewer.model.site_bodyid[site_id]) != body_id:
        continue
      local_matrix = self.viewer.model.site_quat[site_id]
      site_matrix = np.empty((3, 3), dtype=np.float64)
      mujoco.mju_quat2Mat(site_matrix.reshape(-1), local_matrix)
      site_position = position + matrix @ self.viewer.model.site_pos[site_id]
      self._append_object_preview(
          self.viewer.model.site_type[site_id],
          self.viewer.model.site_size[site_id],
          site_position, matrix @ site_matrix,
          self.viewer.model.site_rgba[site_id], mujoco.mjtObj.mjOBJ_SITE,
      )

  def _draw_gizmo(self):
    node = self._target_node()
    if node is None:
      return
    pose = self._world_pose(node)
    if pose is None:
      return
    origin = (
        self._gizmo_preview_world_position
        if self._gizmo_preview_world_position is not None
        else pose[0]
    )
    matrix = self._preview_world_matrix(node)
    self._draw_object_preview(node, origin, matrix)
    length = self._gizmo_length(origin)
    display_origin = self._gizmo_display_origin(node, origin, length)
    if self._gizmo_mode == "rotate":
      ring_radius = length * 0.84
      for axis in ("x", "y", "z"):
        color = gizmo.AXIS_COLORS[axis]
        if axis == self._gizmo_axis:
          color = (1.0, 1.0, 0.15, 1.0)
        ring_thickness = max(0.003, min(0.017, length * 0.022))
        for start, end in gizmo.rotation_ring_segments(
            display_origin, axis, radius=ring_radius, segments=96, basis=matrix
        ):
          self._make_axis_geom(start, end, color, radius=ring_thickness)
    else:
      # Keep the thickness proportional to the on-screen handle length.
      # A fixed world-space radius becomes visually huge when zoomed in.
      radius = max(0.003, min(0.022, length * 0.03))
      if self._gizmo_mode == "scale":
        radius *= 1.15
      for axis, (_, end) in gizmo.axis_segments(
          display_origin, length, basis=matrix
      ).items():
        color = gizmo.AXIS_COLORS[axis]
        if axis == self._gizmo_axis:
          color = (1.0, 1.0, 0.15, 1.0)
        self._make_axis_geom(display_origin, end, color, radius=radius)

  def _draw_preview(self):
    self._remove_owned_geoms()
    # Scene Authoring owns no viewport visuals while another Studio tab is
    # active. This also prevents stale gizmos from remaining visible after
    # switching to Inspector or another panel.
    if self.viewer is None or not self._authoring_interaction_enabled():
      return
    if self.state.preview_position is not None and self.state.placing:
      self._make_primitive_preview(self.state.preview_position)
    self._draw_gizmo()

  def _mouse(self):
    coords = picking.viewport_coordinates()
    if coords is None:
      return None
    x, y, aspect = coords
    return x, y, aspect, picking.make_ray(self.viewer, x, y, aspect)

  def _placement_position(self, point):
    if point is None:
      return None
    x, y, z = point
    offset = self.state.size[0]
    if self.state.mode == AuthoringMode.PLACE_BOX:
      offset = self.state.size[2]
    return (x, y, z + max(0.0, offset))

  def _unique_name(self, primitive_type):
    index = 1
    prefix = f"authored_{primitive_type}_"
    while mujoco.mj_name2id(
        self.viewer.model, mujoco.mjtObj.mjOBJ_BODY,
        f"{prefix}{index:03d}",
    ) >= 0:
      index += 1
    return f"{prefix}{index:03d}"

  def _create_from_cursor(self, position):
    primitive_type = (
        "sphere" if self.state.mode == AuthoringMode.PLACE_SPHERE else "box"
    )
    self._send(messages.CreatePrimitiveEvent(
        request_id=self._next_request(), primitive_type=primitive_type,
        name=self._unique_name(primitive_type), position=tuple(position),
        size=tuple(max(0.01, x) for x in self.state.size),
        mass=max(0.001, self.state.mass), dynamic=self.state.dynamic,
    ))

  def _select_or_start_drag(self, mouse):
    x, y, aspect, ray = mouse
    picked = picking.ux.Pick(
        self.viewer.model, self.viewer.data, self.viewer.camera,
        x, y, aspect, self.viewer.vis_options,
    )

    if imgui.IsMouseDoubleClicked(imgui.MouseButton.Left):
      node = self._pick_gizmo_node(picked, ray)
      if node is not None:
        self._activate_gizmo_node(node)
        return True
      return False

    node = self._target_node()
    if node is not None and imgui.IsMouseClicked(imgui.MouseButton.Left):
      pose = self._world_pose(node)
      if pose is not None:
        length = self._gizmo_length(pose[0])
        display_origin = self._gizmo_display_origin(node, pose[0], length)
        # Pick only a small screen-space neighborhood around a visible
        # handle. A fixed world-space threshold becomes huge after zooming.
        pick_threshold = max(0.0025, length * 0.04)
        if self._gizmo_mode == "rotate":
          axis = gizmo.closest_rotation_axis(
              ray[0], ray[1], display_origin, radius=length * 0.84,
              threshold=pick_threshold,
              basis=self._preview_world_matrix(node),
          )
        else:
          axis = gizmo.closest_axis(
              ray[0], ray[1], display_origin, length=length,
              threshold=pick_threshold,
              basis=self._preview_world_matrix(node),
          )
        self.state.hovered_axis = axis
        if axis is not None:
          self._start_gizmo_drag(axis, mouse, ray, node)
          return True

    if imgui.IsMouseClicked(imgui.MouseButton.Left):
      if picked.geom >= 0:
        picked_node = self._node_for_model_object("geom", picked.geom)
        if picked_node is not None:
          self._select_inspected_node(picked_node.key)
          return True
      if picked.body > 0:
        name = mujoco.mj_id2name(
            self.viewer.model, mujoco.mjtObj.mjOBJ_BODY, picked.body
        )
        for scene_node in self._scene_nodes.values():
          if scene_node.kind == "body" and scene_node.name == name:
            self._select_inspected_node(scene_node.key)
            return True
    return False

  def handle_mouse_input(self) -> bool:
    # Only Scene Authoring may own viewport mouse input while its panel is
    # active. Returning False delegates the event to Studio's normal handler.
    if self.viewer is None or not self._authoring_interaction_enabled():
      return False
    io = imgui.GetIO()
    if io.WantCaptureMouse:
      return False
    mouse = self._mouse()
    if mouse is None:
      return False
    x, y, aspect, ray = mouse

    if self._gizmo_axis is not None:
      node = self._target_node()
      if node is not None:
        self._update_gizmo_drag(mouse, ray, node)
      if imgui.IsMouseReleased(imgui.MouseButton.Left):
        self._commit_gizmo_drag(node)
      return True

    if self.state.placing:
      point, _ = picking.pick_point(self.viewer, x, y, aspect)
      self.state.preview_position = self._placement_position(point)
      if imgui.IsMouseClicked(imgui.MouseButton.Left) and point is not None:
        self._create_from_cursor(self.state.preview_position)
      return True

    return self._select_or_start_drag(mouse)

  def handle_keyboard_input(self) -> bool:
    # Do not consume R/S/P/undo/delete shortcuts outside Scene Authoring.
    if (
        self.viewer is None
        or not self._authoring_interaction_enabled()
        or imgui.GetIO().WantCaptureKeyboard
    ):
      return False
    if imgui.IsKeyChordPressed(int(imgui.Key.Ctrl) | int(imgui.Key.Z)):
      self._send(messages.UndoEvent(self._next_request()))
      return True
    if imgui.IsKeyChordPressed(int(imgui.Key.Ctrl) | int(imgui.Key.Y)):
      self._send(messages.RedoEvent(self._next_request()))
      return True
    if imgui.IsKeyPressed(imgui.Key.R, False) and self._target_node() is not None:
      self._pause()
      self._gizmo_mode = "rotate"
      self._clear_gizmo_drag()
      return True
    if imgui.IsKeyPressed(imgui.Key.S, False) and self._target_node() is not None:
      self._pause()
      self._gizmo_mode = "scale"
      self._clear_gizmo_drag()
      return True
    if imgui.IsKeyPressed(imgui.Key.P, False) and self._target_node() is not None:
      self._pause()
      self._gizmo_mode = "translate"
      self._clear_gizmo_drag()
      return True
    if imgui.IsKeyPressed(imgui.Key.T, False) and self._target_node() is not None:
      self._pause()
      self._gizmo_mode = "translate"
      self._clear_gizmo_drag()
      return True
    if imgui.IsKeyPressed(imgui.Key.Escape, False) and self._target_node() is not None:
      self._gizmo_mode = "translate"
      self._clear_gizmo_drag()
      return True
    if imgui.IsKeyChordPressed(imgui.Key.Delete) and self.state.selected_name:
      self._send(messages.DeleteObjectEvent(
          self._next_request(), self.state.selected_name
      ))
      return True
    return False

  def _select_inspected_node(self, key):
    if key in self._scene_nodes:
      self._pause()
      self.state.inspected_node_key = key
      node = self._scene_nodes[key]
      self.state.status = f"Inspected {node.kind}: {node.name or '(unnamed)'}"

  def _draw_scene_node(self, key):
    node = self._scene_nodes.get(key)
    if node is None:
      return
    selected = node.key == self.state.inspected_node_key
    if node.child_keys:
      flags = int(imgui.TreeNodeFlags.SpanAvailWidth)
      if node.parent_key is None:
        flags |= int(imgui.TreeNodeFlags.DefaultOpen)
      if selected:
        flags |= int(imgui.TreeNodeFlags.Selected)
      opened = imgui.TreeNodeEx(node.key, flags, node.label)
      if imgui.IsItemClicked(imgui.MouseButton.Left):
        self._select_inspected_node(node.key)
        # Scene Tree selection is also an authoring selection. This lets the
        # user edit a body directly with P/R/S instead of first selecting a
        # child geom in the viewport.
        self._activate_gizmo_node(node)
      if node.kind == "body":
        self._draw_create_context(node)
      if opened:
        for child_key in node.child_keys:
          self._draw_scene_node(child_key)
        imgui.TreePop()
      return

    if imgui.Selectable(f"{node.label}##{node.key}", selected):
      self._select_inspected_node(node.key)
      self._activate_gizmo_node(node)
    if node.kind == "body":
      self._draw_create_context(node)

  def _draw_scene_tree(self):
    if not self._scene_nodes:
      imgui.TextDisabled("Waiting for scene scan...")
      return
    child_flags = int(imgui.ChildFlags.Borders)
    if imgui.BeginChild("SceneTreeViewport", imgui.Vec2(0, 260), child_flags):
      for key in self._scene_root_keys:
        self._draw_scene_node(key)
    imgui.EndChild()

  def _draft_from_node(self, node):
    draft = {
        "name": node.name,
        "position": tuple(node.position),
    }
    if node.kind in ("body", "geom", "site"):
      draft["quaternion"] = tuple(node.quaternion)
    if node.kind in ("geom", "site", "joint"):
      draft["type_id"] = node.type_id
    if node.kind == "body":
      draft["mass"] = node.mass if node.mass is not None else 0.0
    elif node.kind in ("geom", "site"):
      draft["size"] = tuple(node.size)
      draft["rgba"] = tuple(node.rgba)
      if node.kind == "geom":
        draft["mass"] = node.mass if node.mass is not None else 1.0
    elif node.kind == "joint":
      draft["axis"] = tuple(node.axis)
      draft["range"] = tuple(node.range)
      draft["damping"] = tuple(node.damping)
    draft.update(node.attributes)
    return draft

  def _send_property_edit(self, node, draft):
    values = {
        key: tuple(value) if isinstance(value, (list, tuple)) else value
        for key, value in draft.items()
    }
    self._send(messages.EditNodeEvent(
        request_id=self._next_request(),
        kind=node.kind,
        target_path=tuple(node.target_path),
        element_index=node.element_index,
        values=values,
    ))

  def _draw_property(self, label, value):
    if imgui.BeginTable(f"PropertyRow##{label}", 2):
      imgui.TableNextColumn()
      imgui.TextDisabled(label)
      imgui.TableNextColumn()
      imgui.TextWrapped(value)
      imgui.EndTable()

  def _format_vector(self, values):
    return "[" + ", ".join(f"{float(value):.6g}" for value in values) + "]"

  def _parse_vector(self, text, length):
    cleaned = text.strip().strip("[]()")
    if not cleaned:
      return tuple() if length == 0 else None
    try:
      values = tuple(float(item) for item in cleaned.replace(",", " ").split())
    except ValueError:
      return None
    return values if len(values) == length else None

  def _draw_editable_vector(self, node, draft, label, key):
    values = tuple(draft[key])
    buffer_key = f"{node.key}:{key}"
    buffer = self._property_vector_buffers.setdefault(
        buffer_key, self._format_vector(values)
    )
    if imgui.BeginTable(f"EditVector##{node.key}_{key}", 2):
      imgui.TableNextColumn()
      imgui.TextDisabled(label)
      imgui.TableNextColumn()
      imgui.SetNextItemWidth(-1)
      changed, buffer = imgui.InputText(
          f"##{node.key}_{key}", buffer
      )
      if changed:
        self._property_vector_buffers[buffer_key] = buffer
      if imgui.IsItemDeactivatedAfterEdit():
        parsed = self._parse_vector(buffer, len(values))
        if parsed is not None:
          if key == "quaternion":
            norm = float(np.linalg.norm(parsed))
            parsed = (
                (1.0, 0.0, 0.0, 0.0)
                if norm <= 1e-8 else tuple(value / norm for value in parsed)
            )
          draft[key] = tuple(parsed)
          self._send_property_edit(node, {key: draft[key]})
          self._property_vector_buffers.pop(buffer_key, None)
      imgui.EndTable()

  def _draw_editable_extra(self, node, draft, key):
    value = draft[key]
    label = properties.display_label(key)
    if isinstance(value, bool):
      changed, value = imgui.Checkbox(f"{label}##{node.key}_{key}", value)
      if changed:
        draft[key] = bool(value)
        self._send_property_edit(node, {key: draft[key]})
    elif isinstance(value, str):
      if imgui.BeginTable(f"EditExtraText##{node.key}_{key}", 2):
        imgui.TableNextColumn()
        imgui.TextDisabled(label)
        imgui.TableNextColumn()
        imgui.SetNextItemWidth(-1)
        changed, value = imgui.InputText(f"##{node.key}_{key}", value)
        imgui.EndTable()
      if changed:
        draft[key] = value
        self._send_property_edit(node, {key: value})
    elif isinstance(value, (tuple, list)):
      self._draw_editable_vector(node, draft, label, key)
    elif isinstance(value, int):
      if imgui.BeginTable(f"EditExtraInt##{node.key}_{key}", 2):
        imgui.TableNextColumn()
        imgui.TextDisabled(label)
        imgui.TableNextColumn()
        imgui.SetNextItemWidth(-1)
        changed, value = imgui.InputInt(f"##{node.key}_{key}", int(value))
        imgui.EndTable()
      if changed:
        draft[key] = int(value)
        self._send_property_edit(node, {key: draft[key]})
    else:
      self._draw_editable_scalar(node, draft, label, key)

  def _draw_editable_scalar(self, node, draft, label, key, minimum=None):
    changed = False
    value = float(draft[key])
    if imgui.BeginTable(f"EditScalar##{node.key}_{key}", 2):
      imgui.TableNextColumn()
      imgui.TextDisabled(label)
      imgui.TableNextColumn()
      imgui.SetNextItemWidth(-1)
      changed, value = imgui.InputFloat(
          f"##{node.key}_{key}", float(draft[key]), 0.01, 0.1
      )
      imgui.EndTable()
    if changed:
      value = float(value)
      if minimum is not None:
        value = max(minimum, value)
      draft[key] = value
      self._send_property_edit(node, {key: draft[key]})

  def _draw_editable_name(self, node, draft):
    changed = False
    value = str(draft.get("name", node.name))
    if node.kind == "body" and not node.target_path:
      self._draw_property("Name", node.name or "world")
      return
    if imgui.BeginTable(f"EditName##{node.key}", 2):
      imgui.TableNextColumn()
      imgui.TextDisabled("Name")
      imgui.TableNextColumn()
      imgui.SetNextItemWidth(-1)
      changed, value = imgui.InputText(
          f"##{node.key}_name", str(draft["name"])
      )
      imgui.EndTable()
    if changed:
      draft["name"] = value
      self._send_property_edit(node, {"name": value})

  def _draw_editable_type(self, node, draft):
    options = _GEOM_TYPES if node.kind in ("geom", "site") else _JOINT_TYPES
    values = [value for value, _ in options]
    labels = [label for _, label in options]
    try:
      current = values.index(int(draft["type_id"]))
    except ValueError:
      current = 0
    changed, selected = imgui.Combo(
        f"Type##{node.key}", current, labels
    )
    if changed:
      draft["type_id"] = values[selected]
      self._send_property_edit(node, {"type_id": draft["type_id"]})

  def _draw_property_splitter(self):
    imgui.Separator()
    imgui.InvisibleButton("##PropertyResize", imgui.Vec2(-1, 8))
    if imgui.IsItemHovered() or imgui.IsItemActive():
      imgui.SetMouseCursor(imgui.MouseCursor.ResizeNS)
    if imgui.IsItemActive():
      delta = imgui.GetMouseDragDelta(imgui.MouseButton.Left).y
      self._property_height = max(180.0, min(1100.0, self._property_height + delta))
      imgui.ResetMouseDragDelta(imgui.MouseButton.Left)

  def _draw_inspected_properties(self):
    node = self._scene_nodes.get(self.state.inspected_node_key)
    if node is None:
      imgui.TextDisabled("Select a node in the scene tree to inspect it.")
      return
    draft = self._property_drafts.setdefault(
        node.key, self._draft_from_node(node)
    )
    child_flags = int(imgui.ChildFlags.Borders)
    if not imgui.BeginChild(
        "PropertiesViewport", imgui.Vec2(0, self._property_height), child_flags
    ):
      imgui.EndChild()
      return
    imgui.Text(node.label)
    self._draw_property("Kind", node.kind)
    self._draw_editable_name(node, draft)
    if node.kind in ("geom", "site", "joint"):
      self._draw_editable_type(node, draft)
    if node.kind in ("body", "geom", "site"):
      self._draw_editable_vector(node, draft, "Position", "position")
      self._draw_editable_vector(node, draft, "Quaternion", "quaternion")
    if node.kind in ("geom", "site"):
      self._draw_editable_vector(node, draft, "Size", "size")
      if node.kind == "geom":
        self._draw_editable_scalar(node, draft, "Mass", "mass", 0.001)
      changed = False
      color = list(draft["rgba"])
      if imgui.BeginTable(f"EditColor##{node.key}", 2):
        imgui.TableNextColumn()
        imgui.TextDisabled("Color")
        imgui.TableNextColumn()
        changed, color = imgui.ColorEdit4(
            f"##{node.key}_rgba", draft["rgba"]
        )
        imgui.EndTable()
      if changed:
        draft["rgba"] = tuple(float(value) for value in color)
        self._send_property_edit(node, {"rgba": draft["rgba"]})
    elif node.kind == "body":
      self._draw_editable_scalar(node, draft, "Mass", "mass", 0.0)
    elif node.kind == "joint":
      self._draw_editable_vector(node, draft, "Position", "position")
      self._draw_editable_vector(node, draft, "Axis", "axis")
      self._draw_editable_vector(node, draft, "Range", "range")
      self._draw_editable_vector(node, draft, "Damping", "damping")

    reserved = {
        "position", "quaternion", "type_id", "size", "rgba", "mass",
        "axis", "range", "damping",
    }
    for key in node.attributes:
      if key not in reserved and key in draft:
        self._draw_editable_extra(node, draft, key)
    imgui.EndChild()

  def _selected_body_node(self):
    node = self._scene_nodes.get(self.state.inspected_node_key)
    return node if node is not None and node.kind == "body" else None

  def _unique_node_name(self, kind):
    names = {
        node.name for node in self._scene_nodes.values() if node.name
    }
    prefix = f"authored_{kind}_"
    index = 1
    while f"{prefix}{index:03d}" in names:
      index += 1
    return f"{prefix}{index:03d}"

  def _create_size_defaults(self, kind, type_id):
    if kind == "site":
      defaults = {
          int(mujoco.mjtGeom.mjGEOM_SPHERE): [0.01, 0.0, 0.0],
          int(mujoco.mjtGeom.mjGEOM_CAPSULE): [0.03, 0.1, 0.0],
          int(mujoco.mjtGeom.mjGEOM_ELLIPSOID): [0.1, 0.1, 0.1],
          int(mujoco.mjtGeom.mjGEOM_CYLINDER): [0.03, 0.1, 0.0],
          int(mujoco.mjtGeom.mjGEOM_BOX): [0.1, 0.1, 0.1],
      }
    else:
      defaults = {
          int(mujoco.mjtGeom.mjGEOM_PLANE): [1.0, 1.0, 0.0],
          int(mujoco.mjtGeom.mjGEOM_SPHERE): [0.05, 0.0, 0.0],
          int(mujoco.mjtGeom.mjGEOM_CAPSULE): [0.03, 0.1, 0.0],
          int(mujoco.mjtGeom.mjGEOM_ELLIPSOID): [0.1, 0.1, 0.1],
          int(mujoco.mjtGeom.mjGEOM_CYLINDER): [0.03, 0.1, 0.0],
          int(mujoco.mjtGeom.mjGEOM_BOX): [0.1, 0.1, 0.1],
          int(mujoco.mjtGeom.mjGEOM_MESH): [1.0, 1.0, 1.0],
      }
    return list(defaults.get(type_id, defaults[next(iter(defaults))]))

  def _mesh_asset_names(self):
    if self.viewer is None or self.viewer.model.nmesh <= 0:
      return ()
    names = []
    for mesh_id in range(self.viewer.model.nmesh):
      name = mujoco.mj_id2name(
          self.viewer.model, mujoco.mjtObj.mjOBJ_MESH, mesh_id
      )
      if name:
        names.append(name)
    return tuple(names)

  def _unique_mesh_name(self, mesh_file):
    stem = Path(mesh_file).stem or "mesh"
    stem = re.sub(r"[^A-Za-z0-9_]+", "_", stem).strip("_") or "mesh"
    base = f"authored_mesh_{stem}"
    names = set(self._mesh_asset_names())
    candidate = base
    index = 1
    while candidate in names:
      index += 1
      candidate = f"{base}_{index:03d}"
    return candidate

  def _open_create_dialog(self, kind, parent_node):
    self._pause()
    self._create_dialog_kind = kind
    self._create_dialog_parent_path = tuple(parent_node.target_path)
    options = _CREATE_GEOM_TYPES if kind == "geom" else _CREATE_SITE_TYPES
    self._create_dialog_type_id = options[0][0]
    self._create_dialog_mesh_file = ""
    self._create_dialog_size = self._create_size_defaults(
        kind, self._create_dialog_type_id
    )
    self._create_dialog_mass = 1.0
    self._create_dialog_pending = True
    self._create_error = ""

  def _open_mesh_browser(self):
    current = Path(self._create_dialog_mesh_file).expanduser()
    if not current.is_absolute():
      source = self.state.scene_source_path
      base = Path(source).expanduser().parent if source else Path.cwd()
      current = base / current
    directory = current if current.is_dir() else current.parent
    self._mesh_browser_dir = str(directory)
    self._mesh_browser_error = ""
    self._mesh_browser_pending = True

  def _draw_mesh_file_browser(self):
    if self._mesh_browser_pending:
      imgui.OpenPopup("Select Mesh File##SceneAuthoring")
      self._mesh_browser_pending = False
    if not imgui.BeginPopup("Select Mesh File##SceneAuthoring"):
      return

    directory = Path(self._mesh_browser_dir or Path.cwd()).expanduser()
    if not directory.is_dir():
      directory = Path.cwd()
      self._mesh_browser_dir = str(directory)

    changed, value = imgui.InputText("Folder##MeshBrowser", str(directory))
    if changed:
      candidate = Path(value).expanduser()
      if candidate.is_dir():
        self._mesh_browser_dir = str(candidate)
        self._mesh_browser_error = ""
      else:
        self._mesh_browser_error = "Folder does not exist."
    if imgui.Button("Up##MeshBrowser", imgui.Vec2(90, 0)):
      self._mesh_browser_dir = str(directory.parent)
      self._mesh_browser_error = ""
    imgui.SameLine()
    imgui.TextWrapped(str(directory))

    if self._mesh_browser_error:
      imgui.TextColored(
          imgui.Vec4(1.0, 0.35, 0.25, 1.0), self._mesh_browser_error
      )

    if imgui.BeginChild(
        "MeshBrowserFiles", imgui.Vec2(520, 260), True
    ):
      try:
        entries = sorted(
            directory.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower())
        )
      except OSError as exc:
        entries = []
        self._mesh_browser_error = str(exc)
      extensions = {".obj", ".stl", ".ply", ".msh", ".mesh"}
      for entry in entries:
        if entry.is_dir():
          if imgui.Selectable(f"[DIR] {entry.name}##MeshBrowser"):
            self._mesh_browser_dir = str(entry)
            self._mesh_browser_error = ""
        elif entry.suffix.lower() in extensions:
          if imgui.Selectable(f"{entry.name}##MeshBrowser"):
            self._create_dialog_mesh_file = str(entry)
            imgui.CloseCurrentPopup()
      imgui.EndChild()

    if imgui.Button("Cancel##MeshBrowser", imgui.Vec2(100, 0)):
      imgui.CloseCurrentPopup()
    imgui.EndPopup()

  def _create_node(self, kind, parent_path=None, values_override=None):
    if parent_path is None:
      parent = self._selected_body_node()
      if kind != "body" and parent is None:
        self._create_error = (
            "Select a body in Scene before creating a geom or site."
        )
        return
      parent_path = tuple(parent.target_path) if parent is not None else ()
    else:
      parent_path = tuple(parent_path)

    name = self._unique_node_name(kind)
    if values_override is not None:
      values = dict(values_override)
    elif kind == "body":
      values = {
          "position": (0.0, 0.0, 0.0),
          "quaternion": (1.0, 0.0, 0.0, 0.0),
          "dynamic": False,
          "with_default_site": True,
          "default_site_name": self._unique_node_name("site"),
      }
    else:
      values = {
          "position": (0.0, 0.0, 0.0),
          "quaternion": (1.0, 0.0, 0.0, 0.0),
          "type": int(mujoco.mjtGeom.mjGEOM_BOX),
          "size": (0.1, 0.1, 0.1),
          "mass": 1.0,
          "rgba": (
              (0.35, 0.65, 1.0, 1.0)
              if kind == "geom" else (1.0, 0.3, 0.2, 1.0)
          ),
      }
    self._pending_create_name = name
    self._create_error = ""
    self._send(messages.CreateNodeEvent(
        request_id=self._next_request(),
        kind=kind,
        parent_path=parent_path,
        name=name,
        values=values,
    ))

  def _draw_create_context(self, node):
    popup_id = f"Create##{node.key}"
    if imgui.IsItemClicked(imgui.MouseButton.Right):
      self._select_inspected_node(node.key)
      imgui.OpenPopup(popup_id)
    if not imgui.BeginPopup(popup_id):
      return
    imgui.Text("Create")
    imgui.TextDisabled(f"Child of {node.name or '(unnamed body)'}")
    imgui.Separator()
    if imgui.Button("Body", imgui.Vec2(180, 0)):
      self._create_node("body", parent_path=node.target_path)
      imgui.CloseCurrentPopup()
    if imgui.Button("Geom", imgui.Vec2(180, 0)):
      self._open_create_dialog("geom", node)
      imgui.CloseCurrentPopup()
    if imgui.Button("Site", imgui.Vec2(180, 0)):
      self._open_create_dialog("site", node)
      imgui.CloseCurrentPopup()
    imgui.EndPopup()

  def _draw_create_dialog(self):
    if self._create_dialog_pending:
      imgui.OpenPopup("Create Element##SceneAuthoring")
      self._create_dialog_pending = False
    if not imgui.BeginPopup("Create Element##SceneAuthoring"):
      return

    kind = self._create_dialog_kind
    options = _CREATE_GEOM_TYPES if kind == "geom" else _CREATE_SITE_TYPES
    values = [value for value, _ in options]
    labels = [label for _, label in options]
    try:
      current = values.index(self._create_dialog_type_id)
    except ValueError:
      current = 0
    changed, selected = imgui.Combo("Type", current, labels)
    if changed:
      self._create_dialog_type_id = values[selected]
      self._create_dialog_size = self._create_size_defaults(
          kind, self._create_dialog_type_id
      )

    is_mesh = self._create_dialog_type_id == int(mujoco.mjtGeom.mjGEOM_MESH)
    if is_mesh:
      changed, mesh_file = imgui.InputText(
          "Mesh file", self._create_dialog_mesh_file
      )
      if changed:
        self._create_dialog_mesh_file = mesh_file
      imgui.SameLine()
      if imgui.Button("Browse...##MeshBrowser", imgui.Vec2(100, 0)):
        self._open_mesh_browser()
      imgui.TextDisabled(
          "Path to an OBJ, STL, PLY or supported MuJoCo mesh file."
      )

    descriptions = {
        "plane": "Half-size X/Y",
        "sphere": "Radius",
        "capsule": "Radius / half-length",
        "ellipsoid": "Radii X/Y/Z",
        "cylinder": "Radius / half-length",
        "box": "Half-extents X/Y/Z",
        "mesh": "Mesh scale X/Y/Z",
    }
    type_label = labels[selected].lower()
    imgui.TextDisabled(descriptions.get(type_label, "Size"))
    for index, label in enumerate(("Size X", "Size Y", "Size Z")):
      imgui.SetNextItemWidth(180)
      changed, value = imgui.InputFloat(
          f"{label}##create_{kind}_{index}",
          self._create_dialog_size[index], 0.01, 0.1
      )
      if changed:
        self._create_dialog_size[index] = max(0.0, float(value))
    if kind == "geom":
      imgui.SetNextItemWidth(180)
      changed, value = imgui.InputFloat(
          "Mass##create_geom_mass", self._create_dialog_mass, 0.01, 0.1
      )
      if changed:
        self._create_dialog_mass = max(0.001, float(value))

    if self._create_error:
      imgui.TextColored(
          imgui.Vec4(1.0, 0.35, 0.25, 1.0), self._create_error
      )
    if imgui.Button("Create", imgui.Vec2(100, 0)):
      mesh_file = self._create_dialog_mesh_file.strip()
      if is_mesh and not mesh_file:
        self._create_error = "Choose a mesh file before creating the geom."
      else:
        values = {
            "position": (0.0, 0.0, 0.0),
            "quaternion": (1.0, 0.0, 0.0, 0.0),
            "type": self._create_dialog_type_id,
            "size": tuple(self._create_dialog_size),
            "rgba": (
                (0.35, 0.65, 1.0, 1.0)
                if kind == "geom" else (1.0, 0.3, 0.2, 1.0)
            ),
        }
        if kind == "geom":
          values["mass"] = self._create_dialog_mass
          if is_mesh:
            values["mesh_file"] = mesh_file
            values["meshname"] = self._unique_mesh_name(mesh_file)
        self._create_node(
            kind,
            parent_path=self._create_dialog_parent_path,
            values_override=values,
        )
        imgui.CloseCurrentPopup()
    imgui.SameLine()
    if imgui.Button("Cancel", imgui.Vec2(100, 0)):
      imgui.CloseCurrentPopup()
    imgui.EndPopup()
    self._draw_mesh_file_browser()

  def _open_save_popup(self):
    self._pause()
    if not self._save_folder:
      source = Path(self.state.scene_source_path) if self.state.scene_source_path else Path.cwd()
      self._save_folder = str(source.parent)
    if not self._save_filename:
      source = Path(self.state.scene_source_path) if self.state.scene_source_path else Path("scene.xml")
      self._save_filename = f"{source.stem}_edited{source.suffix or '.xml'}"
    self._save_error = ""
    self._save_popup_pending = True

  def _save_target_path(self):
    folder = self._save_folder.strip()
    filename = self._save_filename.strip()
    if not filename:
      return ""
    if Path(filename).suffix.lower() not in (".xml", ".mjcf"):
      filename = f"{filename}.xml"
    return str(Path(folder).expanduser() / filename) if folder else filename

  def _draw_save_popup(self):
    if not imgui.BeginPopup("Save File##SceneAuthoring"):
      return
    imgui.Text("Choose where to save the edited MJCF.")
    imgui.Separator()
    changed, value = imgui.InputText("Folder", self._save_folder)
    if changed:
      self._save_folder = value
    changed, value = imgui.InputText("File name", self._save_filename)
    if changed:
      self._save_filename = value
    target = self._save_target_path()
    imgui.TextDisabled(f"Path: {target or '(enter a file name)'}")
    if self._save_error:
      imgui.TextColored(
          imgui.Vec4(1.0, 0.35, 0.35, 1.0), self._save_error
      )
    if imgui.Button("Save", imgui.Vec2(120, 0)):
      if not target:
        self._save_error = "File name cannot be empty."
      else:
        self._send(messages.SaveSceneEvent(
            self._next_request(), target
        ))
        imgui.CloseCurrentPopup()
    imgui.SameLine()
    if imgui.Button("Cancel", imgui.Vec2(120, 0)):
      imgui.CloseCurrentPopup()
    imgui.EndPopup()

  def _draw_selection_section(self):
    viewport_selection = self.state.selected_name or "(none)"
    tree_selection = self._scene_nodes.get(self.state.inspected_node_key)
    tree_label = tree_selection.label if tree_selection else "(none)"
    imgui.TextWrapped(f"Viewport: {viewport_selection}")
    imgui.TextWrapped(f"Tree: {tree_label}")

    if imgui.BeginTable(
        "AuthoringActions", 3,
        int(imgui.TableFlags.SizingStretchSame),
    ):
      for label, event_type in (
          ("Undo", messages.UndoEvent),
          ("Redo", messages.RedoEvent),
      ):
        imgui.TableNextColumn()
        if imgui.Button(label, imgui.Vec2(-1, 0)):
          self._send(event_type(self._next_request()))
      imgui.TableNextColumn()
      if imgui.Button("Delete", imgui.Vec2(-1, 0)):
        if self.state.selected_name:
          self._send(messages.DeleteObjectEvent(
              self._next_request(), self.state.selected_name
          ))
      imgui.EndTable()

    if imgui.Button("Save File", imgui.Vec2(-1, 0)):
      self._open_save_popup()
    if self._last_saved_path:
      imgui.TextWrapped(f"Saved: {self._last_saved_path}")
    if self._save_error:
      imgui.TextColored(
          imgui.Vec4(1.0, 0.35, 0.35, 1.0),
          f"Save error: {self._save_error}",
      )

  def _draw_panel(self):
    visible = imgui.Begin("Scene Authoring")
    # Begin() may still return true for a window hosted in an inactive dock
    # tab. Focus identifies the selected tab, while RootAndChildWindows keeps
    # Property/Scene child widgets associated with Scene Authoring.
    focused = imgui.IsWindowFocused(
        int(imgui.FocusedFlags.RootAndChildWindows)
    )
    # Clicking Studio's Run/Pause controls moves ImGui focus away from this
    # window, but the Scene Authoring dock tab is still selected. Keep the
    # session active while the window remains visible; an inactive dock tab
    # reports visible=False and ends the session.
    panel_active = bool(visible and (focused or self._scene_authoring_active))
    self._sync_authoring_pause(panel_active)
    if not visible:
      imgui.End()
      return

    if self._save_popup_pending:
      imgui.OpenPopup("Save File##SceneAuthoring")
      self._save_popup_pending = False

    section_flags = (
        int(imgui.TreeNodeFlags.Framed)
        | int(imgui.TreeNodeFlags.SpanAvailWidth)
    )
    if imgui.TreeNodeEx("Selection", section_flags):
      self._draw_selection_section()
      imgui.TreePop()
    if imgui.TreeNodeEx("Scene", section_flags):
      if self.state.scene_source_path:
        imgui.TextWrapped(self.state.scene_source_path)
      self._draw_scene_tree()
      imgui.TreePop()
    if imgui.TreeNodeEx("Property", section_flags):
      self._draw_inspected_properties()
      self._draw_property_splitter()
      imgui.TreePop()

    self._draw_save_popup()
    self._draw_create_dialog()
    imgui.End()

  @studio_messages.handler
  def _on_build_gui(self, _: studio_messages.BuildGuiEvent) -> None:
    self._draw_panel()
    self._draw_preview()
