"""Plugin-owned undo/redo command history."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Command:
  kind: str
  payload: dict


class CommandHistory:
  def __init__(self):
    self._undo: list[Command] = []
    self._redo: list[Command] = []

  def clear(self):
    self._undo.clear()
    self._redo.clear()

  def push(self, command: Command):
    self._undo.append(command)
    self._redo.clear()

  def undo(self):
    if not self._undo:
      return None
    command = self._undo.pop()
    self._redo.append(command)
    return command

  def redo(self):
    if not self._redo:
      return None
    command = self._redo.pop()
    self._undo.append(command)
    return command
