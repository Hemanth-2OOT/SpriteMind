"""PixelScript Abstract Syntax Tree and action definitions.

Defines the core vocabulary for SpriteMind drawing actions:
PIXEL, RECT, LINE, FILL, PALETTE, LAYER, PART, REUSE, TRANSFORM, ERASE, END.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any, Union


class ActionType(str, Enum):
    PALETTE = "PALETTE"
    LAYER = "LAYER"
    PART = "PART"
    PIXEL = "PIXEL"
    RECT = "RECT"
    LINE = "LINE"
    FILL = "FILL"
    ERASE = "ERASE"
    REUSE = "REUSE"
    TRANSFORM = "TRANSFORM"
    END = "END"


@dataclass
class PixelAction:
    action_type: ActionType
    # Parameters vary by action_type:
    # PALETTE: index (int), color (str '#RRGGBB' or '#RRGGBBAA' or list/tuple [r,g,b,a])
    # LAYER: id (int), name (str), opacity (float 0.0-1.0)
    # PART: name (str), parent (Optional[str]), anchor_x (int), anchor_y (int)
    # PIXEL: x (int), y (int), color (int or str)
    # RECT: x (int), y (int), w (int), h (int), color (int or str), filled (bool)
    # LINE: x0 (int), y0 (int), x1 (int), y1 (int), color (int or str)
    # FILL: x (int), y (int), color (int or str)
    # ERASE: x (int), y (int), w (int, default 1), h (int, default 1)
    # REUSE: part_name (str), dest_x (int), dest_y (int), flip_x (bool), flip_y (bool)
    # TRANSFORM: part_name (str), dx (int), dy (int), rot_deg (int)
    # END: marker (optional, e.g. 'PART' or 'PROGRAM')
    params: Dict[str, Any] = field(default_factory=dict)
    raw_text: Optional[str] = None
    line_number: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.action_type.value,
            "params": self.params,
            "raw_text": self.raw_text,
            "line_number": self.line_number,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PixelAction":
        return cls(
            action_type=ActionType(data["type"]),
            params=data.get("params", {}),
            raw_text=data.get("raw_text"),
            line_number=data.get("line_number"),
        )


@dataclass
class PixelScriptProgram:
    actions: List[PixelAction] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": self.metadata,
            "actions": [a.to_dict() for a in self.actions],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PixelScriptProgram":
        return cls(
            actions=[PixelAction.from_dict(a) for a in data.get("actions", [])],
            metadata=data.get("metadata", {}),
        )
