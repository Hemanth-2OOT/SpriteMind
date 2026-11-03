"""Tests for PixelScript parser, AST, and validator."""
import pytest
from spritemind.pixelscript.ast import ActionType, PixelAction, PixelScriptProgram
from spritemind.pixelscript.parser import PixelScriptParser
from spritemind.pixelscript.validator import PixelScriptValidator
from spritemind.pixelscript.serializer import PixelScriptSerializer


def test_pixelscript_parser_and_serializer():
    script = """# META: width=32 height=32
PALETTE 1 #ff0055ff
LAYER 0 base 1.00
PART torso none 16 16
RECT 12 12 8 8 1 1
PIXEL 16 16 1
LINE 10 10 20 20 1
FILL 14 14 1
ERASE 15 15 2 2
END PART
END PROGRAM"""
    prog = PixelScriptParser.parse_text(script)
    assert len(prog.actions) == 10
    assert prog.metadata["width"] == "32"
    assert prog.metadata["height"] == "32"

    serialized = PixelScriptSerializer.serialize_program(prog)
    assert "RECT 12 12 8 8 1 1" in serialized
    assert "PIXEL 16 16 1" in serialized


def test_pixelscript_validator():
    script = """
RECT 4 4 10 10 1 1
RECT 2 2 -5 10 1 1
PIXEL 100 100 1
REUSE nonexistent_part 10 10 0 0
"""
    prog = PixelScriptParser.parse_text(script)
    report = PixelScriptValidator.validate(prog, canvas_width=32, canvas_height=32)

    assert not report.is_valid
    assert report.invalid_action_count > 0
    assert len(report.errors) > 0
    assert any("RECT width" in err for err in report.errors)
    assert any("nonexistent_part" in w for w in report.warnings)
