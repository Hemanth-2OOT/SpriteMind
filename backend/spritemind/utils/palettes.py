"""Standard pixel art color palettes."""
from typing import Dict, List, Tuple

# RGBA tuples: (R, G, B, A)
PALETTES: Dict[str, List[Tuple[int, int, int, int]]] = {
    "pico8": [
        (0, 0, 0, 0),         # 0: Transparent
        (0, 0, 0, 255),       # 1: Black
        (29, 43, 83, 255),    # 2: Dark Blue
        (126, 37, 83, 255),   # 3: Dark Purple
        (0, 135, 81, 255),    # 4: Dark Green
        (171, 82, 54, 255),   # 5: Brown
        (95, 87, 79, 255),    # 6: Dark Gray
        (194, 195, 199, 255), # 7: Light Gray
        (255, 241, 232, 255), # 8: White
        (255, 0, 77, 255),    # 9: Red
        (255, 163, 0, 255),   # 10: Orange
        (255, 236, 39, 255),  # 11: Yellow
        (0, 228, 54, 255),    # 12: Green
        (41, 173, 255, 255),  # 13: Blue
        (131, 118, 156, 255), # 14: Indigo
        (255, 119, 168, 255), # 15: Pink
        (255, 204, 170, 255), # 16: Peach
    ],
    "gameboy": [
        (0, 0, 0, 0),         # 0: Transparent
        (15, 56, 15, 255),    # 1: Darkest Green
        (48, 98, 48, 255),    # 2: Dark Green
        (139, 172, 15, 255),  # 3: Light Green
        (155, 188, 15, 255),  # 4: Lightest Green
    ],
    "dawnbringer16": [
        (0, 0, 0, 0),         # 0: Transparent
        (20, 12, 28, 255),    # 1: Black
        (68, 36, 52, 255),    # 2: Deep purple
        (48, 52, 109, 255),   # 3: Navy blue
        (78, 74, 78, 255),    # 4: Dark grey
        (133, 76, 48, 255),   # 5: Brown
        (52, 101, 36, 255),   # 6: Deep green
        (208, 70, 72, 255),   # 7: Red
        (117, 113, 97, 255),  # 8: Grey
        (89, 125, 206, 255),  # 9: Light blue
        (210, 125, 44, 255),  # 10: Orange
        (133, 149, 161, 255), # 11: Slate grey
        (109, 170, 44, 255),  # 12: Light green
        (210, 170, 153, 255), # 13: Peach
        (109, 194, 202, 255), # 14: Cyan
        (218, 212, 94, 255),  # 15: Yellow
        (222, 238, 214, 255), # 16: Cream white
    ],
    "cyberpunk": [
        (0, 0, 0, 0),         # 0: Transparent
        (10, 10, 20, 255),    # 1: Night Black
        (255, 0, 85, 255),    # 2: Neon Pink
        (0, 240, 255, 255),   # 3: Neon Cyan
        (255, 222, 0, 255),   # 4: Neon Yellow
        (113, 28, 145, 255),  # 5: Cyber Violet
        (0, 255, 128, 255),   # 6: Acid Green
        (34, 40, 49, 255),    # 7: Metal Dark
        (238, 238, 238, 255), # 8: Pure White
    ],
    "endesga32": [
        (0, 0, 0, 0),
        (190, 74, 47, 255), (215, 118, 67, 255), (234, 212, 170, 255), (228, 166, 114, 255),
        (184, 111, 80, 255), (115, 62, 57, 255), (62, 39, 49, 255), (162, 38, 51, 255),
        (228, 59, 68, 255), (247, 118, 34, 255), (254, 174, 52, 255), (254, 231, 97, 255),
        (99, 199, 77, 255), (62, 137, 72, 255), (38, 92, 66, 255), (25, 60, 62, 255),
        (18, 78, 137, 255), (0, 153, 219, 255), (44, 232, 245, 255), (255, 255, 255, 255),
        (192, 203, 220, 255), (139, 155, 180, 255), (90, 105, 136, 255), (58, 68, 102, 255),
        (38, 43, 68, 255), (24, 20, 37, 255), (255, 0, 68, 255), (255, 119, 168, 255),
        (24, 20, 37, 255), (20, 12, 28, 255), (12, 8, 16, 255)
    ],
}


def hex_to_rgba(hex_str: str) -> Tuple[int, int, int, int]:
    """Convert #RGB, #RGBA, #RRGGBB, or #RRGGBBAA to RGBA tuple."""
    hex_str = hex_str.lstrip("#")
    if len(hex_str) == 3:
        r, g, b = [int(c * 2, 16) for c in hex_str]
        return (r, g, b, 255)
    elif len(hex_str) == 4:
        r, g, b, a = [int(c * 2, 16) for c in hex_str]
        return (r, g, b, a)
    elif len(hex_str) == 6:
        r = int(hex_str[0:2], 16)
        g = int(hex_str[2:4], 16)
        b = int(hex_str[4:6], 16)
        return (r, g, b, 255)
    elif len(hex_str) == 8:
        r = int(hex_str[0:2], 16)
        g = int(hex_str[2:4], 16)
        b = int(hex_str[4:6], 16)
        a = int(hex_str[6:8], 16)
        return (r, g, b, a)
    raise ValueError(f"Invalid hex color string: #{hex_str}")


def rgba_to_hex(rgba: Tuple[int, int, int, int]) -> str:
    r, g, b, a = rgba
    return f"#{r:02x}{g:02x}{b:02x}{a:02x}"
