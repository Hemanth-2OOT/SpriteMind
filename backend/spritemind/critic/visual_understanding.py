"""Visual Intelligence & Morphological Analysis Engine for SpriteMind.

Inspects rendered sprite canvases, decomposes connected components and anatomy
across humanoid and non-humanoid taxonomies, detects defects, and formulates repairs.
"""
from typing import Dict, Any, List, Tuple, Optional, Set
from collections import deque
import numpy as np

from spritemind.utils.palettes import PALETTES
from spritemind.renderer.engine import RenderResult
from spritemind.pixelscript.ast import PixelScriptProgram


# Supported character categories with structural profile expectations
HUMANOID_CATEGORIES = {"knight", "zombie", "rogue", "wizard", "skeleton", "goblin"}
NON_HUMANOID_CATEGORIES = {
    "slime": "amorphous",
    "robot": "mechanical",
    "animal": "quadruped",
    "fantasy_creature": "hybrid",
}


def find_connected_components(alpha: np.ndarray) -> List[Dict[str, Any]]:
    """Identifies all 8-connected components in the alpha mask."""
    h, w = alpha.shape
    visited = np.zeros((h, w), dtype=bool)
    components = []

    for y in range(h):
        for x in range(w):
            if alpha[y, x] > 0 and not visited[y, x]:
                # Start BFS
                comp_pixels = []
                queue = deque([(x, y)])
                visited[y, x] = True

                min_x, max_x = x, x
                min_y, max_y = y, y

                while queue:
                    cx, cy = queue.popleft()
                    comp_pixels.append((cx, cy))
                    min_x = min(min_x, cx)
                    max_x = max(max_x, cx)
                    min_y = min(min_y, cy)
                    max_y = max(max_y, cy)

                    # 8-connectivity
                    for dx in (-1, 0, 1):
                        for dy in (-1, 0, 1):
                            if dx == 0 and dy == 0:
                                continue
                            nx, ny = cx + dx, cy + dy
                            if 0 <= nx < w and 0 <= ny < h:
                                if alpha[ny, nx] > 0 and not visited[ny, nx]:
                                    visited[ny, nx] = True
                                    queue.append((nx, ny))

                components.append({
                    "pixels": comp_pixels,
                    "count": len(comp_pixels),
                    "bbox": {"min_x": min_x, "min_y": min_y, "max_x": max_x, "max_y": max_y},
                    "center": (round(sum(p[0] for p in comp_pixels) / len(comp_pixels), 1),
                               round(sum(p[1] for p in comp_pixels) / len(comp_pixels), 1)),
                })

    # Sort components by size descending
    components.sort(key=lambda c: c["count"], reverse=True)
    return components


def detect_interior_holes(alpha: np.ndarray, main_component_bbox: Optional[Dict[str, int]] = None) -> List[Tuple[int, int]]:
    """Detects unexpected transparent holes fully enclosed inside solid sprite regions.
    Uses border-seed flood-fill to differentiate background from interior enclosed voids.
    """
    h, w = alpha.shape
    # Outside background mask
    outside = np.zeros((h, w), dtype=bool)
    queue = deque()

    # Seed all canvas perimeter borders
    for x in range(w):
        if alpha[0, x] == 0:
            outside[0, x] = True
            queue.append((x, 0))
        if alpha[h - 1, x] == 0:
            outside[h - 1, x] = True
            queue.append((x, h - 1))
    for y in range(h):
        if alpha[y, 0] == 0:
            outside[y, 0] = True
            queue.append((0, y))
        if alpha[y, w - 1] == 0:
            outside[y, w - 1] = True
            queue.append((w - 1, y))

    # Flood fill exterior transparent background
    while queue:
        cx, cy = queue.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < w and 0 <= ny < h:
                if alpha[ny, nx] == 0 and not outside[ny, nx]:
                    outside[ny, nx] = True
                    queue.append((nx, ny))

    # Any pixel with alpha == 0 that was NOT reached from the exterior is an enclosed hole!
    holes = []
    min_x = main_component_bbox.get("min_x", 0) if main_component_bbox else 0
    max_x = main_component_bbox.get("max_x", w - 1) if main_component_bbox else w - 1
    min_y = main_component_bbox.get("min_y", 0) if main_component_bbox else 0
    max_y = main_component_bbox.get("max_y", h - 1) if main_component_bbox else h - 1

    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            if alpha[y, x] == 0 and not outside[y, x]:
                # Verify that it has at least 3 opaque neighbors in cardinal/diagonal directions
                opaque_neighbors = sum(
                    1 for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                    if (dx != 0 or dy != 0) and 0 <= x + dx < w and 0 <= y + dy < h and alpha[y + dy, x + dx] > 0
                )
                if opaque_neighbors >= 4:
                    holes.append((x, y))

    return holes


def check_palette_violations(comp_array: np.ndarray, palette_name: str) -> List[Dict[str, Any]]:
    """Detects pixels whose RGB colors deviate from the designated palette."""
    palette_colors = PALETTES.get(palette_name, PALETTES.get("pico8", []))
    valid_rgb_set = {tuple(c[:3]) for c in palette_colors if c[3] > 0}

    h, w, _ = comp_array.shape
    violations = []

    for y in range(h):
        for x in range(w):
            r, g, b, a = comp_array[y, x]
            if a > 0:
                pixel_rgb = (int(r), int(g), int(b))
                if pixel_rgb not in valid_rgb_set:
                    # Find closest valid palette color index
                    closest_idx = 1
                    min_dist = float("inf")
                    for p_idx, p_col in enumerate(palette_colors):
                        if p_idx == 0:
                            continue
                        dist = sum((pixel_rgb[k] - p_col[k]) ** 2 for k in range(3))
                        if dist < min_dist:
                            min_dist = dist
                            closest_idx = p_idx

                    violations.append({
                        "x": x,
                        "y": y,
                        "current_rgb": pixel_rgb,
                        "closest_palette_idx": closest_idx,
                    })

    return violations


class VisualUnderstandingPipeline:
    """Comprehensive visual analyzer for pixel art character sprites."""

    def __init__(self):
        pass

    def analyze(
        self,
        render_res: RenderResult,
        category: str = "knight",
        palette_name: str = "pico8",
        prog: Optional[PixelScriptProgram] = None,
    ) -> Dict[str, Any]:
        comp = render_res.composite_array
        alpha = comp[:, :, 3]
        total_pixels = int(np.count_nonzero(alpha > 0))
        h, w = alpha.shape

        components = find_connected_components(alpha)
        main_comp = components[0] if components else None
        main_bbox = main_comp["bbox"] if main_comp else {"min_x": 0, "min_y": 0, "max_x": 0, "max_y": 0}

        defects: List[Dict[str, Any]] = []
        defect_counter = 0
        parts_dict = render_res.canvas.parts if (hasattr(render_res, "canvas") and render_res.canvas and hasattr(render_res.canvas, "parts")) else {}

        # 1. Stray Pixel Detection & Disconnected Island Components
        for comp_item in components:
            if main_comp and comp_item != main_comp:
                cb = comp_item["bbox"]
                mb = main_bbox
                dist_x = max(0, mb["min_x"] - cb["max_x"], cb["min_x"] - mb["max_x"])
                dist_y = max(0, mb["min_y"] - cb["max_y"], cb["min_y"] - mb["max_y"])
                dist = max(dist_x, dist_y)

                # Floating isolated stray pixel (1 or 2 pixels)
                if comp_item["count"] <= 2:
                    is_particle = category in ("wizard", "slime") and dist <= 2
                    if dist >= 2 and not is_particle:
                        for px, py in comp_item["pixels"]:
                            defects.append({
                                "id": f"defect_{defect_counter}",
                                "type": "stray_pixel",
                                "category": category,
                                "severity": "moderate",
                                "confidence": 0.98,
                                "description": f"Isolated floating stray pixel at ({px}, {py}) separated by {dist}px from main body",
                                "bbox": {"min_x": px, "min_y": py, "max_x": px, "max_y": py},
                                "affected_pixels": [[px, py]],
                                "suggested_fix": f"ERASE {px} {py} 1 1",
                            })
                            defect_counter += 1
                elif dist > 3:
                    # Floating island / disconnected component
                    shift_x = -(cb["min_x"] - mb["max_x"]) if cb["min_x"] > mb["max_x"] else (mb["min_x"] - cb["max_x"] if cb["max_x"] < mb["min_x"] else 0)
                    shift_y = -(cb["min_y"] - mb["max_y"]) if cb["min_y"] > mb["max_y"] else (mb["min_y"] - cb["max_y"] if cb["max_y"] < mb["min_y"] else 0)
                    shift_x = max(-6, min(6, shift_x))
                    shift_y = max(-6, min(6, shift_y))
                    defects.append({
                        "id": f"defect_{defect_counter}",
                        "type": "disconnected_part",
                        "category": category,
                        "severity": "critical",
                        "confidence": 0.95,
                        "description": f"Disconnected floating component of {comp_item['count']}px separated by {dist}px from torso",
                        "bbox": cb,
                        "affected_pixels": [list(p) for p in comp_item["pixels"]],
                        "suggested_fix": f"TRANSFORM wings {shift_x} {shift_y} 0" if "wings" in parts_dict else f"ERASE {cb['min_x']} {cb['min_y']} {cb['max_x']-cb['min_x']+1} {cb['max_y']-cb['min_y']+1}",
                    })
                    defect_counter += 1

        # 2. Unexpected Interior Holes
        holes = detect_interior_holes(alpha, main_bbox)
        if holes:
            # Group contiguous hole pixels
            hole_mask = np.zeros((h, w), dtype=bool)
            for hx, hy in holes:
                hole_mask[hy, hx] = True
            hole_components = find_connected_components(hole_mask.astype(np.uint8) * 255)

            for hc in hole_components:
                # Suggest filling with dominant adjacent color
                hx, hy = hc["pixels"][0]
                # Sample color from top or left adjacent opaque pixel
                fill_color_idx = 1
                for dx, dy in ((0, -1), (-1, 0), (0, 1), (1, 0)):
                    nx, ny = hx + dx, hy + dy
                    if 0 <= nx < w and 0 <= ny < h and alpha[ny, nx] > 0:
                        # Find closest palette color index
                        pix_rgb = tuple(comp[ny, nx][:3])
                        pal = PALETTES.get(palette_name, PALETTES.get("pico8", []))
                        for p_i, p_col in enumerate(pal):
                            if tuple(p_col[:3]) == pix_rgb:
                                fill_color_idx = p_i
                                break
                        break

                defects.append({
                    "id": f"defect_{defect_counter}",
                    "type": "unexpected_hole",
                    "category": category,
                    "severity": "minor" if hc["count"] <= 2 else "moderate",
                    "confidence": 0.90,
                    "description": f"Internal void / unexpected transparent hole of {hc['count']}px inside solid body region",
                    "bbox": hc["bbox"],
                    "affected_pixels": [list(p) for p in hc["pixels"]],
                    "suggested_fix": f"PIXEL {hx} {hy} {fill_color_idx}" if hc["count"] == 1 else f"RECT {hc['bbox']['min_x']} {hc['bbox']['min_y']} {hc['bbox']['max_x'] - hc['bbox']['min_x'] + 1} {hc['bbox']['max_y'] - hc['bbox']['min_y'] + 1} {fill_color_idx} 1",
                })
                defect_counter += 1

        # 3. Disconnected Body Parts (Humanoid & Non-Humanoid)
        parts_dict = render_res.canvas.parts
        torso_part = parts_dict.get("torso")

        if torso_part and torso_part.bounding_box:
            tb = torso_part.bounding_box
            for pname, part in parts_dict.items():
                if pname != "torso" and part.bounding_box:
                    pb = part.bounding_box
                    dx = max(0, tb["min_x"] - pb["max_x"], pb["min_x"] - tb["max_x"])
                    dy = max(0, tb["min_y"] - pb["max_y"], pb["min_y"] - tb["max_y"])
                    gap = max(dx, dy)

                    # Category-specific gap tolerances
                    max_allowed_gap = 2
                    if category == "slime":
                        max_allowed_gap = 4
                    elif category == "wizard":
                        max_allowed_gap = 3  # floating hat or staff

                    if gap > max_allowed_gap:
                        # Compute directional correction translation
                        shift_x = -(pb["min_x"] - tb["max_x"]) if pb["min_x"] > tb["max_x"] else (tb["min_x"] - pb["max_x"] if pb["max_x"] < tb["min_x"] else 0)
                        shift_y = -(pb["min_y"] - tb["max_y"]) if pb["min_y"] > tb["max_y"] else (tb["min_y"] - pb["max_y"] if pb["max_y"] < tb["min_y"] else 0)
                        # Bound shift to close the gap
                        shift_x = max(-4, min(4, shift_x))
                        shift_y = max(-4, min(4, shift_y))

                        defects.append({
                            "id": f"defect_{defect_counter}",
                            "type": "disconnected_part",
                            "category": category,
                            "severity": "critical" if gap > 5 else "moderate",
                            "confidence": 0.94,
                            "description": f"Anatomical part '{pname}' is disconnected from torso by a {gap}px spatial gap",
                            "bbox": pb,
                            "affected_pixels": [],
                            "suggested_fix": f"TRANSFORM {pname} {shift_x} {shift_y} 0",
                        })
                        defect_counter += 1

        # 4. Palette Inconsistencies
        violations = check_palette_violations(comp, palette_name)
        if violations:
            # Group violations
            v_coords = [[v["x"], v["y"]] for v in violations]
            sample_v = violations[0]
            defects.append({
                "id": f"defect_{defect_counter}",
                "type": "palette_leak",
                "category": category,
                "severity": "minor" if len(violations) <= 5 else "moderate",
                "confidence": 0.99,
                "description": f"Detected {len(violations)} pixels with out-of-palette RGB values in '{palette_name}'",
                "bbox": {
                    "min_x": min(v["x"] for v in violations),
                    "min_y": min(v["y"] for v in violations),
                    "max_x": max(v["x"] for v in violations),
                    "max_y": max(v["y"] for v in violations),
                },
                "affected_pixels": v_coords[:20],
                "suggested_fix": f"PIXEL {sample_v['x']} {sample_v['y']} {sample_v['closest_palette_idx']}",
            })
            defect_counter += 1

        # 5. Missing Limb / Severe Asymmetry (for humanoid characters)
        if category in HUMANOID_CATEGORIES:
            has_arm_left = "arm_left" in parts_dict and parts_dict["arm_left"].bounding_box is not None
            has_arm_right = "arm_right" in parts_dict and parts_dict["arm_right"].bounding_box is not None
            has_shield_or_weapon = "shield" in parts_dict or "weapon" in parts_dict

            if has_arm_left and not has_arm_right and not has_shield_or_weapon:
                alb = parts_dict["arm_left"].bounding_box
                rx = (tb["max_x"] + 1) if torso_part and torso_part.bounding_box else 19
                rw = alb.get("width", 2)
                rh = alb.get("height", 4)
                defects.append({
                    "id": f"defect_{defect_counter}",
                    "type": "missing_limb",
                    "category": category,
                    "severity": "moderate",
                    "confidence": 0.88,
                    "description": "Character has left arm but missing right arm counterpart without weapon or shield",
                    "bbox": alb,
                    "affected_pixels": [],
                    "suggested_fix": f"PART arm_right torso {rx} {alb['min_y']}\nRECT {rx} {alb['min_y']} {rw} {rh} 10 1",
                })
                defect_counter += 1

        # 6. Proportion Check
        head_part = parts_dict.get("head")
        if torso_part and head_part and torso_part.bounding_box and head_part.bounding_box:
            tb = torso_part.bounding_box
            hb = head_part.bounding_box
            torso_h = max(1, tb["max_y"] - tb["min_y"] + 1)
            head_h = max(1, hb["max_y"] - hb["min_y"] + 1)

            # In standard proportions, head height should not exceed 2.2x torso height
            if head_h > torso_h * 2.2 and category not in ("slime", "chibi"):
                defects.append({
                    "id": f"defect_{defect_counter}",
                    "type": "incorrect_proportions",
                    "category": category,
                    "severity": "minor",
                    "confidence": 0.82,
                    "description": f"Head height ({head_h}px) is disproportionately large compared to torso ({torso_h}px)",
                    "bbox": hb,
                    "affected_pixels": [],
                    "suggested_fix": "TRANSFORM head 0 1 0",
                })
                defect_counter += 1

        # Calculate Overall Visual Quality Score
        penalty = 0
        for d in defects:
            if d["severity"] == "critical":
                penalty += 30
            elif d["severity"] == "moderate":
                penalty += 15
            else:
                penalty += 8

        if total_pixels < 25:
            penalty += 35
        elif total_pixels < 60:
            penalty += 15

        quality_score = max(10, min(99, 98 - penalty))

        # Anatomical decomposition summary
        anatomy_summary = {
            "total_components": len(components),
            "main_component_pixels": main_comp["count"] if main_comp else 0,
            "bounding_box": main_bbox,
            "parts_detected": list(parts_dict.keys()),
            "category_profile": NON_HUMANOID_CATEGORIES.get(category, "humanoid"),
        }

        return {
            "quality_score": quality_score,
            "pixel_fill_count": total_pixels,
            "detected_defects": defects,
            "has_defects": len(defects) > 0,
            "defects_count": len(defects),
            "anatomy": anatomy_summary,
            "validation_errors": render_res.validation.errors,
            "validation_warnings": render_res.validation.warnings,
        }
