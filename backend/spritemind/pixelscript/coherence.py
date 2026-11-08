"""Keep animated frames anatomically connected.

The motion profiles (and the temporal model) move parts independently. Rounding
those part offsets to the pixel grid can tear a limb off its parent, split a
sprite into several floating islands, or shave pixels off a limb that rotated.
This module takes the proposed per-part offsets for one frame and returns the
closest set that keeps every part attached to its parent and intact.
"""
import math
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from spritemind.pixelscript.ast import ActionType, PixelAction
from spritemind.pixelscript.parser import PixelScriptParser
from spritemind.pixelscript.serializer import PixelScriptSerializer
from spritemind.renderer.engine import PixelScriptEngine

Offset = Tuple[int, int, int]

# A part may drift this many grid steps (Chebyshev) away from its parent and
# still read as connected; one step covers a diagonal shoulder or hip.
MAX_ATTACH_DISTANCE = 1
# Rotating a part narrower than this shatters it into loose pixels.
MIN_ROTATE_EXTENT = 4
# A part that keeps less than this share of its pixels has been destroyed.
MIN_PIXEL_RETENTION = 0.8


def _clean_script(base_script: str) -> List[str]:
    return [ln.strip() for ln in base_script.splitlines() if ln.strip() and not ln.strip().startswith("TRANSFORM")]


def _script_with_offsets(base_script: str, offsets: Dict[str, Offset]) -> str:
    """Injects transforms before the finishing pass so shading and outline follow the moved parts."""
    lines = _clean_script(base_script)
    transforms = [f"TRANSFORM {part} {dx} {dy} {rot}" for part, (dx, dy, rot) in offsets.items() if dx or dy or rot]
    if not transforms:
        return "\n".join(lines)
    cut = len(lines)
    for index, line in enumerate(lines):
        if line.startswith(("SHADE ", "OUTLINE ")):
            cut = index
            break
    return "\n".join(lines[:cut] + transforms + lines[cut:])


def _rasterize(
    base_script: str, offsets: Dict[str, Offset], width: int, height: int, palette_name: str
):
    result = PixelScriptEngine.execute(
        _script_with_offsets(base_script, offsets),
        width=width,
        height=height,
        palette_name=palette_name,
    )
    pixels = {name: set(part.pixels) for name, part in result.canvas.parts.items()}
    parents = {name: part.parent for name, part in result.canvas.parts.items()}
    return pixels, parents


def _distance(a: Set[Tuple[int, int]], b: Set[Tuple[int, int]]) -> int:
    """Smallest Manhattan gap between two pixel sets (0 or 1 means touching)."""
    if not a or not b:
        return 10**6
    if len(a) * len(b) <= 4096:
        best = 10**6
        for (ay, ax) in a:
            for (by, bx) in b:
                d = abs(ay - by) + abs(ax - bx)
                if d < best:
                    best = d
                    if best <= MAX_ATTACH_DISTANCE:
                        return best
        return best
    # Cheap path for big parts: compare bounding boxes.
    ay0, ax0 = min(p[0] for p in a), min(p[1] for p in a)
    ay1, ax1 = max(p[0] for p in a), max(p[1] for p in a)
    by0, bx0 = min(p[0] for p in b), min(p[1] for p in b)
    by1, bx1 = max(p[0] for p in b), max(p[1] for p in b)
    dy = max(0, by0 - ay1 - 1, ay0 - by1 - 1)
    dx = max(0, bx0 - ax1 - 1, ax0 - bx1 - 1)
    return dy + dx


def _components(pixels: Iterable[Tuple[int, int]]) -> int:
    remaining = set(pixels)
    count = 0
    while remaining:
        count += 1
        stack = [remaining.pop()]
        while stack:
            y, x = stack.pop()
            for neighbour in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                if neighbour in remaining:
                    remaining.remove(neighbour)
                    stack.append(neighbour)
    return count


def _extent(pixels: Set[Tuple[int, int]]) -> int:
    if not pixels:
        return 0
    ys = [p[0] for p in pixels]
    xs = [p[1] for p in pixels]
    return min(max(ys) - min(ys), max(xs) - min(xs)) + 1


def _candidates(offset: Offset, pixels: Optional[Set[Tuple[int, int]]], part: str) -> List[Offset]:
    dx, dy, rot = offset
    # A thin limb rotated by a few degrees breaks into a staircase that loses
    # its corners, so it swings without rotating.
    thin = pixels is not None and _extent(pixels) < MIN_ROTATE_EXTENT
    options: List[Offset] = [(dx, dy, 0)] if thin else [(dx, dy, rot), (dx, dy, rot // 2)]
    if part.startswith("leg") and abs(dx) > 1:
        options.append(((1 if dx > 0 else -1), dy, 0))
    options.append(((1 if dx > 0 else -1 if dx < 0 else 0), dy, 0))
    options.append((0, (1 if dy > 0 else -1 if dy < 0 else 0), 0))
    options.append((0, 0, 0))
    seen: List[Offset] = []
    for candidate in options:
        if candidate not in seen:
            seen.append(candidate)
    return seen


def _parent_order(parents: Dict[str, Optional[str]]) -> List[str]:
    ordered: List[str] = []
    pending = dict(parents)
    while pending:
        progressed = False
        for name, parent in list(pending.items()):
            if parent is None or parent in ordered or parent not in pending:
                ordered.append(name)
                del pending[name]
                progressed = True
        if not progressed:  # cycle or unknown parent: keep the rest in place
            ordered.extend(pending)
            break
    return ordered


def _frame_stats(
    base_script: str, offsets: Dict[str, Offset], width: int, height: int, palette_name: str
) -> Tuple[int, int]:
    """Islands and opaque pixel count of what is actually drawn.

    Parts keep pixels that a later part painted over, so the union of part pixel
    sets can look connected while the rendered silhouette is already torn.
    """
    result = PixelScriptEngine.execute(
        _script_with_offsets(base_script, offsets),
        width=width,
        height=height,
        palette_name=palette_name,
    )
    alpha = result.composite_array[..., 3] > 0
    rows, cols = alpha.nonzero()
    return _components(zip(rows.tolist(), cols.tolist())), int(len(rows))


def _resolve(
    base_script: str,
    offsets: Dict[str, Offset],
    width: int,
    height: int,
    palette_name: str,
    base_pixels: Dict[str, Set[Tuple[int, int]]],
    parents: Dict[str, Optional[str]],
) -> Dict[str, Offset]:
    base_counts = {name: len(px) for name, px in base_pixels.items()}
    base_components = {name: _components(px) for name, px in base_pixels.items()}
    resolved: Dict[str, Offset] = dict(offsets)

    def usable(name: str, pixels: Dict[str, Set[Tuple[int, int]]]) -> bool:
        own = pixels.get(name, set())
        before = base_counts.get(name, 0)
        if before and len(own) < MIN_PIXEL_RETENTION * before:
            return False
        # A part may not gain new islands: rotation used to split a 2px arm into
        # four floating specks that still passed a loose component budget.
        return _components(own) <= base_components.get(name, 1)

    def attached(name: str, pixels: Dict[str, Set[Tuple[int, int]]]) -> bool:
        parent = parents.get(name)
        if parent is None or parent not in pixels:
            return True
        return _distance(pixels.get(name, set()), pixels[parent]) <= MAX_ATTACH_DISTANCE

    for _ in range(2):
        changed = False
        pixels, _ = _rasterize(base_script, resolved, width, height, palette_name)
        for name in _parent_order(parents):
            current = resolved.get(name, (0, 0, 0))
            if not any(current):
                continue
            if usable(name, pixels) and attached(name, pixels):
                continue
            for candidate in _candidates(current, base_pixels.get(name), name):
                if candidate == current:
                    continue
                trial = dict(resolved)
                trial[name] = candidate
                trial_pixels, _ = _rasterize(base_script, trial, width, height, palette_name)
                if usable(name, trial_pixels) and attached(name, trial_pixels):
                    resolved = trial
                    pixels = trial_pixels
                    changed = True
                    break
        if not changed:
            break
    return resolved


def _attenuate(offsets: Dict[str, Offset], scale: float) -> Dict[str, Offset]:
    return {
        name: (
            int(round(dx * scale)),
            int(round(dy * scale)),
            int(round(rot * scale)),
        )
        for name, (dx, dy, rot) in offsets.items()
    }


def stabilize_offsets(
    base_script: str,
    offsets: Dict[str, Offset],
    width: int = 32,
    height: int = 32,
    palette_name: str = "pico8",
) -> Dict[str, Offset]:
    """Returns offsets that keep every part attached to its parent and whole."""
    if not any(any(v) for v in offsets.values()):
        return dict(offsets)

    from spritemind.pixelscript.rig import apply_limb_flex

    offsets = apply_limb_flex(offsets)
    base_pixels, parents = _rasterize(base_script, {}, width, height, palette_name)
    base_islands, _ = _frame_stats(base_script, {}, width, height, palette_name)
    best: Optional[Dict[str, Offset]] = None
    best_islands = 10**6
    best_count = -1

    # Moving pixels around cannot add islands that the static sprite did not
    # already have. If a pose still fragments the silhouette, soften the pose
    # rather than shipping a torn frame.
    for scale in (1.0, 0.5, 0.0):
        resolved = _resolve(
            base_script, _attenuate(offsets, scale), width, height, palette_name, base_pixels, parents
        )
        islands, count = _frame_stats(base_script, resolved, width, height, palette_name)
        if islands < best_islands or (islands == best_islands and count > best_count):
            best, best_islands, best_count = resolved, islands, count
        if islands <= base_islands:
            return resolved

    return best if best is not None else dict(offsets)



@lru_cache(maxsize=8)
def _geometry_for(base_script: str, width: int, height: int, palette_name: str):
    from spritemind.pixelscript.rig import RigGeometry

    return RigGeometry.from_script(base_script, width, height, palette_name)


def _snap_limb_chains(
    base_script: str,
    offsets: Dict[str, Offset],
    width: int = 32,
    height: int = 32,
    palette_name: str = "pico8",
) -> Dict[str, Offset]:
    """Closes elbows and knees after the per-part stabilizer has run."""
    if not offsets:
        return dict(offsets)
    return _geometry_for(base_script, width, height, palette_name).snap_limb_chains(offsets)


MAX_JOINT_GAP = 2 ** 0.5 + 1e-6
# A bent joint may move a handful of pixels; it may not restyle the whole frame.
BEND_PIXEL_BUDGET = 6


def _closest_offset(a: Set[Tuple[int, int]], b: Set[Tuple[int, int]]):
    """Vector from the nearest pixel of ``a`` to the nearest pixel of ``b``."""
    if not a or not b:
        return None
    best = None
    best_d = float("inf")
    for ax, ay in a:
        for bx, by in b:
            d = (ax - bx) ** 2 + (ay - by) ** 2
            if d < best_d:
                best_d = d
                best = (bx - ax, by - ay)
    return best, best_d ** 0.5


def _close_joint_gaps(
    base_script: str,
    offsets: Dict[str, Offset],
    width: int = 32,
    height: int = 32,
    palette_name: str = "pico8",
) -> Dict[str, Offset]:
    """Pulls a bent elbow or knee until the rendered pixels actually touch.

    Rotating a two-pixel-wide forearm drags its contact row off the parent grid,
    so a purely geometric pin can still leave a one-pixel hole. Measuring the
    rendered masks and nudging the child is the only reliable check.
    """
    from spritemind.pixelscript.rig import LIMB_CHAINS

    out = dict(offsets)
    chains = set(LIMB_CHAINS)
    for _ in range(3):
        result = PixelScriptEngine.execute(
            _script_with_offsets(base_script, out), width=width, height=height, palette_name=palette_name
        )
        parts = result.canvas.parts
        moved = False
        for child in chains:
            if child not in parts:
                continue
            parent = parts[child].parent
            if not parent or parent not in parts:
                continue
            nearest = _closest_offset(parts[child].pixels, parts[parent].pixels)
            if nearest is None:
                continue
            delta, gap = nearest
            if gap <= MAX_JOINT_GAP:
                continue
            dx, dy, rot = out.get(child, (0, 0, 0))
            out[child] = (dx + delta[0], dy + delta[1], rot)
            moved = True
        if not moved:
            break
    return out


def build_frame_script(
    base_script: str,
    offsets: Dict[str, Offset],
    width: int = 32,
    height: int = 32,
    palette_name: str = "pico8",
) -> str:
    """Stabilize the offsets for one frame and return the frame script.

    A second segment must follow its upper segment even when it does not bend,
    otherwise the limb breaks apart as soon as the pose rotates. The elbow and
    knee lag is laid over that attached pose and kept only when it costs no
    silhouette integrity, so a bent joint never tears a held-together frame.
    """
    from spritemind.pixelscript.rig import apply_limb_flex

    plain = stabilize_offsets(base_script, offsets, width=width, height=height, palette_name=palette_name)
    zero = _snap_limb_chains(base_script, plain, width, height, palette_name)
    lagged = apply_limb_flex(zero)
    if lagged == zero:
        return _script_with_offsets(base_script, zero)
    bent = _close_joint_gaps(
        base_script,
        _snap_limb_chains(base_script, lagged, width, height, palette_name),
        width,
        height,
        palette_name,
    )
    zero_islands, zero_count = _frame_stats(base_script, zero, width, height, palette_name)
    bent_islands, bent_count = _frame_stats(base_script, bent, width, height, palette_name)
    if bent_islands > zero_islands or abs(bent_count - zero_count) > BEND_PIXEL_BUDGET:
        return _script_with_offsets(base_script, zero)
    return _script_with_offsets(base_script, bent)
