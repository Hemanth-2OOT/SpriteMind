"""Export functions for PNG, GIF, and Spritesheet formats with metadata."""
import io
import base64
import math
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Tuple, Optional


def array_to_image(arr: np.ndarray, scale: int = 1) -> Image.Image:
    """Converts (H, W, 4) uint8 RGBA array to PIL Image with nearest-neighbor scaling."""
    img = Image.fromarray(arr, mode="RGBA")
    if scale > 1:
        img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    return img


def array_to_png_bytes(arr: np.ndarray, scale: int = 1) -> bytes:
    img = array_to_image(arr, scale)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def array_to_base64(arr: np.ndarray, scale: int = 1) -> str:
    png_bytes = array_to_png_bytes(arr, scale)
    b64 = base64.b64encode(png_bytes).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def export_gif(frames: List[np.ndarray], fps: int = 8, scale: int = 4, loop: int = 0) -> bytes:
    """Exports a list of RGBA frames as an animated GIF with transparency."""
    if not frames:
        raise ValueError("No frames provided for GIF export")

    pil_frames = []
    duration = int(1000 / max(1, fps))

    for frame in frames:
        img = array_to_image(frame, scale)
        pil_frames.append(img)

    buf = io.BytesIO()
    pil_frames[0].save(
        buf,
        format="GIF",
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration,
        loop=loop,
        disposal=2,
    )
    return buf.getvalue()


def export_spritesheet(
    frames: List[np.ndarray],
    cols: Optional[int] = None,
    scale: int = 1,
    padding: int = 0,
) -> Tuple[bytes, Dict[str, Any]]:
    """Tiles frames into a grid spritesheet PNG and returns JSON metadata."""
    if not frames:
        raise ValueError("No frames provided for spritesheet export")

    n_frames = len(frames)
    frame_h, frame_w, _ = frames[0].shape

    if cols is None or cols <= 0:
        cols = n_frames if n_frames <= 8 else math.ceil(math.sqrt(n_frames))
    rows = math.ceil(n_frames / cols)

    sheet_w = (cols * frame_w + (cols - 1) * padding) * scale
    sheet_h = (rows * frame_h + (rows - 1) * padding) * scale

    sheet = Image.new("RGBA", (sheet_w, sheet_h), (0, 0, 0, 0))
    frame_metadata: List[Dict[str, Any]] = []

    for idx, frame in enumerate(frames):
        row = idx // cols
        col = idx % cols
        x = col * (frame_w + padding) * scale
        y = row * (frame_h + padding) * scale

        frame_img = array_to_image(frame, scale)
        sheet.paste(frame_img, (x, y))

        frame_metadata.append({
            "frame_index": idx,
            "x": x,
            "y": y,
            "width": frame_w * scale,
            "height": frame_h * scale,
            "col": col,
            "row": row,
        })

    buf = io.BytesIO()
    sheet.save(buf, format="PNG")

    meta = {
        "sheet_width": sheet_w,
        "sheet_height": sheet_h,
        "frame_width": frame_w * scale,
        "frame_height": frame_h * scale,
        "columns": cols,
        "rows": rows,
        "total_frames": n_frames,
        "scale": scale,
        "frames": frame_metadata,
    }

    return buf.getvalue(), meta
