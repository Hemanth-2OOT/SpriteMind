"""Inter-frame continuity metric used by the coherence report."""
import numpy as np

from spritemind.renderer.continuity import continuity_report, opaque_mask, pair_delta


def _frame(pixels):
    rgba = np.zeros((8, 8, 4), dtype=np.uint8)
    for y, x in pixels:
        rgba[y, x] = (255, 255, 255, 255)
    return rgba


def _square(offset=0):
    return _frame([(y, x) for y in range(1, 5) for x in range(1 + offset, 5 + offset)])


def test_identical_frames_have_perfect_iou():
    delta = pair_delta(opaque_mask(_square()), opaque_mask(_square()))
    assert delta["iou"] == 1.0
    assert delta["lost_pixels"] == 0
    assert delta["changed_share"] == 0.0


def test_a_shifted_frame_reports_lost_and_gained_pixels():
    delta = pair_delta(opaque_mask(_square()), opaque_mask(_square(offset=1)))
    assert delta["lost_pixels"] > 0
    assert delta["gained_pixels"] > 0
    assert 0 < delta["iou"] < 1
    assert delta["lost_pixels"] == delta["gained_pixels"]


def test_single_frame_animation_reports_no_pairs():
    report = continuity_report([opaque_mask(_square())])
    assert report["pairs"] == 0
    assert report["mean_iou"] is None
    assert report["worst_pair"] is None


def test_a_deliberately_broken_frame_ranks_worst():
    frames = [opaque_mask(_square()), opaque_mask(_square(offset=1)), opaque_mask(_frame([(0, 7)]))]
    report = continuity_report(frames)
    assert report["pairs"] == 2
    assert report["worst_iou"] == report["per_pair"][1]["iou"]
    assert report["worst_pair"] == {"from_frame": 1, "to_frame": 2}
    assert report["per_pair"][1]["lost_pixels"] > report["per_pair"][0]["lost_pixels"]
    assert report["worst_iou"] < report["mean_iou"]


def test_masks_from_frames_decodes_base64_payloads():
    import base64
    import io
    from PIL import Image

    from spritemind.renderer.continuity import masks_from_frames

    buffer = io.BytesIO()
    Image.fromarray(_square()).save(buffer, format="PNG")
    payload = {"image_base64": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()}

    def decode(frame):
        raw = frame["image_base64"].split(",", 1)[-1]
        return np.array(Image.open(io.BytesIO(base64.b64decode(raw))).convert("RGBA"))

    masks = masks_from_frames([payload, payload], decode)
    assert len(masks) == 2
    assert masks[0].sum() == 16
    assert (masks[0] == masks[1]).all()
