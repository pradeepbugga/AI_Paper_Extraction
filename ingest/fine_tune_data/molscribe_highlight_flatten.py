"""Fix for a real MolScribe decoder bug: a benzene ring whose interior is
covered by a solid pastel color-fill highlight causes the decoder to
hallucinate a phantom atom near the ring's geometric center, wired into
the graph with extra bonds (confirmed via raw-atom-coordinate inspection
and by isolating the color fill as the only variable on an otherwise-
identical real image -- same font, resolution, compression, bond angles,
decodes correctly once only the fill is removed).

Two earlier approaches to detecting "is a fill present" failed:

1. A single global lightness threshold, applied unconditionally, caused
   a real regression: it thinned an already-fragile anti-aliased glyph
   on a small, naturally blue-inked image, breaking a token it had
   gotten right before.
2. A per-paper color-profiling approach (find colors that recur across
   many of a paper's own images) also failed once tested on more papers:
   different papers use color for entirely different, both-legitimate
   reasons that produce the same surface statistics -- e.g.
   `suzuki_iron_2024` washes its WHOLE canvas background in pale green
   and draws different ring fragments in different colored ink (a
   deliberate "which starting material contributed this ring"
   convention), which is real information, not a bug-triggering fill.
   Per-paper color recurrence can't tell these apart.

The actual universal invariant (identified directly from the data, not
assumed): **structural ink is always FAR darker than its own image's
background, regardless of what color that background or ink actually
is; a highlight fill is only SLIGHTLY darker than the background** (a
translucent wash you're meant to still read the linework through).
Measuring this per image, with no cross-image calibration and no
hardcoded absolute color values at all:

  image                          bg lightness   other clusters (gap from bg)
  page100 (real fill present)    248            229 (gap 19), 224 (gap 24)
  page5   (blue ink, no fill)    248            85  (gap 163), 109 (gap 139)
  suzuki  (green bg+ink, no fill) 224            56  (gap 168), 37 (gap 187), 0 (gap 224)

There is a >100-unit margin between the largest observed fill gap (~24)
and the smallest observed ink gap (~139) -- fills and ink occupy
completely separate regions of "distance from this image's own
background," regardless of anything else about the image. This needs no
per-paper profiling: the background is simply each image's own dominant
color cluster, and everything else is classified purely by how far it
sits from that, in that same image.
"""
import cv2
import numpy as np

QUANTIZE = 8
BG_MIN_FRACTION = 0.5   # the background must be the clearly dominant cluster
FILL_MIN_FRACTION = 0.01  # a fill candidate must cover a meaningful area
FILL_MAX_GAP = 70       # lightness gap from background below which a cluster is a
                        # translucent fill, not real ink (observed: fills ~19-24, ink ~139+)
FILL_MIN_SPREAD = 15    # max(channel)-min(channel): a deliberate highlight has a real hue
                        # tint; a neutral gray (spread=0, R=G=B) close to background is
                        # almost always anti-aliasing halo around real ink/text, not a fill
                        # -- confirmed as the cause of a real regression (full-corpus check,
                        # Aug 22 follow-up): several exotic-abbreviation labels ([Sc], [Rf],
                        # [Ca]-style placeholders) have soft gray anti-aliasing that this gap
                        # check alone misclassified as fill, erasing part of the label itself.
MATCH_TOLERANCE = 10    # per-channel distance for matching a pixel to a detected fill color


def detect_fill_colors(img_bgr):
    """Per-image, no cross-image calibration. Returns (background_color,
    [fill_colors]) -- fill_colors is empty if this image has no fill."""
    arr = img_bgr.astype(int)
    total = arr.shape[0] * arr.shape[1]
    q = (arr // QUANTIZE) * QUANTIZE
    colors, counts = np.unique(q.reshape(-1, 3), axis=0, return_counts=True)
    order = np.argsort(-counts)
    colors, counts = colors[order], counts[order]

    bg_frac = counts[0] / total
    if bg_frac < BG_MIN_FRACTION:
        return None, []  # no clear dominant background; don't guess
    bg_color = colors[0]
    bg_lightness = bg_color.mean()

    fills = []
    for color, count in zip(colors[1:], counts[1:]):
        frac = count / total
        if frac < FILL_MIN_FRACTION:
            continue
        gap = bg_lightness - color.mean()
        spread = int(color.max()) - int(color.min())
        if 0 < gap < FILL_MAX_GAP and spread >= FILL_MIN_SPREAD:
            fills.append(tuple(int(c) for c in color))
    return tuple(int(c) for c in bg_color), fills


def flatten_highlights(img_bgr, bg_color, fill_colors):
    """Map pixels matching a detected fill color back to the background
    color; leave everything else (real ink, any hue, anti-aliasing)
    untouched."""
    arr = img_bgr.astype(int)
    mask = np.zeros(arr.shape[:2], dtype=bool)
    for color in fill_colors:
        dist = np.abs(arr - np.array(color)).max(axis=2)
        mask |= (dist <= MATCH_TOLERANCE)
    if not mask.any():
        return img_bgr, False
    out = img_bgr.copy()
    out[mask] = bg_color
    return out, True


def preprocess(img_bgr):
    """Gated, per-image, no calibration required. Returns
    (processed_img_bgr, was_modified)."""
    bg_color, fill_colors = detect_fill_colors(img_bgr)
    if not fill_colors:
        return img_bgr, False
    return flatten_highlights(img_bgr, bg_color, fill_colors)


if __name__ == "__main__":
    import sys
    for path in sys.argv[1:]:
        img = cv2.imread(path)
        out, modified = preprocess(img)
        print(f"{path}: modified={modified}")
        if modified:
            out_path = path.rsplit(".", 1)[0] + "_flattened.png"
            cv2.imwrite(out_path, out)
            print(f"  wrote {out_path}")
