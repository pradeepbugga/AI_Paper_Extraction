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
MATCH_TOLERANCE = 20    # per-channel distance for matching a pixel to a detected fill color.
                        # 10 was too tight: left a residual anti-aliased "ghost" of the fill at
                        # its own edges (e.g. (187,235,235), distance 15 from a detected
                        # (199,250,250) fill) that the flood of the interior didn't reach --
                        # confirmed as the actual cause of 3 of the 4 unexplained regressions
                        # (page17_fig1_seg0 et al.): not a case where removing the fill made
                        # things worse, but a case where the fill was only partially removed.
MINORITY_INK_MAX_RATIO = 0.2  # an ink cluster whose pixel count is under this fraction of the
                              # dominant ink cluster's count is a minority highlight color (e.g.
                              # one bond bolded red while the rest of the structure is black),
                              # not this image's real overall ink color -- harmonize it to match
                              # the majority ink instead of leaving it as an inconsistent outlier.


def detect_fill_colors(img_bgr):
    """Per-image, no cross-image calibration. Returns (fill_colors,
    minority_ink_colors, majority_ink_color, bg_lightness) -- fill_colors
    is empty if this image has no fill; minority_ink_colors is empty if
    all of this image's ink is one consistent color; bg_lightness is None
    if no clear dominant background was found."""
    arr = img_bgr.astype(int)
    total = arr.shape[0] * arr.shape[1]
    q = (arr // QUANTIZE) * QUANTIZE
    colors, counts = np.unique(q.reshape(-1, 3), axis=0, return_counts=True)
    order = np.argsort(-counts)
    colors, counts = colors[order], counts[order]

    bg_frac = counts[0] / total
    if bg_frac < BG_MIN_FRACTION:
        return [], [], None, None  # no clear dominant background; don't guess
    bg_lightness = float(colors[0].mean())

    fills = []
    ink_clusters = []  # (color, count) for everything that isn't background or fill
    for color, count in zip(colors[1:], counts[1:]):
        frac = count / total
        if frac < FILL_MIN_FRACTION:
            continue
        gap = bg_lightness - color.mean()
        spread = int(color.max()) - int(color.min())
        if 0 < gap < FILL_MAX_GAP and spread >= FILL_MIN_SPREAD:
            fills.append(tuple(int(c) for c in color))
        elif gap >= FILL_MAX_GAP:
            ink_clusters.append((tuple(int(c) for c in color), count))

    if not ink_clusters:
        return fills, [], None, bg_lightness
    ink_clusters.sort(key=lambda x: -x[1])
    majority_ink_color, majority_count = ink_clusters[0]
    minority_inks = [c for c, n in ink_clusters[1:] if n / majority_count < MINORITY_INK_MAX_RATIO]
    return fills, minority_inks, majority_ink_color, bg_lightness


DILATE_KERNEL = np.ones((3, 3), np.uint8)
DILATE_ITERATIONS = 2   # sweeps up the anti-aliased boundary ring around a fill region
                        # that per-pixel color matching alone keeps missing regardless of
                        # tolerance -- confirmed directly: even at MATCH_TOLERANCE=20 a faint
                        # ghost of the fill remained visible right at the ring's double-bond
                        # edges (a color like (187,235,235) shading into the black ink).
                        #
                        # DANGER, confirmed directly by rendering the node/edge overlay (Aug 22
                        # follow-up): blind dilation can grow far enough to erase real, thin
                        # bond-line ink when a fill sits flush against it with little clearance
                        # (page61_fig2_seg0.png -- both entire ring skeletons vanished, and the
                        # "fixed" RDKit-valid SMILES this had previously produced turned out to
                        # be coincidental garbage, not a real structure -- a measurement error in
                        # the validity check, not a real success). Every write to `out` below
                        # MUST go through a protect mask that never overwrites a genuinely dark
                        # (real-ink) pixel, however the candidate mask was grown.


WHITE = (255, 255, 255)


def _color_mask(arr, colors, tolerance):
    mask = np.zeros(arr.shape[:2], dtype=bool)
    for color in colors:
        dist = np.abs(arr - np.array(color)).max(axis=2)
        mask |= (dist <= tolerance)
    return mask


def flatten_highlights(img_bgr, fill_colors, minority_inks, majority_ink_color, bg_lightness):
    """Map fill pixels (plus a small dilation to catch their anti-aliased
    boundary) to pure white; harmonize any minority-colored highlight ink
    (e.g. one bond bolded in a second color) to match the image's own
    majority ink color. Leaves the majority ink itself -- any hue -- and
    all background untouched. A dilated candidate mask can geometrically
    overlap real ink; `protect` (same gap rule used to classify ink
    clusters, applied per-pixel) guarantees no genuinely dark pixel is
    ever overwritten, regardless of how the candidate mask was grown."""
    arr = img_bgr.astype(int)
    out = img_bgr.copy()
    modified = False

    pixel_lightness = arr.mean(axis=2)
    protect = (bg_lightness - pixel_lightness) >= FILL_MAX_GAP

    if fill_colors:
        fill_mask = _color_mask(arr, fill_colors, MATCH_TOLERANCE)
        if fill_mask.any():
            fill_mask = cv2.dilate(fill_mask.astype(np.uint8), DILATE_KERNEL,
                                    iterations=DILATE_ITERATIONS).astype(bool)
            fill_mask &= ~protect
            out[fill_mask] = WHITE
            modified = True

    if minority_inks:
        ink_mask = _color_mask(arr, minority_inks, MATCH_TOLERANCE)
        if ink_mask.any():
            ink_mask = cv2.dilate(ink_mask.astype(np.uint8), DILATE_KERNEL,
                                   iterations=DILATE_ITERATIONS).astype(bool)
            out[ink_mask] = majority_ink_color
            modified = True

    return out, modified


def preprocess(img_bgr):
    """Gated, per-image, no calibration required. Returns
    (processed_img_bgr, was_modified)."""
    fill_colors, minority_inks, majority_ink_color, bg_lightness = detect_fill_colors(img_bgr)
    if not fill_colors and not minority_inks:
        return img_bgr, False
    return flatten_highlights(img_bgr, fill_colors, minority_inks, majority_ink_color, bg_lightness)


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
