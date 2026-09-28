"""Sub-crops a figure image down to just its chemical structure region,
dropping an attached NMR/spectrum plot that Stage 1 captured in the same
bounding box.

Found via manual review of DECIMER's low-confidence outputs: the majority
of flagged figures aren't cases where DECIMER struggled with a real
structure -- they're composite crops where a small structure sits above or
beside a much larger, mostly-blank spectrum plot, and DECIMER's decoder
degenerates into repeated tokens on that unfamiliar image shape. The fix is
geometric, not semantic: a real structure's ink is compact and localized in
both dimensions, while everything else in these composites is not:
a spectrum trace or axis line spans nearly the full image width, and
peak-label leader lines (thin vertical strands connecting a rotated
chemical-shift label at the top of the plot down to its peak) span nearly
the full image height despite being only a few pixels wide. Filtering
connected components on width alone isn't enough -- a row of individually
narrow tick-label blobs can still union back into a full-width span, and a
single narrow-but-tall leader line survives a width-only filter untouched.
Excluding components that are too wide OR too tall, and keeping only
components that are compact in both dimensions, handles both for the
common case: a clean gap between the structure and the spectrum below it,
with no other compact content on the page. The final crop is the union of
whatever compact regions survive, padded slightly.

Several things were tried and dropped after testing against real examples,
not just theorized about -- filtering by stroke width/thickness (absolute
or relative to the page's own dominant stroke width), and collapsing to a
single "most likely structure" region when multiple compact candidates
survive. Both either failed to fix the leader-line case or fixed it while
silently destroying real catalytic-cycle and multi-compound scope figures
by discarding several of their real structure fragments -- worse than the
original uncropped image, since those should stay intact for DECIMER to
(correctly) fail on, not get mangled into one wrong fragment. Plain OCR
text detection (exclude any compact candidate where an OCR pass finds
recognizable words) has the same problem: a scope figure's per-compound
caption sits close enough to its structure that dilation merges them into
one component, and OCR correctly finding the caption's text then wrongly
excludes the real structure attached to it.

What actually distinguishes a leader-line label block from both a real
structure and a real horizontal caption is that its text is rotated 90
degrees (stacked vertically, to fit several peak labels into a narrow
fan without overlapping). OCR confidence/word-count for a rotated label
block improves dramatically when the crop itself is rotated back to
upright, while a normal horizontal caption's score is already best in its
original orientation (rotating it only makes OCR worse), and a real
structure scores zero words at every orientation regardless. Checking
whether OCR does substantially better rotated than upright is therefore a
narrow, targeted way to exclude specifically the rotated-label case
without also catching -- and wrongly discarding -- horizontal captions
merged into real structures.

This only helps the composite-crop case. Multi-compound scope figures and
catalytic-cycle diagrams have structure-shaped ink spread across the whole
image with no spectrum to exclude -- for those this is a no-op (or a mild,
safe trim), and DECIMER's own confidence gate is still responsible for
flagging them.

A further residual case, found by measuring the corpus rather than
guessing: even in the clean-gap composite style, a small isolated label
can survive the filters above untouched -- a compound ID printed under
the structure ("1e"), an axis-unit caption ("ppm") far below it, or a
handful of tiny integration-tick fragments strung along the bottom edge
of the plot. None of these are wide, tall, or rotated, so nothing above
catches them, and their bounding boxes stretch the final union crop loose
even though the real structure itself was found correctly.

The same per-candidate OCR check already used for rotated labels turns
out to generalize here too, with one addition: only apply it to
non-dominant candidates, and only when one candidate already clearly
dominates the rest by area (DOMINANT_AREA_FRACTION) -- that split was
verified across the full corpus to reliably separate "one real structure
plus minor leftover debris" from "several real structure fragments
spread across a multi-compound diagram" (where no single candidate
dominates). Within that gate, a non-dominant candidate is excluded only
if OCR reads a token of length >= 2 containing a letter or digit --
requiring 2+ characters specifically preserves single-character
notational symbols ("+", "=") that a real reaction scheme can leave as
its own small isolated component, which OCR reads confidently as literal
text but which are not labels to discard.

A separate leftover shape has no text at all to detect: a plain
peak-integration tick mark, floating just above its peak line with a gap
wide enough that dilation doesn't merge it into the (correctly excluded)
spectrum trace. Neither OCR check can catch this since there's nothing to
read. What actually distinguishes it from real content is that it's
almost pure whitespace even within its own tight bounding box (measured
directly: a real tick's bounding box contained ~19 ink pixels versus
189-628 for genuine small structure fragments in a cycle diagram, an
order of magnitude apart) -- a tick is just a short stroke, while any
real structure fragment has actual 2D shape to it. Proximity to an
excluded component alone doesn't discriminate (a real reaction-scheme
symbol like "+" can sit just as close to an excluded component as a tick
sits to a spectrum), so this checks both together: only a non-dominant
candidate that is BOTH extremely sparse AND close to something already
excluded gets dropped. Verified directly against a real "+" symbol
example: it was closer to an excluded component than the tick was (8.6px
vs 16px), but had more than double the tick's ink count (40-44 vs 19),
so the sparsity requirement alone keeps it safe even though the
proximity requirement alone would not have.
"""

from pathlib import Path
import re

import numpy as np
import pytesseract
from PIL import Image
from scipy import ndimage

INK_THRESHOLD = 245
WIDE_FRACTION = 0.6  # component spanning this much of image width is a spectrum trace/axis, not a structure
TALL_FRACTION = 0.6  # component spanning this much of image height is a peak-label leader line, not a structure
MIN_REGION_AREA_FRACTION = 0.0005
PADDING_PX = 15
DILATION_ITERATIONS = 8
OCR_MIN_CONFIDENCE = 30
ROTATED_LABEL_SCORE_RATIO = 2.0  # rotated OCR score must exceed upright score by this much to call it a rotated label block
DOMINANT_AREA_FRACTION = 0.8  # a candidate this much of total kept area is treated as "the structure"; others are checked for label text
LABEL_TEXT_MIN_CONFIDENCE = 40
ALNUM_RE = re.compile(r"[A-Za-z0-9]")
TICK_MAX_INK_PIXELS = 30  # a real structure fragment has far more raw ink than a plain tick mark (measured: ticks ~19px, real fragments 189-628px)
TICK_MAX_DIST_TO_EXCLUDED = 20  # a tick sits just outside dilation's reach of the spectrum trace it belongs to


def _ocr_score(pil_image):
    data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)
    return sum(c for w, c in zip(data["text"], data["conf"]) if w.strip() and c > OCR_MIN_CONFIDENCE)


def _is_rotated_label_block(region_gray_crop):
    pil_crop = Image.fromarray(region_gray_crop)
    upright_score = _ocr_score(pil_crop)
    rotated_score = max(
        _ocr_score(pil_crop.rotate(90, expand=True)),
        _ocr_score(pil_crop.rotate(270, expand=True)),
    )
    return rotated_score > ROTATED_LABEL_SCORE_RATIO * max(upright_score, 1)


def _is_label_text(region_gray_crop, upscale=4):
    pil_crop = Image.fromarray(region_gray_crop)
    pil_crop = pil_crop.resize((pil_crop.width * upscale, pil_crop.height * upscale), Image.LANCZOS)
    for orientation in (pil_crop, pil_crop.rotate(90, expand=True), pil_crop.rotate(270, expand=True)):
        for psm in (6, 3, 7, 8):
            data = pytesseract.image_to_data(orientation, config=f"--psm {psm}", output_type=pytesseract.Output.DICT)
            for w, c in zip(data["text"], data["conf"]):
                w = w.strip()
                if w and c >= LABEL_TEXT_MIN_CONFIDENCE and len(w) >= 2 and ALNUM_RE.search(w):
                    return True
    return False


def crop_to_structure(image_path):
    """Returns (cropped_rgb_array, was_cropped)."""
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img)
    gray = np.array(img.convert("L"))
    height, width = gray.shape

    ink = gray < INK_THRESHOLD
    dilated = ndimage.binary_dilation(ink, iterations=DILATION_ITERATIONS)
    labeled, num_components = ndimage.label(dilated)

    if num_components == 0:
        return arr, False

    min_area = MIN_REGION_AREA_FRACTION * width * height
    compact_regions = []
    excluded_mask = np.zeros_like(ink)
    for comp_id, (slice_y, slice_x) in enumerate(ndimage.find_objects(labeled), start=1):
        comp_width = slice_x.stop - slice_x.start
        comp_height = slice_y.stop - slice_y.start
        if comp_width >= WIDE_FRACTION * width or comp_height >= TALL_FRACTION * height:
            excluded_mask |= labeled == comp_id
            continue
        if comp_width * comp_height < min_area:
            continue
        if _is_rotated_label_block(gray[slice_y, slice_x]):
            continue
        compact_regions.append((slice_x.start, slice_y.start, slice_x.stop, slice_y.stop))

    if not compact_regions:
        return arr, False

    if len(compact_regions) > 1:
        areas = [(r[2] - r[0]) * (r[3] - r[1]) for r in compact_regions]
        total_area = sum(areas)
        dominant_idx = max(range(len(areas)), key=lambda i: areas[i])
        if areas[dominant_idx] / total_area >= DOMINANT_AREA_FRACTION:
            dist_to_excluded = ndimage.distance_transform_edt(~excluded_mask) if excluded_mask.any() else None

            def is_tick_near_spectrum(region):
                if dist_to_excluded is None:
                    return False
                x0, y0, x1, y1 = region
                region_ink = ink[y0:y1, x0:x1]
                if not region_ink.any() or region_ink.sum() > TICK_MAX_INK_PIXELS:
                    return False
                return dist_to_excluded[y0:y1, x0:x1][region_ink].min() < TICK_MAX_DIST_TO_EXCLUDED

            compact_regions = [
                r
                for i, r in enumerate(compact_regions)
                if i == dominant_idx
                or not (_is_label_text(gray[r[1]:r[3], r[0]:r[2]]) or is_tick_near_spectrum(r))
            ]

    x0 = max(0, min(r[0] for r in compact_regions) - PADDING_PX)
    y0 = max(0, min(r[1] for r in compact_regions) - PADDING_PX)
    x1 = min(width, max(r[2] for r in compact_regions) + PADDING_PX)
    y1 = min(height, max(r[3] for r in compact_regions) + PADDING_PX)

    if (x1 - x0) * (y1 - y0) >= 0.9 * width * height:
        return arr, False

    return arr[y0:y1, x0:x1], True


def main():
    import sys

    image_path = Path(sys.argv[1])
    cropped, was_cropped = crop_to_structure(image_path)
    out_path = image_path.with_stem(image_path.stem + "_cropped")
    Image.fromarray(cropped).save(out_path)
    print(f"cropped={was_cropped} -> {out_path}")


if __name__ == "__main__":
    main()
