"""Runs DECIMER Segmentation (Mask R-CNN, trained specifically to locate
chemical structure depictions on journal pages) across every has_structures=1
figure in every paper's figure_tags.json, replacing the hand-built geometric
crop in structure_crop.py.

Must run in the `decimer_seg` conda env (Python 3.10) -- decimer-segmentation
pins an older TensorFlow/Mask-RCNN stack incompatible with the `decimer`
OCSR package's env. Writes one segment_manifest.json per paper:
{image_path: [segment_file_path, ...]} -- usually one segment, but a
multi-compound figure can legitimately produce several, and an image with
no detected structure produces an empty list (batch_decimer_extract.py
falls back to the original image in that case). Segment crops are written
alongside the originals under a "segments/" subfolder.

Crops to each detection's mask bbox (via get_expanded_masks -- `expand=True`'s
dilation still pulls in nearby disconnected ink, e.g. a separated substituent
label, before the bbox is taken), computed ourselves from the mask array
rather than trusting apply_mask's own crop, since apply_mask also whitens
every pixel outside the mask's own irregular outline and that's not needed
here -- the raw rectangular crop from the original image is enough once the
bbox is tight.

An earlier version of this script added a flat pixel margin (with a
same-figure-neighbor clamp) around this tight bbox, to recover a small
number of confirmed cases where the tight mask bbox clipped a real
substituent label. That margin was reverted: visual review across several
dense figures showed the tight mask-derived bbox is consistently clean.
That review turned out to be too small a sample -- a later corpus-wide audit
(comparing every segment's crop against its own parent image) found the mask
itself silently stops mid-label in ~1 in 8 segments corpus-wide (left/right
axis specifically), even where the parent image has plenty of untouched
whitespace past the mask's own boundary the model simply didn't reach into
(confirmed: DECIMER's Bpin->[Po] and OTf->[O]-style element hallucinations
trace directly back to this). Padding the whole image with a white border
before segmentation was tried as a fix and rejected: even at 300px of
padding (larger than some of the source images), the model's own mask
boundary barely moved and never recovered the missing label -- this isn't a
"starved for canvas space" artifact, so more padding doesn't solve it.
Truncation concentrates heavily in small, tightly-pre-cropped SI figures
(median parent-image area for a truncated segment is ~100K px^2 vs ~968K
px^2 corpus-wide) -- the same model behaves better given a large
multi-structure page image than a small single-compound crop, for reasons
that weren't tracked down further since a reliable post-hoc fix was found
instead.

Fix: after the tight mask bbox, walk outward from each edge over the
*parent image's own pixels* while real content continues, stopping only at
a genuinely large blank run -- recovers the label the mask cut short
without depending on the model's own (apparently capped) decision. This is
NOT the same as the earlier reverted flat-margin approach: a flat margin is
a shape with no relation to where the actual ink is, so on a dense page it
just as often grabs a neighbor's label as it does nothing at all. The
ink-walk only extends while it's still looking at real content, and is
capped three ways so it can never reproduce the old neighbor-bleed failure:
(1) a max pixel distance per side (EXTEND_MAX_PX), (2) never further than
halfway across the gap to another segment's own tight bbox in the same
figure (so two segments extending toward each other along the SAME axis can
only ever meet in the middle, never cross into each other's own detected
area), and (3) a final cross-segment pass after every segment in a figure
has computed its own extension independently: if two segments end up
overlapping where their original tight bboxes did not, both are reverted to
their tight bboxes entirely. (2) alone isn't sufficient -- confirmed on a
real dense figure that two segments extending along DIFFERENT axes (one
growing sideways, another below-left of it growing downward) can still meet
at a corner, since neither one's independent per-side computation can see
the other's own extension while computing its own; (3) is the guaranteed
backstop for that case.

The blank-run threshold for "this is a genuine stop, not just normal
spacing" needed real calibration, not a small arbitrary number: a first
version used a flat 6px blank run as the stop condition and mostly worked,
but a real case (a carbonyl "O" sitting above a double bond) had an exactly
6px gap between the bond and its own atom label -- colliding with the
threshold and stopping the walk one row short of content that was sitting
right there. Fixed by requiring a much larger blank run (MERGE_GAP_PX) before
accepting a gap as real, and always extending to the position of the LAST
actual content found within the cap (not to wherever a blank run happened to
start) -- this bridges normal bond-to-label typographic spacing while still
refusing to cross a genuinely large gap (or a sibling's own bbox, per (2)
above). Increasing the merge-gap doesn't reduce safety against neighbor-
bleed, since caps (1) and (2) bound the walk independently of how the gap
threshold is tuned.

Validated corpus-wide before shipping: fully recovered every confirmed
truncation case tested (Bpin, OTf, COOCH3, and the carbonyl-O case above,
across two different papers and both the left/right and top/bottom axes),
left a confirmed different-content case (an unrelated reagent-conditions
text block sitting close to a product structure) untouched rather than
absorbing it, and introduced zero NET new bbox overlaps on the two densest
scope-table figures in the corpus (35 and 59 segments) after the final
cross-segment resolve pass -- the only overlaps present in either figure's
final output were already in the raw, pre-extension Mask R-CNN detections
themselves, a separate and pre-existing over-segmentation issue on that one
figure, not something this change causes.

A side that exhausts its cap without ever finding a real gap -- or a segment
reverted by the cross-segment overlap-resolve pass -- is recorded in the
manifest as still-truncated (`truncated_sides`) even after the best recovery
attempt. Surfaced through to decimer_results.json and from there into
reaction_link.py's per-segment flag mechanism, so Stage 5 knows a segment's
crop (and therefore its SMILES) may still be incomplete even after this fix.
Maximizing recovery doesn't mean claiming perfection: this flag is the
recall backstop for whatever the extension still couldn't fix.
"""

import json
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from decimer_segmentation import get_expanded_masks

PAPERS_DIR = Path(__file__).resolve().parent.parent / "data" / "papers"

INK_DARK = 200                     # grayscale pixel value below this counts as "ink"
LABEL_GAP_LOOKAHEAD_PX = 15         # how far past a candidate gap to look for content resuming
GAP_BLANK_ROWS_COLS = 6             # baseline blank run considered for a candidate gap
MERGE_GAP_PX = GAP_BLANK_ROWS_COLS + LABEL_GAP_LOOKAHEAD_PX  # required blank run to accept a stop as real
POST_INK_BUFFER_PX = 3              # small cushion past the last real content found
EXTEND_MAX_PX = 70                  # hard cap per side, regardless of neighbor spacing
SIBLING_BUFFER_PX = 2               # stay this far clear of the halfway point to a sibling's bbox

# Small fixed cushion applied to a two-pass-recovered bbox (see
# run_paper_two_pass) -- unlike EXTEND_MAX_PX, this isn't trying to
# "recover" anything; the wide-crop DECIMER mask should already be complete
# by the time this runs (see pdf_ingest.py's WIDE_MARGIN_K), so this is
# purely a small anti-aliasing/edge-softness safety margin.
SAFETY_MARGIN_PX = 4


def has_structures(tags):
    tag = tags.get("has_structures")
    return bool(tag) and tag[0] == 1


def _is_blank_line(line):
    return not (line < INK_DARK).any()


def _side_limit(bbox, siblings, side):
    """How far this side may extend before crossing into a sibling
    segment's own tight bbox -- a hard geometric ceiling independent of the
    gap-walk, since densely-packed figures can have neighbor content with no
    blank gap at all separating it from this segment's own label."""
    x0, y0, x1, y1 = bbox
    limit = None
    for sx0, sy0, sx1, sy1 in siblings:
        y_overlaps = min(y1, sy1) > max(y0, sy0)
        x_overlaps = min(x1, sx1) > max(x0, sx0)
        if side in ("left", "right") and not y_overlaps:
            continue
        if side in ("top", "bottom") and not x_overlaps:
            continue
        if side == "left" and sx1 <= x0:
            limit = sx1 if limit is None else max(limit, sx1)
        elif side == "right" and sx0 >= x1:
            limit = sx0 if limit is None else min(limit, sx0)
        elif side == "top" and sy1 <= y0:
            limit = sy1 if limit is None else max(limit, sy1)
        elif side == "bottom" and sy0 >= y1:
            limit = sy0 if limit is None else min(limit, sy0)
    return limit


def _side_cap(bbox, siblings, side):
    """Caps this side's extension at HALF the gap to the nearest sibling's
    own (original, unextended) bbox, not the whole gap -- that sibling is
    computing its own cap the same way, so each side only ever claims its
    own half. Using the full gap would let two segments extend toward each
    other from opposite sides of the same gap and still collide in the
    middle, since each only knows the other's pre-extension position. This
    alone doesn't cover two segments extending along DIFFERENT axes and
    meeting at a corner -- see resolve_overlaps for that backstop."""
    x0, y0, x1, y1 = bbox
    limit = _side_limit(bbox, siblings, side)
    if limit is None:
        return EXTEND_MAX_PX
    if side == "left":
        room = (x0 - limit) // 2 - SIBLING_BUFFER_PX
    elif side == "right":
        room = (limit - x1) // 2 - SIBLING_BUFFER_PX
    elif side == "top":
        room = (y0 - limit) // 2 - SIBLING_BUFFER_PX
    else:
        room = (limit - y1) // 2 - SIBLING_BUFFER_PX
    return max(min(EXTEND_MAX_PX, room), 0)


def _fetch_line(gray, x0, y0, x1, y1, side, step):
    h, w = gray.shape
    if side == "left":
        pos = x0 - step
        return pos, pos >= 0, (gray[y0:y1, pos] if pos >= 0 else None)
    if side == "right":
        pos = x1 - 1 + step
        return pos, pos < w, (gray[y0:y1, pos] if pos < w else None)
    if side == "top":
        pos = y0 - step
        return pos, pos >= 0, (gray[pos, x0:x1] if pos >= 0 else None)
    pos = y1 - 1 + step
    return pos, pos < h, (gray[pos, x0:x1] if pos < h else None)


def _walk_side(gray, x0, y0, x1, y1, side, cap):
    """Returns (new_edge, incomplete). Extends to the position of the LAST
    real content found within `cap`, bridging any blank run shorter than
    MERGE_GAP_PX (ordinary spacing between a bond and its atom label, or
    between letters/words) rather than stopping at the first qualifying
    blank run -- see module docstring for the real case that motivated this.
    incomplete=True means content was still present right at the boundary of
    what this side was allowed to take (cap reached without a genuine
    trailing gap), so recovery is likely still partial."""
    h, w = gray.shape
    blank_run = 0
    last_content_step = 0
    reached_cap_with_content = False
    for step in range(1, cap + 1):
        pos, in_bounds, line = _fetch_line(gray, x0, y0, x1, y1, side, step)
        if not in_bounds:
            break
        if _is_blank_line(line):
            blank_run += 1
            if blank_run >= MERGE_GAP_PX:
                break
        else:
            blank_run = 0
            last_content_step = step
            if step == cap:
                reached_cap_with_content = True

    if last_content_step == 0:
        edge = {"left": x0, "right": x1, "top": y0, "bottom": y1}[side]
        return edge, False

    # POST_INK_BUFFER_PX is a cushion added past the last step CONFIRMED
    # in-bounds by _fetch_line -- but that confirmation only guarantees
    # last_content_step itself was in-bounds, not last_content_step + the
    # buffer. Left unclamped, a segment whose real content runs right up to
    # the image's own edge (no sibling there to cap it, so cap defaults to
    # EXTEND_MAX_PX) can compute an edge a few pixels past position 0 (or
    # past w/h on the other sides) -- confirmed on a real case
    # (copper_iron_2025/images_SI/page22_fig0.png) where this produced
    # bbox=[14, -3, 294, 168] on a 307x165 image. That negative y0 then hit
    # numpy's negative-index wraparound at the actual crop step, silently
    # producing a nonsensical 3-pixel-tall crop instead of an error.
    extend_to = min(last_content_step + POST_INK_BUFFER_PX, cap)
    if side == "left":
        edge = max(0, x0 - extend_to)
    elif side == "top":
        edge = max(0, y0 - extend_to)
    elif side == "right":
        edge = min(w, x1 + extend_to)
    else:
        edge = min(h, y1 + extend_to)
    return edge, reached_cap_with_content


def extend_bbox(gray, bbox, siblings=()):
    """siblings: bboxes of every OTHER segment detected in the same parent
    figure -- required so extension on a densely-packed page can never grab
    a neighbor's own structure/label, even where there's no real whitespace
    gap for the ink-walk to stop at on its own. Returns
    (new_bbox, truncated_sides) where truncated_sides lists any edge that
    exhausted its cap without finding a real gap -- still possibly clipped
    even after the best safe recovery attempt."""
    x0, y0, x1, y1 = bbox
    truncated_sides = []

    left_cap = _side_cap(bbox, siblings, "left")
    new_x0, left_incomplete = _walk_side(gray, x0, y0, x1, y1, "left", left_cap)

    right_cap = _side_cap(bbox, siblings, "right")
    new_x1, right_incomplete = _walk_side(gray, x0, y0, x1, y1, "right", right_cap)

    # Sibling relevance for top/bottom must be checked against the ALREADY
    # left/right-extended x-range, since that's the range the walk itself
    # scans -- using the original narrower x-range here let a sibling that
    # only becomes horizontally adjacent after widening slip through
    # unclamped (confirmed: caused a real overlap before this fix, on the
    # same figure resolve_overlaps also had to backstop for a different
    # reason -- see both docstrings).
    widened_bbox = (new_x0, y0, new_x1, y1)
    top_cap = _side_cap(widened_bbox, siblings, "top")
    new_y0, top_incomplete = _walk_side(gray, new_x0, y0, new_x1, y1, "top", top_cap)

    bottom_cap = _side_cap(widened_bbox, siblings, "bottom")
    new_y1, bottom_incomplete = _walk_side(gray, new_x0, y0, new_x1, y1, "bottom", bottom_cap)

    # Second left/right pass, now using the top/bottom-extended y-range --
    # fixes a real case (suzuki_iron_2024/images_SI/page29_fig0.png) where a
    # label sits DIAGONALLY adjacent to the tight bbox (both above AND to
    # the left of it, e.g. "F3C" above-left of the ring's own tight-mask
    # region). The first left/right pass above only sees the ORIGINAL,
    # unwidened y-range, so it can't find a label that only becomes visible
    # once the y-range has already been extended upward -- it stopped at a
    # small unrelated ink fragment within the narrow original y-range,
    # x0=27, well short of the label's real left edge at x=15. Re-running
    # left/right with the final y-range catches this. Budget is shared with
    # the first pass (capped at the same EXTEND_MAX_PX from the ORIGINAL
    # tight edge, not an additional EXTEND_MAX_PX on top) so this doesn't
    # loosen the documented "never more than EXTEND_MAX_PX per side"
    # guarantee -- it only lets a diagonal label be found within that same
    # budget, not extend further than a purely horizontal/vertical case
    # could.
    widened_bbox2 = (new_x0, new_y0, new_x1, new_y1)

    left_cap2 = max(0, left_cap - (x0 - new_x0))
    retry_x0, left_incomplete2 = _walk_side(gray, new_x0, new_y0, new_x1, new_y1, "left", left_cap2)
    if retry_x0 < new_x0:
        new_x0, left_incomplete = retry_x0, left_incomplete2

    right_cap2 = max(0, right_cap - (new_x1 - x1))
    retry_x1, right_incomplete2 = _walk_side(gray, new_x0, new_y0, new_x1, new_y1, "right", right_cap2)
    if retry_x1 > new_x1:
        new_x1, right_incomplete = retry_x1, right_incomplete2

    for side, incomplete in [
        ("left", left_incomplete), ("right", right_incomplete),
        ("top", top_incomplete), ("bottom", bottom_incomplete),
    ]:
        if incomplete:
            truncated_sides.append(side)

    return [new_x0, new_y0, new_x1, new_y1], truncated_sides


def resolve_overlaps(tight_bboxes, final_bboxes):
    """Guarantees zero NEW overlaps among a figure's segments. The per-side
    half-gap clamp in _side_cap prevents two segments from colliding when
    they extend toward each other along the SAME axis, but two segments can
    still meet at a corner when they extend along DIFFERENT axes (one
    growing sideways, another below-left of it growing downward) -- neither
    one's independent per-side computation can see the other's own
    extension while computing its own. Confirmed on a real 59-segment dense
    figure. Fallback: if extension introduces an overlap that wasn't already
    present in the raw tight detections, revert both segments in that pair
    back to their tight bboxes entirely -- loses recovery only for that
    specific rare conflicting pair, never corrupts data. Returns
    (resolved_bboxes, reverted_flags)."""
    def overlaps(a, b):
        ax0, ay0, ax1, ay1 = a
        bx0, by0, bx1, by1 = b
        return not (ax1 <= bx0 or bx1 <= ax0 or ay1 <= by0 or by1 <= ay0)

    boxes = [list(b) for b in final_bboxes]
    reverted = [False] * len(boxes)
    n = len(boxes)
    for i in range(n):
        for j in range(i + 1, n):
            if overlaps(tight_bboxes[i], tight_bboxes[j]):
                continue  # pre-existing in the raw detections, not ours to fix
            if not overlaps(boxes[i], boxes[j]):
                continue
            boxes[i] = list(tight_bboxes[i])
            boxes[j] = list(tight_bboxes[j])
            reverted[i] = reverted[j] = True
    return boxes, reverted


def _load_wide_metadata(paper_dir):
    """image_path -> its raw_extraction.json/SI_raw_extraction.json figure
    entry, for every vector-figure that has a wide-crop counterpart (see
    pdf_ingest.py's WIDE_MARGIN_K). Only vector_region figures ever have
    one; raster images fall back to the single-pass path below."""
    meta = {}
    for filename in ("raw_extraction.json", "SI_raw_extraction.json"):
        path = paper_dir / filename
        if not path.exists():
            continue
        data = json.load(open(path))
        for page in data["pages"]:
            for im in page["images"]:
                if "wide_path" in im:
                    meta[im["path"]] = im
    return meta


def _mask_bboxes(masks):
    out = []
    for i in range(masks.shape[2]):
        ys, xs = np.where(masks[:, :, i])
        if ys.size == 0:
            continue
        out.append((i, [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]))
    return out


def _boxes_overlap(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1


def two_pass_segment(paper_dir, image_path, meta_entry):
    """Two-pass DECIMER recovery -- see pdf_ingest.py's WIDE_MARGIN_K
    comment for why this exists. DECIMER's own mask on a tightly-cropped
    isolated compound can silently exclude a label sitting diagonally or
    vertically outside its own tight bbox; confirmed this is independent of
    how much BLANK padding is added (batch_segment.py's history), but fully
    resolved by revealing real surrounding PAGE content instead.

    Pass 1 runs DECIMER on the existing tight crop to get each segment's own
    "anchor" mask. Pass 2 runs DECIMER on a much wider crop of the same
    region (pre-rendered by pdf_ingest.py from the source PDF) and keeps
    only the wide-crop mask(s) overlapping each anchor, discarding anything
    else the wider frame reveals -- neighboring compounds, captions, tables.
    This relies on DECIMER keeping unrelated content as separate masks
    rather than merging it into the anchor; confirmed on real cases
    (suzuki_iron_2024 SI page 29's four compounds, a caption-adjacent
    structure) that it reliably does, the same "high spatial precision"
    that lets it tell same-page structures apart in the full-page case.
    This is NOT guaranteed on a densely packed multi-compound figure, where
    DECIMER can still merge several compounds into one mask on its own,
    independent of this two-pass step -- a separate, harder problem (see
    too_many_fragments in decimer_extract.py) that this fix doesn't touch.

    Returns (seg_idxs, final_bboxes, truncated_sides_list, wide_img,
    ground_bboxes) -- final_bboxes/wide_img are the same shape
    _single_pass_segment returns and are used to crop+save the actual
    segment pixels (in the WIDE image's own pixel space, since that's where
    the recovered pixels live). ground_bboxes is the SAME boxes translated
    back into the TIGHT image's pixel space and clamped to its bounds --
    this, not final_bboxes, is what must be stored as each segment's
    "bbox" in the manifest, since that field is Stage 5's grounding
    coordinate for Set-of-Mark: reaction_link.py draws boxes and crops
    chunks from the TIGHT image (images/pageN_figM.png), never the wide
    one. Storing final_bboxes there instead is a real, confirmed bug (not
    hypothetical): the wide and tight images are offset from each other by
    the render margin, so the same raw pixel numbers address DIFFERENT real
    page content in each -- confirmed on redox_neutral_2024/page5_fig0.png,
    where a segment's saved crop (correctly cropped from the wide image at
    its own bbox) showed one structure while Stage 5, drawing that same
    bbox on the tight image, landed on a completely different, unrelated
    structure a few hundred pixels away and correctly read its label --
    producing a real compound_id/yield attached to the wrong structure's
    SMILES. ground_bboxes can legitimately be smaller than the segment's
    real content for recovered/previously-truncated cases (clamped at the
    tight image's own edge) -- that's fine, it only needs to point Set-of-
    Mark at roughly the right place, not capture every pixel.

    Returns None if the wide crop is missing/unreadable, signaling the
    caller to fall back to the single-pass path."""
    tight_path = paper_dir / image_path
    wide_path = paper_dir / meta_entry["wide_path"]
    if not wide_path.exists():
        return None

    tight_img = cv2.imread(str(tight_path))
    wide_img = cv2.imread(str(wide_path))
    if tight_img is None or wide_img is None:
        return None

    tight_h, tight_w = tight_img.shape[:2]
    anchors = _mask_bboxes(get_expanded_masks(tight_img))
    if not anchors:
        return [], [], [], wide_img, []
    wide_boxes = _mask_bboxes(get_expanded_masks(wide_img))

    # Both crops were rendered at the same zoom from the same PDF page (see
    # pdf_ingest.py) -- derive it from each crop's own stored width/bbox
    # rather than assuming a hardcoded constant, so this stays correct even
    # if the render zoom ever changes.
    tight_bbox_pt = meta_entry["bbox"]
    wide_bbox_pt = meta_entry["wide_bbox"]
    zoom = tight_img.shape[1] / (tight_bbox_pt[2] - tight_bbox_pt[0])
    dx = (tight_bbox_pt[0] - wide_bbox_pt[0]) * zoom
    dy = (tight_bbox_pt[1] - wide_bbox_pt[1]) * zoom

    wide_h, wide_w = wide_img.shape[:2]
    seg_idxs, padded_bboxes, anchors_in_wide, truncated_sides_list = [], [], [], []
    for seg_idx, anchor in anchors:
        anchor_in_wide = [anchor[0] + dx, anchor[1] + dy, anchor[2] + dx, anchor[3] + dy]
        matched = [b for _, b in wide_boxes if _boxes_overlap(b, anchor_in_wide)]
        if matched:
            recovered = [
                min(b[0] for b in matched), min(b[1] for b in matched),
                max(b[2] for b in matched), max(b[3] for b in matched),
            ]
            truncated_sides = []
        else:
            # The wide crop is a strict superset of the tight crop's own
            # content, so this shouldn't normally happen -- fall back to the
            # anchor itself (translated) rather than losing the segment, and
            # flag it since the wide-crop recovery step didn't actually run.
            recovered = anchor_in_wide
            truncated_sides = ["unmatched"]

        x0 = max(0, recovered[0] - SAFETY_MARGIN_PX)
        y0 = max(0, recovered[1] - SAFETY_MARGIN_PX)
        x1 = min(wide_w, recovered[2] + SAFETY_MARGIN_PX)
        y1 = min(wide_h, recovered[3] + SAFETY_MARGIN_PX)
        seg_idxs.append(seg_idx)
        padded_bboxes.append([int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))])
        anchors_in_wide.append([int(round(v)) for v in anchor_in_wide])
        truncated_sides_list.append(truncated_sides)

    # Safety net against the small-margin padding introducing a new overlap
    # between two segments that weren't already touching at their own
    # anchors -- same backstop used by the single-pass path.
    final_bboxes, reverted = resolve_overlaps(anchors_in_wide, padded_bboxes)
    for i, was_reverted in enumerate(reverted):
        if was_reverted:
            truncated_sides_list[i] = list(set(truncated_sides_list[i]) | {"overlap_reverted"})

    # Translate back to the TIGHT image's own pixel space for grounding --
    # see this function's docstring for why storing final_bboxes (wide-
    # space) as the manifest "bbox" instead would be wrong.
    ground_bboxes = []
    for bbox in final_bboxes:
        gx0 = max(0, min(tight_w, bbox[0] - dx))
        gy0 = max(0, min(tight_h, bbox[1] - dy))
        gx1 = max(0, min(tight_w, bbox[2] - dx))
        gy1 = max(0, min(tight_h, bbox[3] - dy))
        ground_bboxes.append([int(round(gx0)), int(round(gy0)), int(round(gx1)), int(round(gy1))])

    return seg_idxs, final_bboxes, truncated_sides_list, wide_img, ground_bboxes


def _single_pass_segment(full_path):
    """Original tight-mask + ink-walk-extension behavior, unchanged --
    fallback path for raster images and any vector figure whose wide crop
    is missing or unreadable. Returns (seg_idxs, final_bboxes,
    truncated_sides_list, img), same shape as two_pass_segment."""
    img = cv2.imread(str(full_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    masks = get_expanded_masks(img)

    # Compute every detection's own tight bbox first -- extension needs to
    # see every sibling in the figure before any one of them decides how
    # far it's safe to extend.
    seg_idxs, tight_bboxes = [], []
    for seg_idx in range(masks.shape[2]):
        ys, xs = np.where(masks[:, :, seg_idx])
        if ys.size == 0:
            continue
        seg_idxs.append(seg_idx)
        tight_bboxes.append([int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1])

    extended_bboxes, truncated_sides_list = [], []
    for i, bbox in enumerate(tight_bboxes):
        siblings = [b for j, b in enumerate(tight_bboxes) if j != i]
        final_bbox, truncated_sides = extend_bbox(gray, bbox, siblings=siblings)
        extended_bboxes.append(final_bbox)
        truncated_sides_list.append(truncated_sides)

    resolved_bboxes, reverted = resolve_overlaps(tight_bboxes, extended_bboxes)
    for i, was_reverted in enumerate(reverted):
        if was_reverted:
            # Extension was reverted by resolve_overlaps -- back to the
            # tight mask bbox, so this crop may be truncated on any side
            # regardless of what extend_bbox itself reported.
            truncated_sides_list[i] = list(set(truncated_sides_list[i]) | {"left", "right", "top", "bottom"})

    return seg_idxs, resolved_bboxes, truncated_sides_list, img


def _save_segments(segments_dir, stem, seg_idxs, bboxes, truncated_sides_list, src_img, ground_bboxes=None):
    """ground_bboxes, if given, are stored as each entry's "bbox" instead of
    the crop bboxes -- used by the two-pass path, where the crop itself has
    to come from the wide image (that's where the recovered pixels are) but
    the stored "bbox" needs to be in the TIGHT image's pixel space, since
    that's what Stage 5 grounds Set-of-Mark on (see two_pass_segment's
    docstring for the real bug this fixes). Defaults to bboxes itself,
    which is already tight-space for the single-pass path."""
    if ground_bboxes is None:
        ground_bboxes = bboxes
    img_h, img_w = src_img.shape[:2]
    seg_entries = []
    for seg_idx, bbox, ground_bbox, truncated_sides in zip(seg_idxs, bboxes, ground_bboxes, truncated_sides_list):
        px0, py0, px1, py1 = bbox
        # Defensive clamp independent of any upstream clamping -- a
        # negative index here silently wraps around in numpy instead of
        # raising, producing a nonsensical crop rather than an error.
        px0, py0 = max(0, px0), max(0, py0)
        px1, py1 = min(img_w, px1), min(img_h, py1)
        seg = src_img[py0:py1, px0:px1]
        if seg.shape[0] == 0 or seg.shape[1] == 0:
            continue
        seg_filename = f"{stem}_seg{seg_idx}.png"
        cv2.imwrite(str(segments_dir / seg_filename), seg)
        seg_entries.append({
            "path": f"segments/{seg_filename}",
            "bbox": list(ground_bbox),
            "truncated_sides": truncated_sides,
        })
    return seg_entries


def run_paper(paper_dir):
    tags_path = paper_dir / "figure_tags.json"
    if not tags_path.exists():
        return {}

    with open(tags_path) as f:
        all_tags = json.load(f)

    targets = [path for path, tags in all_tags.items() if has_structures(tags)]
    segments_dir = paper_dir / "segments"
    segments_dir.mkdir(exist_ok=True)
    wide_meta = _load_wide_metadata(paper_dir)

    manifest = {}
    pbar = tqdm(targets, desc=paper_dir.name, unit="fig", mininterval=1.0)
    for image_path in pbar:
        full_path = paper_dir / image_path
        if not full_path.exists():
            tqdm.write(f"  MISSING {image_path}")
            continue
        pbar.set_postfix_str(image_path[-40:])
        start = time.time()

        result = None
        if image_path in wide_meta:
            result = two_pass_segment(paper_dir, image_path, wide_meta[image_path])
        if result is not None:
            seg_idxs, bboxes, truncated_sides_list, src_img, ground_bboxes = result
        else:
            seg_idxs, bboxes, truncated_sides_list, src_img = _single_pass_segment(full_path)
            ground_bboxes = bboxes

        elapsed = time.time() - start
        stem = Path(image_path).stem.replace("/", "_")
        seg_entries = _save_segments(
            segments_dir, stem, seg_idxs, bboxes, truncated_sides_list, src_img, ground_bboxes=ground_bboxes,
        )

        manifest[image_path] = seg_entries
        tqdm.write(f"  {image_path} -> {len(seg_entries)} segment(s) ({elapsed:.1f}s)")

    return manifest


def main():
    paper_dirs = sorted(p for p in PAPERS_DIR.iterdir() if p.is_dir())
    for paper_dir in paper_dirs:
        manifest = run_paper(paper_dir)
        if not manifest:
            continue
        out_path = paper_dir / "segment_manifest.json"
        with open(out_path, "w") as f:
            json.dump(manifest, f, indent=2)
        total_segments = sum(len(v) for v in manifest.values())
        zero_segment = sum(1 for v in manifest.values() if not v)
        truncated = sum(1 for v in manifest.values() for s in v if s["truncated_sides"])
        print(
            f"{paper_dir.name}: {len(manifest)} figures, {total_segments} segments total, "
            f"{zero_segment} with no detected structure, {truncated} still truncated after recovery -> {out_path}"
        )


if __name__ == "__main__":
    main()
