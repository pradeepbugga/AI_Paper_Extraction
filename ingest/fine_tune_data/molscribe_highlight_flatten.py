"""Preprocessing fix for a real MolScribe decoder bug found by scanning
this project's full corpus (2,145 real segments): whenever a benzene
ring's interior is covered by a solid pastel color-fill highlight (a
common SI-figure convention -- e.g. shading the "site of interest" ring
gray/cyan, sometimes with one bond bolded in a second color), the decoder
hallucinates an extra atom near the ring's geometric center and wires it
into the graph with extra bonds, producing an impossible bicyclic/fused
ring topology. Confirmed via raw-atom-coordinate inspection
(`return_atoms_bonds=True`): the phantom atom sits at a genuinely new
coordinate, not a duplicate of any real vertex, and is specific to rings
under a color fill -- the same rings decode perfectly once the fill is
removed and nothing else about the image (font, resolution, compression,
bond angles) changes.

Fix: detect whether the image actually contains a real color-fill region
(a light, pastel color occupying a meaningful fraction of the image) and,
only if so, flatten every pixel to pure black/white by a lightness
threshold -- pixels lighter than the threshold become background, darker
pixels (including a differently-colored highlighted bond, which is real
structure, not fill) become foreground.

This MUST be gated. An earlier, ungated version of this same threshold
applied unconditionally caused a real regression on a small, naturally
blue-inked image containing an already-fragile fused abbreviation label
(`TsN`) -- the hard binarization thinned an already-marginal anti-aliased
glyph on a tiny (148x72px) image, breaking a genuinely correct token it
had gotten right before. The image had no fill at all; its own ink color
(blue, ~88 average lightness) is dark and legitimate, not a wash-out
highlight. The detector below distinguishes the two cases by checking
the LIGHTNESS OF THE DOMINANT SECONDARY COLOR ITSELF: a real highlight
fill is a large fraction of a color that is itself light (~180-245,
meant to be a translucent background wash you can still read black ink
through); real ink -- whatever its hue -- is dark (meant for contrast).
Verified: fires on 26/26 known real-fill images (100% of the images this
was built for, from `copper_iron_2025`), does not fire on any of 3 known
clean/already-correct images from other papers (including the exact
image that regressed under the earlier ungated version).

Quantization note: colors are bucketed to the nearest QUANTIZE=8 to
absorb anti-aliasing noise around a genuine fill's edges into one
dominant bucket. The upper lightness bound (245) is deliberately below
248 -- pure white (255,255,255) quantized by `// 8 * 8` lands exactly on
248, and an earlier version of this detector used an upper bound of 253,
which let that quantized-white bucket itself get misidentified as a
"light fill" on every image (it's ~90%+ of any image by construction).
Keep the upper bound comfortably below 248 or the gate stops gating.
"""
import cv2
import numpy as np

LIGHTNESS_THRESHOLD = 200
FILL_MIN_FRACTION = 0.02
FILL_MIN_LIGHTNESS = 180
FILL_MAX_LIGHTNESS = 245
QUANTIZE = 8


def has_highlight_fill(img_bgr):
    """True if img_bgr contains a real color-fill highlight region."""
    arr = img_bgr.astype(int)
    total = arr.shape[0] * arr.shape[1]
    quantized = (arr // QUANTIZE) * QUANTIZE
    colors, counts = np.unique(quantized.reshape(-1, 3), axis=0, return_counts=True)
    for color, count in zip(colors, counts):
        if count / total > FILL_MIN_FRACTION and FILL_MIN_LIGHTNESS < color.mean() < FILL_MAX_LIGHTNESS:
            return True
    return False


def flatten_highlights(img_bgr):
    """Binarize to pure black/white by per-pixel average lightness."""
    lightness = img_bgr.astype(float).mean(axis=2)
    out = np.where(lightness[..., None] > LIGHTNESS_THRESHOLD, 255, 0).astype("uint8")
    return np.repeat(out, 3, axis=2)


def preprocess(img_bgr):
    """Gated preprocessing pass: flattens color-fill highlights if and only
    if one is actually detected; returns the image untouched otherwise.

    Returns (processed_img_bgr, was_modified: bool).
    """
    if has_highlight_fill(img_bgr):
        return flatten_highlights(img_bgr), True
    return img_bgr, False


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
