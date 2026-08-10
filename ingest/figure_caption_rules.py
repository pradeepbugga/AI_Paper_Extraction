"""Caption/context keyword rules for Stage 3 figure tagging.

High-precision text overrides for the specific tags where pure CLIP vision
classification has a demonstrated, real failure mode (see figure_classify.py's
leave-one-out results): telling a DFT-computed structure from an X-ray one,
or identifying a spectrum's technique, are things a caption almost always
states plainly even when the pixels alone don't make it obvious.

Every pattern here was checked against real caption/context text from
figure_captions.json before being kept. Several broader candidates were
tried and dropped because they fired mostly on unrelated procedural prose,
not the property they were meant to signal: bare "DFT", "calculated", and
"free energy" mostly matched HRMS mass-calc reporting and Hammett/LFER
analysis (not DFT); "screening" matched procedure headings regardless of
what the actual figure looked like; "reaction flask/vial/setup" matched
routine synthesis prose in nearly every procedure, photographed or not.

Tags not covered here (has_structures, has_plotted_data) stay vision-only
on purpose -- captions in this corpus don't reliably describe composition,
and has_structures in particular is already working well from vision alone.
"""

import re

LEVEL_OF_THEORY_RE = re.compile(
    r"B3LYP|PBE0|M0[0-9]|def2|ω?B97|level of theory", re.IGNORECASE
)
XRAY_RE = re.compile(r"X-ray|ORTEP|crystal structure|crystallographic", re.IGNORECASE)
NMR_WITH_PARAMS_RE = re.compile(r"NMR\s*\(\s*\d+\s*MHz", re.IGNORECASE)
HPLC_RE = re.compile(r"\bHPLC\b", re.IGNORECASE)
HRMS_RE = re.compile(r"\bHRMS\b", re.IGNORECASE)
MS_MZ_RE = re.compile(r"\bMS\b\s*\(m/z\)", re.IGNORECASE)
IR_NEAT_RE = re.compile(r"\bIR\b\s*\(neat\)", re.IGNORECASE)
SCOPE_RE = re.compile(r"substrate scope|\bscope\b", re.IGNORECASE)
PHOTOGRAPH_RE = re.compile(r"photograph", re.IGNORECASE)


def classify_from_caption(text):
    """Returns {tag: 0/1} for only the tags a keyword rule actually fired
    on -- a tag with no match is simply absent from the dict, left for the
    vision classifier to decide rather than defaulted to 0.
    """
    if not text:
        return {}

    results = {}

    if LEVEL_OF_THEORY_RE.search(text):
        results["is_computed"] = 1
    elif XRAY_RE.search(text):
        # Measured, not computed -- an explicit negative even though pure
        # vision can't reliably tell an X-ray structure from a DFT one
        # (this was the specific confusion that motivated this module).
        results["is_computed"] = 0

    if (NMR_WITH_PARAMS_RE.search(text) or HPLC_RE.search(text)
            or HRMS_RE.search(text) or MS_MZ_RE.search(text) or IR_NEAT_RE.search(text)):
        results["is_spectrum"] = 1

    if SCOPE_RE.search(text):
        results["has_grid_layout"] = 1

    if PHOTOGRAPH_RE.search(text):
        results["is_photo"] = 1

    return results
