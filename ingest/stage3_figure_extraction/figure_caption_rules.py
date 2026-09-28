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

is_mechanism_related has no vision counterpart at all -- it's caption-only,
flagging figures that belong to a paper's mechanistic-investigation
discussion (proposed catalytic cycles, intermediates, DFT pathways), which
routinely contain theoretical/partial structures DECIMER isn't equipped
to read (dangling ligand bonds, transition states). Checked against real
caption/context text before adding: raw "mechanis*" matches 32 figures
corpus-wide, mostly noise from one shared block of SI procedural text
reused across ~20 consecutive figures -- but restricted to has_structures=1
(the only population this tag matters for), it's 13 matches, 0 confirmed
false positives (spot-checked a sample: a real proposed Fe-NHC
intermediate, plus ordinary real reagents/products cited as supporting
evidence in the same mechanistic section -- both legitimately belong to
"the mechanism discussion" even though only the former is itself a
theoretical structure). Deliberately not gated on has_structures the way
has_grid_layout is -- false positives at has_structures=0 are inert (the
tag is simply unused there), so there's no correctness reason to force it
off, unlike has_grid_layout where an ungated false positive would be a
real semantic contradiction.
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
MECHANISM_RE = re.compile(r"mechanis", re.IGNORECASE)


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

    if MECHANISM_RE.search(text):
        results["is_mechanism_related"] = 1

    return results
