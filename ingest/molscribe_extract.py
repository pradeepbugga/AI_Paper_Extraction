"""Runs MolScribe structure recognition (via fine_tune_data's ensemble/
OCR-correction pipeline) on has_structures=1 figures, replacing the plain
DECIMER OCSR path (decimer_extract.py) that had been the production Stage 3
recognizer despite the MolScribe track's extensive dictionary/OCR-correction
work (handoff_16-18) never actually being wired in -- confirmed 2026-08-24
that decimer_results.json (what Stage 5 reads) was still 100% DECIMER
output, 9.8% RDKit-invalid corpus-wide, while the validated MolScribe path
measures ~3.4% on the same corpus.

Deliberately mirrors decimer_extract.py's extract_structure() output
contract field-for-field (smiles, mean_confidence, need_human_review,
detected_missing_abbreviations, has_generic_substituent, too_many_fragments,
rdkit_valid) so batch_molscribe_extract.py can write into the exact same
decimer_results.json shape Stage 4/5 already consume -- no downstream code
needs to change.

Must run in the dedicated `molscribe_venv` (see reference_runpod_pod.md),
NOT the `decimer_seg` or `paper_extraction` conda envs -- MolScribe and
DECIMER were kept in separate environments throughout this project
specifically to avoid dependency conflicts (TensorFlow vs PyTorch stacks),
so this module cannot import anything from decimer_extract.py directly
(its top-level `from DECIMER import predict_SMILES` would crash on import
here). The annotation-stripping and fragment/generic-substituent checks
below are therefore deliberate duplicates of decimer_extract.py's own
model-agnostic logic, not a shared import -- keep both in sync by hand if
either changes.

Two real differences from the DECIMER-path checks, confirmed empirically
before writing this module (not assumed):
- **Generic-substituent detection**: DECIMER emits literal `[R1]`/`[X]`/`[Z]`
  bracket tokens for an unresolved substituent; MolScribe emits a plain `*`
  wildcard atom instead (confirmed directly:
  `copper_iron_2025/segments/page2_fig1_seg0.png` decodes to
  `*[S@TB2](N)(=O)=O` under MolScribe, vs `NS(=O)(=O)[R1]` under DECIMER for
  the same crop). `_has_generic_substituent` checks for `*` here, not the
  DECIMER bracket pattern.
- **Missing-abbreviation OCR check**: decimer_extract.py's
  KNOWN_MISSING_ABBREVIATIONS/Ts-detection workaround exists because
  DECIMER's training vocabulary has no entry for the plain "Ts" abbreviation
  at all. MolScribe's own dictionary already ships a working "Ts" entry
  (confirmed via molscribe_constants_patch.py's docstring -- it patches
  many other gaps but never needed to add Ts), and molscribe_ocr_correction's
  OCR-override pass (wired into predict_best below) is a strictly more
  general mechanism for the same underlying problem (any bracket-atom
  label OCR reads with high confidence but the raw decode token doesn't
  match). So `detected_missing_abbreviations` is kept in the output schema
  for compatibility (Stage 5 code reads it) but is always `[]` here --
  there is no known MolScribe-specific vocabulary gap needing a dedicated
  OCR hunt the way DECIMER's Ts gap did.
"""

import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

import pytesseract

MEAN_CONFIDENCE_THRESHOLD = 0.85

# --- Annotation stripping (duplicated from decimer_extract.py -- see
# module docstring for why this can't be a cross-venv import) ---
OCR_MIN_CONFIDENCE = 50
OCR_UPSCALE = 4
JUNK_ANNOTATION_TOKEN_RE = re.compile(r"%|=|[()]|^[A-Za-z]{6,}$")
JUNK_NUMBER_TOKEN_RE = re.compile(r"^\d{1,3}[a-z]?$")
JUNK_NUMBER_MIN_CONFIDENCE = 85
JUNK_MASK_PADDING_PX = 2
OCR_PAD_PX = 20

# --- Fragment/generic-substituent checks (same model-agnostic logic as
# decimer_extract.py; generic-substituent DETECTION differs, see above) ---
MAX_DISCONNECTED_FRAGMENTS = 8
MIN_HEAVY_ATOMS_FOR_LARGE_FRAGMENT = 5
MAX_LARGE_FRAGMENTS = 2

# Rough per-fragment heavy-atom count straight from SMILES text, for the
# unparseable-SMILES fallback below -- matches bracket atoms as one unit,
# then two-letter halogens, then the organic-subset single letters.
# Confirmed calibration against real cases before adopting: small
# counterions/junk ([Cl-], [Fe], [Na+], CC) all score 1-2, a real ring
# (c1ccccc1) scores 6 -- same rough boundary MIN_HEAVY_ATOMS_FOR_LARGE_FRAGMENT
# already assumes for the RDKit-parsed path.
_ATOM_TOKEN_RE = re.compile(r"\[[^\]]*\]|Br|Cl|[BCNOSPFI]|[bcnosp]")


def _approx_heavy_atoms(fragment_smiles):
    return len(_ATOM_TOKEN_RE.findall(fragment_smiles))


def _has_generic_substituent(smiles):
    return "*" in smiles


def _has_too_many_fragments(smiles, mol):
    """Same logic/thresholds as decimer_extract.py's version -- see that
    module's docstring for the full derivation. Model-agnostic: operates
    only on the already-decoded SMILES/mol, not on anything DECIMER- or
    MolScribe-specific.

    The mol=None fallback used to be pure fragment COUNT vs
    MAX_DISCONNECTED_FRAGMENTS(8) -- a real, confirmed miss: a genuine
    3-compound merge (redox_neutral_2024/images_page5_fig0_seg47.png,
    each fragment a full tosyl-sulfonamide-piperidine-alkyne structure)
    produced a SMILES RDKit couldn't even parse, so it fell straight to
    this fallback, and 3 fragments is nowhere near 8 -- never flagged,
    2026-08-26. The primary size-aware signal (n_large fragments with
    >=5 heavy atoms) is exactly the check that would have caught it, but
    it's gated behind `mol is not None` -- and a genuine multi-compound
    merge is disproportionately likely to ALSO break RDKit's parser
    (bonds crossing between two unrelated compounds create nonsensical
    valence), so the fallback needs the same size-aware logic, just
    estimated from raw text instead of a parsed Mol."""
    if mol is not None:
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        n_large = sum(1 for f in frags if f.GetNumAtoms() >= MIN_HEAVY_ATOMS_FOR_LARGE_FRAGMENT)
        return n_large >= MAX_LARGE_FRAGMENTS or len(frags) >= MAX_DISCONNECTED_FRAGMENTS
    fragments = smiles.split(".")
    n_large = sum(1 for f in fragments if _approx_heavy_atoms(f) >= MIN_HEAVY_ATOMS_FOR_LARGE_FRAGMENT)
    return n_large >= MAX_LARGE_FRAGMENTS or len(fragments) >= MAX_DISCONNECTED_FRAGMENTS


def _strip_annotation_text(img):
    """Identical approach to decimer_extract.py's version -- paints over
    OCR-detected yield/compound-ID/condition annotation text so OCSR only
    ever sees the structure itself. See that module's docstring for the
    padding rationale."""
    padded = Image.new("RGB", (img.width + 2 * OCR_PAD_PX, img.height + 2 * OCR_PAD_PX), (255, 255, 255))
    padded.paste(img, (OCR_PAD_PX, OCR_PAD_PX))
    scaled = padded.resize((padded.width * OCR_UPSCALE, padded.height * OCR_UPSCALE), Image.LANCZOS)
    data = pytesseract.image_to_data(scaled, config="--psm 11", output_type=pytesseract.Output.DICT)

    cleaned = img.copy()
    draw = ImageDraw.Draw(cleaned)
    for i in range(len(data["text"])):
        token = data["text"][i].strip()
        conf = data["conf"][i]
        if not token:
            continue
        is_junk = (conf >= OCR_MIN_CONFIDENCE and JUNK_ANNOTATION_TOKEN_RE.search(token)) or (
            conf >= JUNK_NUMBER_MIN_CONFIDENCE and JUNK_NUMBER_TOKEN_RE.search(token)
        )
        if not is_junk:
            continue
        x = data["left"][i] // OCR_UPSCALE - OCR_PAD_PX
        y = data["top"][i] // OCR_UPSCALE - OCR_PAD_PX
        w = data["width"][i] // OCR_UPSCALE
        h = data["height"][i] // OCR_UPSCALE
        draw.rectangle(
            [x - JUNK_MASK_PADDING_PX, y - JUNK_MASK_PADDING_PX,
             x + w + JUNK_MASK_PADDING_PX, y + h + JUNK_MASK_PADDING_PX],
            fill="white",
        )
    return cleaned


def extract_structure(image_path, model, ocr_reader):
    """Returns {"smiles": str, "mean_confidence": float,
    "need_human_review": bool, "detected_missing_abbreviations": [str],
    "has_generic_substituent": bool, "rdkit_valid": bool,
    "too_many_fragments": bool, "method": str} -- same shape as
    decimer_extract.py's extract_structure (plus "method", the winning
    ensemble candidate name, kept for debugging/audit but not read by any
    downstream stage). model/ocr_reader are loaded once by the caller (see
    batch_molscribe_extract.py) and threaded through rather than reloaded
    per call."""
    original = Image.open(image_path).convert("RGB")
    cleaned = _strip_annotation_text(original)
    cleaned_bgr = cv2.cvtColor(np.array(cleaned), cv2.COLOR_RGB2BGR)

    from molscribe_ensemble_predict import predict_best
    result = predict_best(model, cleaned_bgr, ocr_reader=ocr_reader)
    smiles = result["smiles"] or ""
    mean_confidence = float(result.get("confidence") or 0.0)

    has_generic_substituent = _has_generic_substituent(smiles)
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    rdkit_valid = mol is not None
    too_many_fragments = _has_too_many_fragments(smiles, mol)
    detected_missing_abbreviations = []  # see module docstring

    return {
        "smiles": smiles,
        "mean_confidence": mean_confidence,
        "need_human_review": mean_confidence < MEAN_CONFIDENCE_THRESHOLD
        or has_generic_substituent
        or not rdkit_valid
        or too_many_fragments,
        "detected_missing_abbreviations": detected_missing_abbreviations,
        "has_generic_substituent": has_generic_substituent,
        "too_many_fragments": too_many_fragments,
        "rdkit_valid": rdkit_valid,
        "method": result.get("method"),
    }
