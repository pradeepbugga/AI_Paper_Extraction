"""Post-hoc correction pass for two real MolScribe failure modes found by
running the pretrained model (+ molscribe_constants_patch) across this
project's full real-paper corpus (2,145 segments, Aug 2026 session):

1. Fragile-glyph label misreads. MolScribe's decoder can confidently
   (~0.95+) propose the *wrong* bracket-abbreviation token when the
   distinguishing glyph is small/italic/superscript -- confirmed directly
   on a real, repeated case: a tert-butyl-on-boron reagent (`tBu`, drawn
   with a tiny italic superscript "t") was misread as `[iBu]` (isobutyl)
   in 10/10 occurrences across `suzuki_iron_2024`, always at high
   confidence, i.e. not a low-confidence/flaggable case on its own.
   EasyOCR resolves the same glyph correctly at 0.95-0.99 confidence with
   nothing more than a 4x upscale. This is NOT a dictionary-expansion gap
   (see molscribe_constants_patch.py for that category) -- the decoder's
   own token choice is wrong before any dictionary lookup happens, so no
   dictionary entry can fix it. Fix: independently OCR a crop around each
   predicted bracket-atom's coordinate, and override the label when OCR
   strongly disagrees with a *different* known dictionary abbreviation.

2. Under-specified formal charges. Once (1) is fixed (and separately, once
   molscribe_constants_patch.py's Dipp/Bpin dictionary entries expand
   correctly), some atoms are still left over-valent because MolScribe's
   graph decoder assigns no formal charge to them:
   - Boron: tetracoordinate "ate"-complex boronates (aryl + alkyl + 2
     ring-O on one boron, as in Ar-B(pin)(tBu) reagents, or a
     boron-boron-bonded diboron reagent once `[Bpin]` expands) fail RDKit
     sanitization otherwise.
   - Nitrogen: aromatic imidazolium/amidinium ring nitrogens in NHC-ligand
     salts (found via the same full-corpus scan that turned up the
     Dipp/Bpin dictionary gap) end up with 3 substituents plus ring double
     bonds -- over-valent for neutral trivalent nitrogen.
   Fix: any plain (non-abbreviated) `B` or `N` atom whose total bond order
   exceeds that element's neutral valence (3 for both) gets a formal
   charge bumped on -- `[B-]` (standard borate-anion valence) or `[N+]`
   (standard ammonium/iminium-cation valence) respectively. Narrowly
   scoped to these two elements since they're the only cases confirmed in
   this corpus; not a general over-valence-fixer for every element.

Both passes are deterministic, run after MolScribe's own decode step, and
touch only atoms/molecules that would otherwise be wrong or unparseable --
nothing about a correctly-decoded atom is changed.
"""
import re
from pathlib import Path

import cv2
from molscribe.constants import ABBREVIATIONS
from molscribe.chemistry import convert_graph_to_smiles

BOND_TYPES = ["", "single", "double", "triple", "aromatic", "solid wedge", "dashed wedge"]
BOND_ORDER = {0: 0.0, 1: 1.0, 2: 2.0, 3: 3.0, 4: 1.5, 5: 1.0, 6: 1.0}
NEUTRAL_VALENCE = {"B": 3.0, "N": 3.0}
CHARGED_SYMBOL = {"B": "[B-]", "N": "[N+]"}

# Fraction of max(image width, image height) to crop on each side of a
# predicted atom's coordinate before handing it to OCR. Wide enough to
# reliably contain a multi-character superscript-prefixed label like
# "tBu" without needing per-label bounding boxes from MolScribe itself
# (which the interface does not expose).
OCR_CROP_HALF_FRACTION = 0.22
OCR_UPSCALE = 4
OCR_MIN_CONFIDENCE = 0.5


def _clean_ocr_text(text):
    return re.sub(r'[^A-Za-z0-9]', '', text)


def predict_with_corrections(model, ocr_reader, image_path):
    """Run MolScribe on image_path, then apply the label-misread and
    boron-valence corrections described above.

    Returns a dict: {"smiles": corrected SMILES, "raw_smiles": MolScribe's
    original output, "label_corrections": [(atom_idx, old, new, ocr_conf)],
    "charge_fixes": [(atom_idx, total_bond_order)]}.
    """
    image_path = str(image_path)
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W = img_bgr.shape[:2]

    out = model.predict_image_file(image_path, return_atoms_bonds=True, return_confidence=True)
    atoms = out["atoms"]
    n = len(atoms)
    coords = [[a["x"], a["y"]] for a in atoms]
    symbols = [a["atom_symbol"] for a in atoms]
    edges = [[0] * n for _ in range(n)]
    for b in out["bonds"]:
        i, j = b["endpoint_atoms"]
        t = BOND_TYPES.index(b["bond_type"])
        edges[i][j] = t
        edges[j][i] = t

    # --- pass 1: OCR cross-check on bracket-abbreviation labels ---
    label_corrections = []
    big = cv2.resize(img_bgr, (W * OCR_UPSCALE, H * OCR_UPSCALE), interpolation=cv2.INTER_LANCZOS4)
    half = int(OCR_CROP_HALF_FRACTION * max(W, H) * OCR_UPSCALE)

    for idx, sym in enumerate(symbols):
        if not (sym.startswith("[") and sym.endswith("]")):
            continue
        label = sym[1:-1]
        x_px, y_px = coords[idx][0] * W * OCR_UPSCALE, coords[idx][1] * H * OCR_UPSCALE
        x0, y0 = max(0, int(x_px - half)), max(0, int(y_px - half))
        x1, y1 = min(big.shape[1], int(x_px + half)), min(big.shape[0], int(y_px + half))
        crop = big[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        for _, text, conf in ocr_reader.readtext(crop):
            ct = _clean_ocr_text(text)
            if ct and ct != label and ct in ABBREVIATIONS and conf > OCR_MIN_CONFIDENCE:
                label_corrections.append((idx, label, ct, conf))
                symbols[idx] = f"[{ct}]"
                break

    # --- pass 2: formal-charge fix (boron, nitrogen) ---
    charge_fixes = []
    for idx, sym in enumerate(symbols):
        if sym not in NEUTRAL_VALENCE:
            continue
        total = sum(BOND_ORDER[e] for e in edges[idx])
        if total > NEUTRAL_VALENCE[sym]:
            symbols[idx] = CHARGED_SYMBOL[sym]
            charge_fixes.append((idx, sym, total))

    smiles_list, molblock_list, _ = convert_graph_to_smiles([coords], [symbols], [edges], images=[img_rgb])

    return {
        "smiles": smiles_list[0],
        "raw_smiles": out["smiles"],
        "label_corrections": label_corrections,
        "charge_fixes": charge_fixes,
    }


if __name__ == "__main__":
    import sys
    import torch
    import easyocr
    from molscribe import MolScribe
    from huggingface_hub import hf_hub_download
    import molscribe_constants_patch
    molscribe_constants_patch.apply()

    ckpt = hf_hub_download("yujieq/MolScribe", "swin_base_char_aux_1m.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MolScribe(ckpt, device=device)
    ocr_reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())

    for path in sys.argv[1:]:
        result = predict_with_corrections(model, ocr_reader, path)
        print(f"\n{path}")
        print(f"  raw:       {result['raw_smiles']}")
        print(f"  corrected: {result['smiles']}")
        print(f"  label_corrections: {result['label_corrections']}")
        print(f"  charge_fixes: {result['charge_fixes']}")
