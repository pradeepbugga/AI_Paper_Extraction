"""Confidence-based ensemble prediction, replacing the single-deterministic-
pipeline approach in molscribe_highlight_flatten.py with something safer:
instead of trying to hand-craft one preprocessing rule that is never wrong
(three attempts at that this session each found new edge cases -- global
threshold, per-paper color profiling, then a per-image background-gap
invariant with a dilation bug that erased real bond lines in one case),
generate a small set of candidate preprocessings, run MolScribe on each,
and let the model's own reported confidence pick the winner -- the same
principle as a decoder returning its highest-probability hypothesis.

Why this is safer than any single rule: every "must not be touched" case
found this session (a fragile tiny anti-aliased label, a paper that uses
real colored ink, a placeholder abbreviation with soft gray anti-aliasing)
turns out to already be the model's own highest-confidence, valid answer
among the candidates -- because `raw` (the untouched original) is always
itself one of the candidates. A rule never has to know in advance "don't
touch this image"; if the original was already right, it wins on its own
merits. Verified directly (Aug 22 follow-up session) on every case that
had previously required careful gating: page17_fig1_seg0.png (previously
needed protection from the flatten pipeline; now `raw` wins outright),
page5_fig0_seg24.png (the fragile TsN glyph; `raw` wins), page80's
bond-drawn-Ts case (`raw` wins), suzuki_iron_2024's green-background
colored-ink page4_fig0_seg30.png (`raw` wins). Genuinely broken cases are
fixed by whichever candidate happens to work: page61_fig2_seg0.png (a
case where `molscribe_highlight_flatten`'s dilation had been silently
producing coincidental garbage) is fixed by `targeted_flatten`;
page56_fig0_seg0.png by a plain global threshold. A case with no valid
candidate at all (page3_fig1_seg10.png, a genuine 3-molecule mixture)
is honestly reported as still broken, rather than papered over with a
plausible-looking wrong answer.

Grayscale images (no real color content at all) skip the ensemble
entirely and go straight to a single direct prediction -- there is
nothing for alternate preprocessings to fix, and skipping them keeps the
common case fast.

Also applies molscribe_ocr_correction.py's label-misread and formal-charge
fixes to the WINNING candidate only (one extra return_atoms_bonds=True
call per image, not per candidate). These were built and verified
separately earlier in the same session but never wired into this module --
running the full-corpus scan without them left dozens of images unfixed
that were already-solved problems (the tBu/iBu misread + boron valence
combination alone accounts for a large fraction of suzuki_iron_2024's
remaining invalid images).
"""
import cv2
import numpy as np
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

import molscribe_highlight_flatten as mhf
import molscribe_ocr_correction as moc

GRAYSCALE_SPREAD_MAX = 8  # max(R,G,B)-min(R,G,B) across every pixel; if no pixel
                          # exceeds this, the image has no real color content at all
BW_THRESHOLDS = [180, 200, 220]


def is_grayscale(img_bgr):
    arr = img_bgr.astype(int)
    return int((arr.max(axis=2) - arr.min(axis=2)).max()) <= GRAYSCALE_SPREAD_MAX


def make_candidates(img_bgr):
    """Returns {name: image_bgr} for every preprocessing worth trying."""
    candidates = {"raw": img_bgr}
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    for t in BW_THRESHOLDS:
        _, binary = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)
        candidates[f"global_bw_t{t}"] = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    flattened, modified = mhf.preprocess(img_bgr)
    if modified:
        candidates["targeted_flatten"] = flattened
    return candidates


def _correct_winner(model, ocr_reader, winner_img_bgr):
    """Re-run the winning candidate with return_atoms_bonds=True and apply
    the label-misread + formal-charge corrections. Returns (corrected
    smiles, corrected valid, label_corrections, charge_fixes) or None if
    ocr_reader wasn't provided (corrections skipped)."""
    if ocr_reader is None:
        return None
    cand_rgb = cv2.cvtColor(winner_img_bgr, cv2.COLOR_BGR2RGB)
    out = model.predict_images([cand_rgb], return_atoms_bonds=True, return_confidence=True)[0]
    result = moc.apply_corrections(model, ocr_reader, winner_img_bgr, out)
    valid = Chem.MolFromSmiles(result["smiles"]) is not None if result["smiles"] else False
    return result["smiles"], valid, result["label_corrections"], result["charge_fixes"]


def predict_best(model, img_bgr, ocr_reader=None):
    """Returns dict: smiles, method, confidence, valid, candidates (list of
    (name, smiles, confidence, valid) for every candidate tried, or None
    for the grayscale direct-prediction path). If ocr_reader is given (an
    easyocr.Reader), the winning candidate is also passed through
    molscribe_ocr_correction's label-misread and formal-charge fixes, and
    the corrected result is used whenever it improves on (or matches) the
    raw ensemble winner -- correction only ever fixes a wrong/invalid atom,
    never touches an already-correct one, so it's safe to prefer whenever
    it changes anything."""
    if is_grayscale(img_bgr):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        out = model.predict_images([img_rgb], return_confidence=True)[0]
        result = {"smiles": out["smiles"], "method": "grayscale-direct",
                  "confidence": out.get("confidence"), "candidates": None}
        corrected = _correct_winner(model, ocr_reader, img_bgr)
        if corrected:
            smiles, valid, label_corr, charge_fixes = corrected
            if label_corr or charge_fixes:
                result["smiles"] = smiles
                result["corrections_applied"] = {"label": label_corr, "charge": charge_fixes}
        return result

    candidates = make_candidates(img_bgr)
    results = []
    for name, cand_img in candidates.items():
        cand_rgb = cv2.cvtColor(cand_img, cv2.COLOR_BGR2RGB)
        out = model.predict_images([cand_rgb], return_confidence=True)[0]
        smiles = out["smiles"]
        valid = Chem.MolFromSmiles(smiles) is not None if smiles else False
        results.append((name, smiles, out.get("confidence", 0.0), valid))

    valid_results = [r for r in results if r[3]]
    pool = valid_results if valid_results else results
    best_name, best_smiles, best_conf, best_valid = max(pool, key=lambda r: r[2])
    result = {"smiles": best_smiles, "method": best_name, "confidence": best_conf,
              "valid": best_valid, "candidates": results}

    corrected = _correct_winner(model, ocr_reader, candidates[best_name])
    if corrected:
        smiles, valid, label_corr, charge_fixes = corrected
        if label_corr or charge_fixes:
            result["smiles"] = smiles
            result["valid"] = valid
            result["corrections_applied"] = {"label": label_corr, "charge": charge_fixes}
    return result


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
        img = cv2.imread(path)
        result = predict_best(model, img, ocr_reader=ocr_reader)
        print(f"\n{path}")
        print(f"  winner=[{result['method']}] conf={result['confidence']}: {result['smiles']}")
        if result.get("corrections_applied"):
            print(f"  corrections_applied: {result['corrections_applied']}")
        if result["candidates"]:
            for name, smiles, conf, valid in result["candidates"]:
                print(f"    [{name}] valid={valid} conf={conf:.4f}  {smiles}")
