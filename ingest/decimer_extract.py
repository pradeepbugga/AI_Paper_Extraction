"""Runs DECIMER structure recognition on has_structures=1 figures, flagging
low-confidence results for human review instead of trying to predict
upstream (via CLIP vision or captions) whether a given image is a good
DECIMER target.

That upstream approach was tried first for is_3d specifically and abandoned:
CLIP vision is confounded by ordinary drawing conventions (metal complexes
are routinely drawn with rounded, ball-like atoms even in plain 2D
depictions -- visually similar to a genuine 3D render without being one),
and captions don't always exist for a given figure. DECIMER's own per-token
confidence is a more directly-grounded signal, since it's asking the actual
question that matters: did DECIMER succeed on THIS image, not what visual
category the image belongs to.

Confirmed directly before picking a threshold: a set of clean 2D structures
scored 0.92-0.998 mean confidence, while a 3D ball-and-stick metal-complex
render DECIMER completely misread (fabricated a structure with none of the
real functional groups: no N, no Cl, no metal) scored 0.787. A correctly-
recognized but unusually-colored 2D structure also dipped to 0.790 despite
being right -- confidence isn't perfectly correlated with correctness, but
that's an acceptable asymmetry: flagging a correct structure for human
review is a much cheaper mistake than silently accepting a wrong one.
"""

from DECIMER import predict_SMILES

MEAN_CONFIDENCE_THRESHOLD = 0.85


def extract_structure(image_path):
    """Returns {"smiles": str, "mean_confidence": float,
    "need_human_review": bool}."""
    smiles, tokens_with_confidence = predict_SMILES(image_path, confidence=True)
    confidences = [c for _, c in tokens_with_confidence]
    mean_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    return {
        "smiles": smiles,
        "mean_confidence": mean_confidence,
        "need_human_review": mean_confidence < MEAN_CONFIDENCE_THRESHOLD,
    }
