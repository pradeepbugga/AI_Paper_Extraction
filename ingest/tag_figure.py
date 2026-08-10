"""Fuses caption keyword rules with CLIP vision classification into the
final per-tag decision for each figure.

figure_caption_rules.py is high-precision but narrow -- it only fires on a
small fraction of figures, concentrated on exactly the cases pure vision
has demonstrated trouble with (is_computed's X-ray/DFT confusion,
is_spectrum's technique, has_grid_layout on oversized/dense images where
CLIP's fixed input resolution collapses detail). figure_classify.py's CLIP
embedding covers every figure but has those specific blind spots. So the
fusion rule is simple: caption wins when it fires, vision decides
otherwise -- including overriding vision's own internal tag-dependency
gate (e.g. has_grid_layout normally requires has_structures=1 from vision,
but a caption explicitly saying "substrate scope" is independent evidence
not subject to the table-vs-grid visual confusion that gate exists for).
"""

from figure_caption_rules import classify_from_caption
from figure_classify import classify as classify_vision


def tag_figure(image_path, caption_text, prototypes, embedding=None):
    """Returns {tag: (label, source)} for every tag figure_classify.py
    knows about. source is "caption_keyword" or "vision_margin=<value>" so
    the decision is traceable, not just a bare 0/1.
    """
    caption_results = classify_from_caption(caption_text)
    vision_results = classify_vision(image_path, prototypes, embedding=embedding)

    final = {}
    for tag, (label, pos_sim, neg_sim) in vision_results.items():
        if tag in caption_results:
            final[tag] = (caption_results[tag], "caption_keyword")
        else:
            final[tag] = (label, f"vision_margin={pos_sim - neg_sim:+.4f}")

    return final
