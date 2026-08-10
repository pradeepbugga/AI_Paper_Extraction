"""Stage 3: multi-label figure tagging via CLIP image embeddings + nearest-prototype.

Each tag (has_structures, has_grid_layout, ...) gets its own positive/negative
prototype built from data/figure_examples/labels.csv. A new figure crop is
tagged 1 for a given tag if its embedding is closer (cosine similarity) to
that tag's positive prototype than to its negative prototype. Tags are
independent binary decisions, not a joint argmax -- a figure can fire on
several tags at once, per the multi-label design.
"""

import csv
from pathlib import Path

import open_clip
import torch
from PIL import Image

MODEL_NAME = "ViT-B-32-quickgelu"
PRETRAINED = "openai"
EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "data" / "figure_examples"
LABELS_CSV = EXAMPLES_DIR / "labels.csv"
IMAGES_DIR = EXAMPLES_DIR / "images"

_device = "cuda" if torch.cuda.is_available() else "cpu"
_model = None
_preprocess = None


def _load_model():
    global _model, _preprocess
    if _model is None:
        _model, _, _preprocess = open_clip.create_model_and_transforms(
            MODEL_NAME, pretrained=PRETRAINED
        )
        _model.eval().to(_device)
    return _model, _preprocess


def embed_image(image_path):
    model, preprocess = _load_model()
    image = Image.open(image_path).convert("RGB")
    tensor = preprocess(image).unsqueeze(0).to(_device)
    with torch.no_grad():
        features = model.encode_image(tensor)
        features /= features.norm(dim=-1, keepdim=True)
    return features.squeeze(0)


def load_labels(labels_csv=LABELS_CSV):
    """Returns (tag_names, [(image_path, {tag: 0/1})])."""
    with open(labels_csv) as f:
        reader = csv.reader(f)
        header = [h.strip() for h in next(reader)]
        tag_names = header[1:]
        rows = []
        for row in reader:
            if not row or not row[0].strip():
                continue
            image_path = row[0].strip()
            labels = {
                tag: int(val.strip())
                for tag, val in zip(tag_names, row[1:])
            }
            rows.append((image_path, labels))
    return tag_names, rows


def _centroid(vecs):
    """Mean of unit vectors, re-normalized -- a plain mean shrinks in magnitude
    proportional to how spread out the vectors are, which would bias raw
    dot-product comparisons toward whichever group happens to be tighter
    (usually the smaller, more homogeneous positive set). Re-normalizing keeps
    every prototype on the unit sphere so pos/neg comparisons are on equal
    footing.
    """
    if not vecs:
        return None
    centroid = torch.stack(vecs).mean(dim=0)
    return centroid / centroid.norm()


def build_prototypes(images_dir=IMAGES_DIR, labels_csv=LABELS_CSV, exclude=None):
    """Builds {tag: {"pos": centroid, "neg": centroid}} from labeled examples.

    exclude: optional image_path to leave out (for leave-one-out validation).
    """
    tag_names, rows = load_labels(labels_csv)
    embeddings = {}
    for image_path, _ in rows:
        if image_path == exclude:
            continue
        embeddings[image_path] = embed_image(images_dir / image_path)

    prototypes = {}
    for tag in tag_names:
        pos_vecs = [embeddings[p] for p, l in rows if p in embeddings and l[tag] == 1]
        neg_vecs = [embeddings[p] for p, l in rows if p in embeddings and l[tag] == 0]
        prototypes[tag] = {
            "pos": _centroid(pos_vecs),
            "neg": _centroid(neg_vecs),
        }
    return prototypes, embeddings


# A grid only means anything as "structures arranged in a grid" -- CLIP's
# holistic embedding otherwise conflates it with any tabular/multi-panel
# layout (data tables, multi-photo strips), so this tag is meaningless
# without has_structures also firing. Encoded as a hard gate rather than
# left to the embedding to learn from too few examples.
TAG_DEPENDENCIES = {"has_grid_layout": "has_structures"}

# Tags answerable from any sub-region of an image ("does this contain a
# structure/spectrum somewhere") -- tiling a large image and OR-ing these
# across tiles recovers detail CLIP's fixed input resolution would
# otherwise collapse (bond lines in a dense structure grid, peak lines in a
# large spectrum trace -- both erased by downsampling before the model ever
# sees them). A tag only belongs here if it clears BOTH conditions, checked
# against real oversized figures, not assumed:
#   1. Its signal is fine-grained relative to image size (a *scale*
#      problem tiling can actually fix) -- not a global/compositional
#      property. has_grid_layout/has_plotted_data fail this: tiling a real
#      R1xR2 matrix made every tile lose the row/column structure that
#      makes it a grid at all, since "gridness" only exists at whole-image
#      scale. is_photo fails this too, despite a perfect 100% whole-image
#      leave-one-out score -- its signal (lighting, texture, blur) is
#      coarse/holistic, so there was nothing for tiling to recover, and it
#      produced two confirmed false positives on large NMR spectra instead
#      (a sparse plot tile apparently resembles a photo's texture locally
#      even though the whole image clearly doesn't).
#   2. Its whole-image decision boundary is otherwise sound -- tiling
#      recovers lost detail, it doesn't fix a confused boundary. is_computed
#      fails this: it also depends on fine detail (ΔG values, distance
#      labels) but its boundary was already noisy for an unrelated reason
#      (X-ray/DFT visual ambiguity), and OR-across-9-tiles just gave that
#      existing weakness 9 chances to misfire instead of 1 -- flipped a
#      correct is_computed=0 to a false 1 on page4_fig0.png. Its real fix
#      is the caption keyword rule (figure_caption_rules.py), not vision.
LOCAL_TAGS = {"has_structures", "is_spectrum"}
TILE_SIZE_THRESHOLD_PX = 900
TILE_GRID = (3, 3)


def _needs_tiling(image_path):
    with Image.open(image_path) as image:
        return max(image.size) >= TILE_SIZE_THRESHOLD_PX


def _tile_embeddings(image_path, grid=TILE_GRID):
    model, preprocess = _load_model()
    image = Image.open(image_path).convert("RGB")
    rows, cols = grid
    w, h = image.size
    tile_w, tile_h = w // cols, h // rows
    embeddings = []
    for r in range(rows):
        for c in range(cols):
            x0, y0 = c * tile_w, r * tile_h
            x1 = w if c == cols - 1 else x0 + tile_w
            y1 = h if r == rows - 1 else y0 + tile_h
            tensor = preprocess(image.crop((x0, y0, x1, y1))).unsqueeze(0).to(_device)
            with torch.no_grad():
                features = model.encode_image(tensor)
                features /= features.norm(dim=-1, keepdim=True)
            embeddings.append(features.squeeze(0))
    return embeddings


def classify(image_path, prototypes, embedding=None):
    """Returns {tag: (0/1, pos_similarity, neg_similarity)}.

    Tiling only runs when embedding isn't already supplied -- i.e. on a
    fresh classification call with a real image_path, not the cached-
    embedding path leave_one_out_eval uses to validate the reference set
    itself (which is deliberately kept untiled, see LOCAL_TAGS above).
    """
    tile_this = embedding is None and _needs_tiling(image_path)
    if embedding is None:
        embedding = embed_image(image_path)

    results = {}
    for tag, proto in prototypes.items():
        pos, neg = proto["pos"], proto["neg"]
        pos_sim = torch.dot(embedding, pos).item() if pos is not None else -1.0
        neg_sim = torch.dot(embedding, neg).item() if neg is not None else -1.0
        label = 1 if pos_sim > neg_sim else 0
        results[tag] = (label, pos_sim, neg_sim)

    if tile_this:
        tile_embeds = _tile_embeddings(image_path)
        for tag in LOCAL_TAGS:
            if tag not in prototypes or results[tag][0] == 1:
                continue
            pos, neg = prototypes[tag]["pos"], prototypes[tag]["neg"]
            for tile_emb in tile_embeds:
                pos_sim = torch.dot(tile_emb, pos).item() if pos is not None else -1.0
                neg_sim = torch.dot(tile_emb, neg).item() if neg is not None else -1.0
                if pos_sim > neg_sim:
                    results[tag] = (1, pos_sim, neg_sim)
                    break

    for tag, requires in TAG_DEPENDENCIES.items():
        if tag in results and requires in results and results[requires][0] == 0:
            label, pos_sim, neg_sim = results[tag]
            results[tag] = (0, pos_sim, neg_sim)

    return results


def leave_one_out_eval(images_dir=IMAGES_DIR, labels_csv=LABELS_CSV):
    """Sanity check: for each labeled image, rebuild prototypes without it,
    classify it, and compare to its true label. Prints per-tag accuracy.
    """
    tag_names, rows = load_labels(labels_csv)
    correct = {tag: 0 for tag in tag_names}
    total = {tag: 0 for tag in tag_names}

    # Pre-embed everything once; prototypes are recomputed per fold from
    # this shared cache so we don't re-run CLIP on every fold.
    all_embeddings = {p: embed_image(images_dir / p) for p, _ in rows}

    for held_out_path, true_labels in rows:
        prototypes = {}
        for tag in tag_names:
            pos_vecs = [
                all_embeddings[p]
                for p, l in rows
                if p != held_out_path and l[tag] == 1
            ]
            neg_vecs = [
                all_embeddings[p]
                for p, l in rows
                if p != held_out_path and l[tag] == 0
            ]
            prototypes[tag] = {"pos": _centroid(pos_vecs), "neg": _centroid(neg_vecs)}

        predictions = classify(held_out_path, prototypes, embedding=all_embeddings[held_out_path])
        for tag in tag_names:
            total[tag] += 1
            if predictions[tag][0] == true_labels[tag]:
                correct[tag] += 1

    print(f"Leave-one-out accuracy over {len(rows)} examples:")
    for tag in tag_names:
        acc = correct[tag] / total[tag]
        print(f"  {tag:<20} {acc:.1%} ({correct[tag]}/{total[tag]})")


if __name__ == "__main__":
    leave_one_out_eval()
