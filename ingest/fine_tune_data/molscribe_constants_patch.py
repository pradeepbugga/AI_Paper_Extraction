"""Adds abbreviation-dictionary entries MolScribe's own `constants.py`
(`ABBREVIATIONS`/`Substitution` table) is missing, so bracket-atom labels
the decoder already recognizes correctly can actually be expanded instead
of falling back to a bare `*` wildcard.

Root cause of the gap (confirmed directly, Aug 2026 session, via
`model.predict_image_file(..., return_atoms_bonds=True)` raw-token
inspection before any dictionary lookup happens):

- `DMT`/`ODMT`/`Trt`/`OTrt` (dimethoxytrityl/trityl protecting groups):
  0% recovery off the shelf -- MolScribe had never seen these labels.
- `NTs`/`TsN`/`OTs` (fused N-tosyl / O-tosylate labels, as opposed to a
  bond-drawn-to-a-separate-"Ts"-label depiction): the decoder correctly
  predicted the fused bracket token (`[TsN]`, confidence ~0.91, same as
  every other atom in the image) but the dictionary only had a bare `'Ts'`
  key, not the fused literal string, so direct lookup failed and the
  generic condensed-formula fallback parser also failed to reassemble it,
  producing a bare `*` wildcard in the final SMILES.
- `Dipp` (2,6-diisopropylphenyl, an extremely common N-aryl substituent on
  NHC carbene ligands): same mechanism again, found via a full-corpus scan
  of remaining RDKit-invalid results (Aug 2026 follow-up session) --
  `[Dipp]` predicted at 0.88-0.90 confidence in multiple independent real
  images, always falling back to `*` because the dictionary had `'Mes'`
  (its close cousin, mesityl) but not `'Dipp'`.
- `Bpin` (pinacol boronate ester, ubiquitous in cross-coupling literature):
  found the same way -- `[Bpin]` predicted with high confidence but absent
  from the dictionary.

This is a monkey-patch (mutates the `ABBREVIATIONS` dict object in place,
does not touch site-packages) specifically because a hand-edit to
`molscribe/constants.py` in `site-packages` was made once already this
project and then silently lost -- a later `pip install`/venv rebuild in
the same session overwrote it with the stock file, with no backup and no
trace anywhere on the pod. Importing this module and calling `apply()`
once (any time before running inference; import order relative to
`molscribe.chemistry` does not matter, since the mutation is visible
through every existing reference to the same dict object) reproduces the
fix deterministically on any fresh pod/venv instead of relying on memory
of a manual edit.
"""
from molscribe.constants import ABBREVIATIONS, Substitution

_NEW_SUBSTITUTIONS = [
    Substitution(['OTs'], '[OH0;D2]S(=O)(=O)c1[cH1][cH1][cH0]([CH3])[cH1][cH1]1',
                 "[O]S(=O)(=O)c1ccc(C)cc1", 0.6),  # Tosylate
    Substitution(['NTs', 'TsN'], '[NH0;D3]S(=O)(=O)c1[cH1][cH1][cH0]([CH3])[cH1][cH1]1',
                 "[NH0]S(=O)(=O)c1ccc(C)cc1", 0.6),  # N-Tosyl
    Substitution(['Trt'], '[CH0;D4](c1[cH][cH][cH][cH][cH]1)(c1[cH][cH][cH][cH][cH]1)c1[cH][cH][cH][cH][cH]1',
                 "[C](c1ccccc1)(c1ccccc1)c1ccccc1", 0.5),  # Trityl
    Substitution(['OTrt'], '[OH0;D2]C(c1[cH][cH][cH][cH][cH]1)(c1[cH][cH][cH][cH][cH]1)c1[cH][cH][cH][cH][cH]1',
                 "[O]C(c1ccccc1)(c1ccccc1)c1ccccc1", 0.6),  # O-Trityl
    Substitution(['DMT'], '[CH0;D4](c1[cH][cH][cH0](OC)[cH][cH]1)(c1[cH][cH][cH0](OC)[cH][cH]1)c1[cH][cH][cH][cH][cH]1',
                 "[C](c1ccc(OC)cc1)(c1ccc(OC)cc1)c1ccccc1", 0.5),  # Dimethoxytrityl
    Substitution(['ODMT'], '[OH0;D2]C(c1[cH][cH][cH0](OC)[cH][cH]1)(c1[cH][cH][cH0](OC)[cH][cH]1)c1[cH][cH][cH][cH][cH]1',
                 "[O]C(c1ccc(OC)cc1)(c1ccc(OC)cc1)c1ccccc1", 0.6),  # O-Dimethoxytrityl
    Substitution(['Dipp'], '[cH0]1c([CH1]([CH3])[CH3])[cH1][cH1][cH1]c1[CH1]([CH3])[CH3]',
                 "[c]1c(C(C)C)ccc(C(C)C)c1", 0.5),  # 2,6-diisopropylphenyl
    Substitution(['Bpin'], '[BH0;D2]1OC([CH3])([CH3])C([CH3])([CH3])O1',
                 "[B]1OC(C)(C)C(C)(C)O1", 0.5),  # pinacol boronate ester
]


def apply():
    """Call once, any time before running MolScribe inference."""
    for sub in _NEW_SUBSTITUTIONS:
        for abbrv in sub.abbrvs:
            ABBREVIATIONS[abbrv] = sub
