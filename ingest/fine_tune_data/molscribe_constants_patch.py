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
- `Ph2P`/`Cy2P`/`Me2P` (reversed-order diphenyl-/dicyclohexyl-/dimethyl-
  phosphino, as drawn on the left-hand side of a symmetric bisphosphine
  ligand so the group reads before the P): investigating 3 images the
  user flagged as "bad segmentation"
  (`miyaura_iron_2025/page3_fig4_seg1/4/5.png`, symmetric alkene-backbone
  bisphosphine ligands) found this is NOT a segmentation problem, and NOT
  a simple missing-dictionary-key problem either -- `PPh2`/`PCy2`/`PMe2`
  (forward order) were never literal dictionary keys at all; they already
  resolved via MolScribe's own generic condensed-formula fallback parser.
  The reversed-order side decodes as genuinely garbled tokens (`[35P2]`,
  `[We2P]`, `[3P]` -- not just reordered characters, actual decoder
  misreads), so no dictionary alias can match them directly. Added
  `Ph2P`/`Cy2P`/`Me2P` as explicit dictionary entries anyway, on the
  theory that `molscribe_ocr_correction.py`'s existing OCR-override pass
  (which crops near any bracket atom's coordinate and overrides with
  whatever valid dictionary abbreviation OCR reads, regardless of how
  garbled the original token was) can rescue these even though a plain
  alias couldn't. **Verify this actually works via `predict_best` with a
  real `ocr_reader`, not just the raw dictionary-lookup path, before
  assuming it's fixed.**
- `F` (plain fluorine) and `PPh2` (diphenylphosphino, forward order --
  already resolved via MolScribe's generic condensed-formula fallback
  parser when the raw decoder spells it correctly, but wasn't a literal
  `ABBREVIATIONS` key, so `molscribe_ocr_correction.py`'s OCR-override
  pass -- which only overrides into a literal dictionary key -- couldn't
  use either as a correction target): added as literal entries
  specifically to let the OCR-override pass rescue cases where the raw
  decoder token is garbled/wrong but an independent OCR read of the same
  crop gets the real label right. See
  [[project_ocr_correction_rerun_2026_08_23]] for whether this actually
  works on the 3 confirmed hard misses it was tried against.
- `iPrO`/`OiPr` (isopropoxy, both orderings -- neither existed) and
  `BPin` (capital-P casing variant of the already-fixed `Bpin` -- the
  decoder's own capitalization choice varies per image, and the
  dictionary lookup is case-sensitive): found during the same manual
  triage as `NMe2` above, on real corpus images
  (`nickelocene_2025/page15_fig0_seg0.png` for `iPrO`,
  `suzuki_iron_2024/page40_fig1_seg10.png` for `BPin`).
- `NMe2` (N,N-dimethylamino / dimethylamide nitrogen -- extremely common,
  e.g. terminal `-C(=O)NMe2` dimethylamide caps): found investigating a
  case the user flagged as a suspected "genuine visual-recognition miss"
  (`redox_neutral_2024/page10_fig0_seg18.png`) -- turned out not to be a
  vision problem at all. The decoder correctly proposed `[NMe2]` at the
  amide nitrogen, but only the singular `NMe` was in the dictionary, not
  `NMe2`, so the whole group silently collapsed to a bare `*` in the
  final SMILES with no trace of the nitrogen or its methyls. A reminder
  that "the wildcard fallback looks like a vision miss" is not evidence
  it IS one -- always check the raw token first.
- `NHPl` (N-phthalimide, Gabriel-synthesis-style protecting group),
  `SePh` (phenylselanyl, common radical-chemistry group), `OMOM`
  (methoxymethyl ether), `Bu3Sn` (tributylstannyl, radical-chemistry
  reagent), `PtBu2` (di-tert-butylphosphino, a very common cross-coupling
  ligand substituent): found via a user-directed manual triage of the
  full-corpus wildcard bucket (Aug 2026 follow-up session) -- all five
  confirmed via the same raw-token method to be confidently predicted
  (as `[NHPl]`, `[SePh]`, `[OMOM]`, `[Bu3Sn]`, `[PtBu2]`) but entirely
  absent from the dictionary, same mechanism as `Dipp`/`Bpin` above (not
  a both-orderings gap like `NTs`/`BocN` -- these simply didn't exist
  under any key).
- `BocN` (N-Boc-protected ring nitrogen, opposite token ordering from the
  `NBoc` key MolScribe's own dictionary already has): same both-orderings
  gap as `NTs`/`TsN` above, found via a full-corpus wildcard-sample scan
  (Aug 2026 follow-up session) -- recurred across at least 6 images in one
  paper alone (`redox_neutral_2024`, same 4-substituted-piperidine
  scaffold). Fixed by aliasing the existing `NBoc` entry under the `BocN`
  key too, rather than duplicating its SMARTS/SMILES.
- **`NBoc`'s own expansion SMILES was separately wrong** (a pre-existing
  MolScribe base-dictionary bug, unrelated to anything added by this
  patch): its SMARTS is `[NH0;D3]...` (zero explicit H, 3 connections --
  correct for a ring nitrogen, since the 2 ring bonds plus the Boc bond
  already account for all 3) but its `.smiles` expansion fragment was
  `[NH1]C(=O)OC(C)(C)C`, copy-pasted from the separate `NHBoc` entry
  (`[NH1;D2]...`, correct for an acyclic/terminal N-Boc that only has one
  other bond before the H). Applying the `[NH1]` fragment to a ring
  nitrogen produces 4 total bonds -- RDKit-invalid. Confirmed by fixing
  the `BocN`-wildcard gap above and finding the resulting structure still
  failed RDKit sanitization for this exact reason on real corpus images.
  Fixed by correcting `NBoc`'s `.smiles` to `[NH0]C(=O)OC(C)(C)C` in place
  (mutates the same object `BocN` is aliased to, so both keys get the fix
  together). Left `NHBoc` untouched -- its own SMARTS/SMILES pair is
  already internally consistent and correct for the acyclic case it's
  meant to cover.

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
    Substitution(['NHPl', 'Nphth'], '[NH0;D3]1C(=O)c2[cH][cH][cH][cH]c2C1=O',
                 "[N]1C(=O)c2ccccc2C1=O", 0.5),  # N-phthalimide, both spellings
    Substitution(['SePh'], '[SeH0;D2]c1[cH][cH][cH][cH][cH]1',
                 "[Se]c1ccccc1", 0.5),  # phenylselanyl
    Substitution(['OMOM'], '[OH0;D2]C[OH0;D2]C',
                 "[O]COC", 0.5),  # methoxymethyl ether
    Substitution(['Bu3Sn'], '[SnH0;D4](C[CH2][CH2][CH3])(C[CH2][CH2][CH3])C[CH2][CH2][CH3]',
                 "[Sn](CCCC)(CCCC)CCCC", 0.5),  # tributylstannyl
    Substitution(['PtBu2'], '[PH0;D3](C([CH3])([CH3])[CH3])C([CH3])([CH3])[CH3]',
                 "[P](C(C)(C)C)C(C)(C)C", 0.5),  # di-tert-butylphosphino
    Substitution(['NMe2'], '[NH0;D3](C)C',
                 "[N](C)C", 0.5),  # N,N-dimethylamino / dimethylamide N
    Substitution(['Ph2P'], '[PH0;D3](c1[cH][cH][cH][cH][cH]1)c1[cH][cH][cH][cH][cH]1',
                 "[P](c1ccccc1)c1ccccc1", 0.5),  # diphenylphosphino, reversed order
    Substitution(['Cy2P'], '[PH0;D3](C1CCCCC1)C1CCCCC1',
                 "[P](C1CCCCC1)C1CCCCC1", 0.5),  # dicyclohexylphosphino, reversed order
    Substitution(['Me2P'], '[PH0;D3](C)C',
                 "[P](C)C", 0.5),  # dimethylphosphino, reversed order
    Substitution(['iPrO', 'OiPr'], '[OH0;D2]C([CH3])[CH3]',
                 "[O]C(C)C", 0.5),  # isopropoxy, both orderings
    Substitution(['F'], '[F;D1]',
                 "F", 0.9),  # plain fluorine -- OCR-override target only
    Substitution(['PPh2'], '[PH0;D3](c1[cH][cH][cH][cH][cH]1)c1[cH][cH][cH][cH][cH]1',
                 "[P](c1ccccc1)c1ccccc1", 0.5),  # diphenylphosphino, forward order, literal key
]


def apply():
    """Call once, any time before running MolScribe inference."""
    for sub in _NEW_SUBSTITUTIONS:
        for abbrv in sub.abbrvs:
            ABBREVIATIONS[abbrv] = sub
    if 'NBoc' in ABBREVIATIONS:
        # NBoc's own SMARTS declares zero explicit H (ring-consistent),
        # but its shipped expansion SMILES wrongly copied NHBoc's
        # explicit-H fragment -- over-valent when applied to a ring N.
        ABBREVIATIONS['NBoc'].smiles = '[NH0]C(=O)OC(C)(C)C'
        if 'BocN' not in ABBREVIATIONS:
            ABBREVIATIONS['BocN'] = ABBREVIATIONS['NBoc']
    if 'Bpin' in ABBREVIATIONS and 'BPin' not in ABBREVIATIONS:
        ABBREVIATIONS['BPin'] = ABBREVIATIONS['Bpin']  # capital-P casing variant
    if 'NHAc' in ABBREVIATIONS and 'AcH' not in ABBREVIATIONS:
        # decoder emits the fused "AcHN" label (Ac drawn before N) as the
        # truncated bracket token [AcH], dropping the N entirely rather
        # than just reordering it -- alias to the same fragment as NHAc.
        ABBREVIATIONS['AcH'] = ABBREVIATIONS['NHAc']
