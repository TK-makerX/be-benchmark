"""Sequence parsing and lightweight input-quality checks."""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional


DNA_PATTERN = re.compile(r"^[ACGTN]+$")


@dataclass
class SequenceAssessment:
    raw_sequence: str
    sequence: str
    valid_dna: bool
    invalid_characters: List[str]
    length: int
    inferred_input_type: str
    pam: Optional[str]
    is_ngg_like: Optional[bool]
    abe_window_bases: str
    cbe_window_bases: str
    abe_candidate_positions_1based: List[int]
    cbe_candidate_positions_1based: List[int]
    warnings: List[str]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def normalize_sequence(sequence: str) -> str:
    """Normalize user sequence without changing biological meaning."""
    return re.sub(r"\s+", "", sequence or "").upper().replace("U", "T")


def infer_input_type(length: int) -> str:
    if length == 20:
        return "20nt_protospacer_only"
    if length == 23:
        return "20nt_protospacer_plus_3nt_pam"
    if length == 30:
        return "30nt_model_window"
    if length == 40:
        return "40bp_model_window"
    if length == 50:
        return "50nt_behive_window"
    if length > 50:
        return "long_sequence_needs_candidate_window_generation"
    return "nonstandard_short_or_partial_sequence"


def _protospacer_for_window(sequence: str) -> str:
    """Return a best-effort 20nt protospacer-like region for input QC.

    This is not a model adapter. It only supports UI warnings and routing.
    """
    if len(sequence) == 30:
        return sequence[4:24]
    if len(sequence) == 40:
        return sequence[10:30]
    if len(sequence) >= 23 and sequence[-2:] == "GG":
        return sequence[:20]
    if len(sequence) >= 20:
        return sequence[:20]
    return sequence


def assess_sequence(sequence: str) -> SequenceAssessment:
    raw = sequence or ""
    seq = normalize_sequence(raw)
    invalid = sorted(set(ch for ch in seq if ch not in "ACGTN"))
    valid = bool(seq) and DNA_PATTERN.match(seq) is not None
    length = len(seq)
    input_type = infer_input_type(length)

    pam = None
    is_ngg = None
    if length == 23:
        pam = seq[20:23] if length == 23 else seq[-3:]
        is_ngg = len(pam) == 3 and pam[1:] == "GG"
    elif length == 30:
        # Common DeepBE/DeepBaseEditor convention:
        # 4bp flank + 20bp protospacer + 3bp PAM + 3bp flank.
        pam = seq[24:27]
        is_ngg = len(pam) == 3 and pam[1:] == "GG"
    elif length == 40:
        # BE_Endo BE_Seq convention in the local adapter: NGG at 1-based 31-33.
        pam = seq[30:33]
        is_ngg = len(pam) == 3 and pam[1:] == "GG"
    elif length > 30:
        pam = seq[-3:]
        is_ngg = len(pam) == 3 and pam[1:] == "GG"

    protospacer = _protospacer_for_window(seq)
    # Common SpCas9 BE activity windows are often described around positions 4-8.
    window = protospacer[3:8] if len(protospacer) >= 8 else ""
    abe_pos = [i + 1 for i, base in enumerate(protospacer[:20]) if 4 <= i + 1 <= 8 and base == "A"]
    cbe_pos = [i + 1 for i, base in enumerate(protospacer[:20]) if 4 <= i + 1 <= 8 and base == "C"]

    warnings: List[str] = []
    if not seq:
        warnings.append("No sequence was provided.")
    if invalid:
        warnings.append(f"Invalid DNA characters detected: {', '.join(invalid)}.")
    if "N" in seq:
        warnings.append("Ambiguous N bases are present; many model adapters require concrete A/C/G/T sequence.")
    if input_type.startswith("nonstandard"):
        warnings.append("Sequence length is not a common model input length; routing is possible but direct model execution may need preprocessing.")
    if is_ngg is False:
        warnings.append("Detected PAM is not NGG-like; common-core T1/T4 evidence is strongest for NGG/SpCas9-like panels.")
    if not abe_pos:
        warnings.append("No A base was found in the common ABE editing window positions 4-8 of the protospacer-like region.")
    if not cbe_pos:
        warnings.append("No C base was found in the common CBE editing window positions 4-8 of the protospacer-like region.")

    return SequenceAssessment(
        raw_sequence=raw,
        sequence=seq,
        valid_dna=valid,
        invalid_characters=invalid,
        length=length,
        inferred_input_type=input_type,
        pam=pam,
        is_ngg_like=is_ngg,
        abe_window_bases=window,
        cbe_window_bases=window,
        abe_candidate_positions_1based=abe_pos,
        cbe_candidate_positions_1based=cbe_pos,
        warnings=warnings,
    )


def infer_editor_family(editor: str) -> str:
    value = (editor or "").strip().lower()
    if any(key in value for key in ["abe", "adenine", "a-to-g", "a2g", "a>g"]):
        return "ABE"
    if any(key in value for key in ["cbe", "cytosine", "c-to-t", "c2t", "c>t", "be3", "be4"]):
        return "CBE"
    if "prime" in value or value.startswith("pe"):
        return "PrimeEditing"
    return "Unknown"
