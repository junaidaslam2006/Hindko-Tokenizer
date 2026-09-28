# -*- coding: utf-8 -*-
"""Preconditions of the equivalence argument:
 (1) every character occurring in a NORMAL piece has a single-character NORMAL piece, so SentencePiece's
     <unk> node is only ever FORCED (a character no piece covers) and its score never decides a comparison;
 (2) every character of every non-NORMAL piece string (kept in the HF Unigram vocab at LOW_SCORE) has a
     single-character NORMAL piece, so a LOW_SCORE entry can never win in HF;
 (3) no NORMAL piece contains U+2581 except as its first character (no piece spans a word boundary)."""
import json, os, collections
from common import *

m, T = load_proto(OLD_MODEL)
normal = {p.piece: i for i, p in enumerate(m.pieces) if p.type == T.NORMAL}
single = {s for s in normal if len(s) == 1}
chars_in_normal = set("".join(normal))
no_single = sorted(c for c in chars_in_normal if c not in single)
nonnormal = [(i, p.piece, T.Type.Name(p.type)) for i, p in enumerate(m.pieces) if p.type != T.NORMAL]
bad_low = sorted({c for _, s, _ in nonnormal for c in s if c not in single and c != "\n"})
out = {"normal_pieces": len(normal), "single_char_normal": len(single),
       "chars_in_normal_without_single_piece": [(c, "U+%04X" % ord(c)) for c in no_single],
       "chars_of_nonnormal_strings_without_single_piece": [(c, "U+%04X" % ord(c)) for c in bad_low],
       "max_len_nonnormal_string": max(len(s) for _, s, _ in nonnormal),
       "normal_with_inner_mark": sum(1 for s in normal if SP_MARK in s[1:])}
print(json.dumps(out, ensure_ascii=False, indent=1))
json.dump(out, open(os.path.join(WORK, "coverage_check.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
