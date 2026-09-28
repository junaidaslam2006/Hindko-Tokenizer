# -*- coding: utf-8 -*-
"""Factory for the A7 PickyBPE candidate, for eval/harness.py (`--custom pickybpe_factory:load_16k`) and for
colab/build_bundle.py (WAVES.json entry: encoder_kind "custom", custom "pickybpe_factory:load_16k").

Both import the module by name, so put this folder on the path first:
    set PYTHONPATH=F:\\Hindko\\_tokenizer\\candidates\\pickybpe

The returned object satisfies eval/adapters.CustomAdapter: encode, decode, encode_batch, encode_isolated,
vocab_size, id_upper, special_tokens, base_ids, unk_ids, token_bytes, merges, byte_level, model_type, identity.
The model file is pinned by sha256: a changed file is refused.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from pickybpe import PickyBPETokenizer  # noqa: E402
from pickybpe_common import sha256_file  # noqa: E402

NAME_16K = "A7_pickybpe_P1_D1_16k_tau0.9"
MODEL_16K = os.path.join(HERE, "models", "pickybpe_P1_D1_16k_tau0.9.json")
SHA256_16K = "1508d094532db24ca3d74704d8be91ba126e2e6f2f648ee3c29c6fcbce4dd6cc"


def load_16k() -> PickyBPETokenizer:
    got = sha256_file(MODEL_16K)
    if got != SHA256_16K:
        raise RuntimeError("%s: sha256 %s != pinned %s" % (MODEL_16K, got, SHA256_16K))
    return PickyBPETokenizer(MODEL_16K, name=NAME_16K)
