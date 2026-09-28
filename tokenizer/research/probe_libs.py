# -*- coding: utf-8 -*-
"""Probe the installed tokenizer libraries for the options the plan relies on."""
import inspect, json
import tokenizers
from tokenizers import trainers, models, pre_tokenizers, normalizers, decoders
import sentencepiece as spm

out = {"tokenizers": tokenizers.__version__, "sentencepiece": spm.__version__}
for name in ["BpeTrainer", "UnigramTrainer", "WordPieceTrainer"]:
    cls = getattr(trainers, name)
    out[name] = cls.__doc__.split("Args:")[-1].strip()[:3000] if cls.__doc__ else None
out["BPE_model_doc"] = models.BPE.__doc__[:2500]
out["pre_tokenizers"] = sorted(n for n in dir(pre_tokenizers) if not n.startswith("_"))
out["normalizers"] = sorted(n for n in dir(normalizers) if not n.startswith("_"))
# sentencepiece trainer flags: print the help text via a bogus flag
try:
    spm.SentencePieceTrainer.train("--help")
except Exception as e:
    out["spm_help_error"] = str(e)[:200]
print(json.dumps(out, indent=1, ensure_ascii=False))
