"""Check that examples/quickstart.py:normalize() == hp.normalize.normalize (1.0.1) on real, stress and fuzz text."""
import importlib.util
import json
import random
import sys

sys.path.insert(0, r"F:\Hindko\_pipeline")
from hp.normalize import normalize as ref, NORMALIZATION_VERSION  # noqa: E402

spec = importlib.util.spec_from_file_location("qs", r"F:\Hindko\hf_upload\examples\quickstart.py")
qs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qs)
assert NORMALIZATION_VERSION == qs.NORMALIZATION_VERSION

texts = []
for p in [r"F:\Hindko\_tokenizer\data\dev_strict.jsonl", r"F:\Hindko\_tokenizer\data\dev_permissive.jsonl"]:
    with open(p, encoding="utf-8") as f:
        texts += [json.loads(l)["text"] for l in f]
n_real = len(texts)
try:
    with open(r"F:\Hindko\_tokenizer\release_build\sp32k\stress\stress.jsonl", encoding="utf-8") as f:
        for l in f:
            o = json.loads(l)
            texts.append(o.get("text") or o.get("s") or "")
except FileNotFoundError:
    pass
n_stress = len(texts) - n_real
rng = random.Random(20260927)
pool = [chr(c) for c in list(range(0x20, 0x7F)) + list(range(0x600, 0x700)) + list(range(0x750, 0x780))
        + list(range(0x8A0, 0x900)) + list(range(0xFB50, 0xFB60)) + list(range(0xFDF0, 0xFE00))
        + list(range(0xFE70, 0xFE90)) + [0x200B, 0x200C, 0x200D, 0x200E, 0x200F, 0x202A, 0x2066, 0xFEFF, 0xAD,
                                         0x0D, 0x0A, 0x09, 0xA0, 0x3000, 0x2028, 0x85, 0x0B, 0x0C, 0x640, 0x65A]]
for _ in range(50000):
    texts.append("".join(rng.choice(pool) for _ in range(rng.randint(1, 40))))
bad = [t for t in texts if qs.normalize(t) != ref(t)]
idem = [t for t in texts[:n_real] if qs.normalize(qs.normalize(t)) != qs.normalize(t)]
print(json.dumps({"real_docs": n_real, "stress_items": n_stress, "fuzz": 50000, "mismatches": len(bad),
                  "idempotency_failures_real": len(idem)}))
if bad:
    print("first mismatch repr:", repr(bad[0][:200]))
