# -*- coding: utf-8 -*-
"""How much of the Arabic-keyboard cost does a simple input fold remove? (report-only add-on to run_eval.py)

For every tokenizer, on test_strict: tokens of
  clean            the canonical text
  ak               perturb_extra.perturb_arabic_keyboard (YEH, KAF, HEH GOAL -> ARABIC YEH, KAF, HEH)
  ak_fold_yk       ak, then ARABIC YEH -> FARSI YEH and ARABIC KAF -> KEHEH everywhere (the fold the report
                   recommends for text known to be Hindko/Urdu); ARABIC HEH is left as typed, because it can also
                   stand for HEH DOACHASHMEE. What remains is the cost of HEH alone (plus the few ARABIC YEH/KAF of
                   the canonical text, which the fold also changes).
Output: results/ak_mitigation.json
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rcommon as C  # noqa: E402
import perturb_extra as PX  # noqa: E402

FOLD = str.maketrans({chr(0x064A): chr(0x06CC), chr(0x0643): chr(0x06A9)})


def main():
    test = C.load_test()
    ak = [PX.perturb_arabic_keyboard(d["text"]).text for d in test]
    fold = [t.translate(FOLD) for t in ak]
    out = {"what": __doc__.strip().splitlines()[0], "tokenizers": {}}
    for k in C.TOK_KEYS:
        t0 = time.time()
        ad = C.load_tokenizer(k)
        n0 = sum(len(ad.encode(d["text"])) for d in test)
        n1 = sum(len(ad.encode(t)) for t in ak)
        n2 = sum(len(ad.encode(t)) for t in fold)
        out["tokenizers"][k] = {"clean": n0, "ak": n1, "ak_fold_yk": n2, "rel_ak": (n1 - n0) / n0,
                                "rel_ak_fold_yk": (n2 - n0) / n0}
        print(k, n0, n1, n2, "%.0fs" % (time.time() - t0), flush=True)
    json.dump(out, open(os.path.join(C.RESULTS, "ak_mitigation.json"), "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
