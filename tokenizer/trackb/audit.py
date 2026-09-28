# -*- coding: utf-8 -*-
"""Static and exhaustive non-interference audit of the Track B extensions (added after the check set found that
Qwen3.5 at k >= 8,192 changes Python code).

  static      classify every new merge of the K_MAX list by its RESULT bytes:
                arabic      contains at least one complete Arabic-script character (Script_Extensions=Arabic):
                            the merge can only fire in text that contains that character;
                partial     not valid UTF-8 (a byte-level piece of a character) ending in a prefix that only
                            Arabic-script characters complete (continued_bpe.in_scope);
                partial_out any other partial-UTF-8 token (e.g. continuation bytes only);
                other       valid UTF-8 without any Arabic-script character: it can fire in non-Arabic text.
              For every base: the first new-token rank of each class and the counts inside each k of the grid.
  codepoints  every Unicode scalar value (U+0000..U+10FFFF without surrogates), alone and after a space, encoded
              by the base and by each extension of the grid: number of code points whose ids change, split into
              Arabic-script and other code points. This covers partial-UTF-8 merges on single characters.

    python audit.py      -> results/audit.json
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tb_common as C  # noqa: E402
import continued_bpe as CB  # noqa: E402

C.env_threads(2)
from tokenizers import Tokenizer  # noqa: E402


def classify(kind, s):
    b = CB.token_text_bytes(kind, s)
    try:
        t = b.decode("utf-8")
    except UnicodeDecodeError:
        return ("partial" if CB.in_scope(kind, s) else "partial_out"), None
    return ("arabic" if C.has_arabic(t) else "other"), t


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--static-only", action="store_true")
    ap.add_argument("--out", default=os.path.join(C.RESULTS, "audit.json"))
    ap.add_argument("--bases", nargs="+", default=C.BASE_ORDER, help="update only these bases in --out")
    a = ap.parse_args()
    out = {"what": __doc__.split("\n\n")[0], "static": {}, "codepoints": {}}
    if os.path.exists(a.out) and a.bases != C.BASE_ORDER:
        old = C.load_json(a.out)
        for b in C.BASE_ORDER:
            if b not in a.bases:
                for sec in ("static", "codepoints"):
                    if b in old.get(sec, {}):
                        out[sec][b] = old[sec][b]
    cps = [chr(c) for c in range(0x110000) if not (0xD800 <= c <= 0xDFFF)]
    texts = cps + [" " + c for c in cps]
    is_ar = [C.has_arabic(c) for c in cps] * 2
    for base in [x for x in C.BASE_ORDER if x in a.bases]:
        ext = C.load_json(os.path.join(C.WORK, base, "extension.json"))
        kind = ext["base_kind"]
        rows = []
        for m in ext["merges"]:
            if m["kind"] != "new":
                continue
            cl, t = classify(kind, m["result"])
            rows.append((m["n_new"], cl, t, m["left"], m["right"]))
        st = {"first_rank": {}, "by_k": {}, "other_examples": []}
        for cl in ("arabic", "partial", "partial_out", "other"):
            f = [r[0] for r in rows if r[1] == cl]
            st["first_rank"][cl] = f[0] if f else None
        for k in C.K_GRID:
            st["by_k"][k] = {cl: sum(1 for r in rows if r[0] <= k and r[1] == cl)
                          for cl in ("arabic", "partial", "partial_out", "other")}
        st["other_examples"] = [{"n_new": r[0], "text": r[2], "left": r[3], "right": r[4]} for r in rows if r[1] == "other"][:40]
        out["static"][base] = st
        C.log(base, "static:", json.dumps(st["by_k"]), "first other:", st["first_rank"]["other"])
        if a.static_only:
            C.dump_json(out, a.out)
            continue
        # exhaustive code-point test
        bt = Tokenizer.from_file(os.path.join(C.SWEEP, base, "k0", "tokenizer.json"))
        base_ids = [e.ids for e in bt.encode_batch(texts, add_special_tokens=False)]
        out["codepoints"][base] = {}
        for k in C.K_GRID:
            et = Tokenizer.from_file(os.path.join(C.SWEEP, base, "k%d" % k, "tokenizer.json"))
            ch_ar = ch_other = 0
            ex = []
            for i, e in enumerate(et.encode_batch(texts, add_special_tokens=False)):
                if e.ids != base_ids[i]:
                    if is_ar[i]:
                        ch_ar += 1
                    else:
                        ch_other += 1
                        if len(ex) < 10:
                            ex.append("U+%04X%s" % (ord(texts[i][-1]), " (after space)" if len(texts[i]) == 2 else ""))
            out["codepoints"][base][k] = {"texts": len(texts), "changed_arabic_script": ch_ar,
                                          "changed_other": ch_other, "other_examples": ex}
            C.log(base, k, out["codepoints"][base][k])
        C.dump_json(out, a.out)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
