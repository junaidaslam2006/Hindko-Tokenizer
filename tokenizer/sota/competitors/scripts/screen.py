# -*- coding: utf-8 -*-
"""Refreshed competitor sweep, stage 4: quick full-test-split screen of EVERY downloaded tokenizer.

For each tokenizer: encode all 491 test_strict documents with the same adapter the harness uses (no special tokens),
decode, and record per-document token counts and exact round trips -> bytes/token, G1 (docs round-tripped), UNK count,
and a behaviour fingerprint (sha256 of the per-document (tokens, round-trip) vector).  Tokenizers with the same
fingerprint as each other, or as one of the 66 baselines (eval/results/test_strict/*/docs.jsonl), are grouped; the full
harness (run_refresh.py) then runs once per distinct fingerprint.
    python screen.py --shard K --of 3
Output: ../screen/<name>.json (resumable), errors in ../logs/screen_errors_shardK.jsonl
"""
import argparse
import gc
import hashlib
import json
import os
import sys
import time
import traceback

os.environ["RAYON_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import refresh_lib as R  # noqa: E402

ROOT = R.ROOT
OUT = os.path.join(ROOT, "screen")
LOG = os.path.join(ROOT, "_download_log.json")


def fingerprint(rows):
    return hashlib.sha256(json.dumps([[r[0], r[1]] for r in rows]).encode()).hexdigest()


def screen(a, docs):
    rows, t0 = [], time.time()
    tot_b = tot_t = rt_ok = unk = 0
    for d in docs:
        t = d["text"]
        ids = a.encode(t)
        ok = a.decode(ids) == t
        n = len(ids)
        rows.append((n, int(ok)))
        tot_b += len(t.encode("utf-8"))
        tot_t += n
        rt_ok += ok
        unk += sum(1 for i in ids if i in a.unk_ids)
    return {"docs": len(docs), "bytes": tot_b, "tokens": tot_t, "bytes_per_token": tot_b / tot_t if tot_t else None,
            "g1_pass_docs": rt_ok, "unk_tokens": unk, "fingerprint": fingerprint(rows), "per_doc_tokens": [r[0] for r in rows],
            "per_doc_rt": [r[1] for r in rows], "seconds": round(time.time() - t0, 1)}


def names_from_log():
    for t in range(10):                       # the downloader may be rewriting the log right now
        try:
            log = json.load(open(LOG, encoding="utf-8"))
            break
        except json.JSONDecodeError:
            time.sleep(3)
    out = []
    for sig, rec in sorted(log.items(), key=lambda kv: (kv[1]["repo"], kv[1].get("dir") or "")):
        if "error" in rec:
            continue
        rec = dict(rec)
        rec["name"] = rec["local_dir"].split("/", 1)[1]
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    docs = R.load_test()
    recs = names_from_log()[a.shard::a.of]
    err = os.path.join(ROOT, "logs", "screen_errors_shard%d.jsonl" % a.shard)
    print("shard %d/%d: %d tokenizers" % (a.shard, a.of, len(recs)), flush=True)
    for i, rec in enumerate(recs):
        p = os.path.join(OUT, rec["name"] + ".json")
        if os.path.exists(p):
            continue
        res = {"name": rec["name"], "repo": rec["repo"], "dir": rec.get("dir"), "revision": rec.get("revision"), "kind": rec["kind"],
               "sig": rec["sig"], "n_members": rec.get("n_members"), "members": rec.get("members"), "downloads": rec.get("downloads"),
               "likes": rec.get("likes"), "created": rec.get("created"), "how": rec.get("how")}
        spec, why = R.make_spec(rec)
        if spec is None:
            res["status"] = "not loadable: " + why
        else:
            res["spec"] = {k: v for k, v in spec.items()}
            try:
                ad = R.adapter(spec)
                res.update(vocab_size=ad.vocab_size, model_type=getattr(ad, "model_type", None),
                           byte_level=getattr(ad, "byte_level", None), byte_fallback=getattr(ad, "byte_fallback", None))
                res["screen"] = screen(ad, docs)
                res["status"] = "ok"
                del ad
            except (Exception, BaseException) as e:  # noqa: BLE001  (pyo3 PanicException derives from BaseException)
                if isinstance(e, (KeyboardInterrupt, SystemExit)):
                    raise
                res["status"] = "error: %s: %s" % (type(e).__name__, str(e).splitlines()[0][:200] if str(e) else "")
                with open(err, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"name": rec["name"], "error": res["status"], "trace": traceback.format_exc()[-2000:]}) + "\n")
        json.dump(res, open(p, "w", encoding="utf-8"), ensure_ascii=False)
        s = res.get("screen") or {}
        print(i, rec["name"], res["status"][:80], s.get("bytes_per_token") and round(s["bytes_per_token"], 3), s.get("g1_pass_docs"),
              flush=True)
        gc.collect()


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
