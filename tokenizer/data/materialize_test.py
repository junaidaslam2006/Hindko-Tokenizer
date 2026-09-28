# -*- coding: utf-8 -*-
"""PLAN Stage 5 (one-shot final test, PLAN 7.6), step 1: materialise the NORMALISED strict TEST view.

    set PYTHONIOENCODING=utf-8 & python data\\materialize_test.py

This is the only script of the study that reads the text of the test split. It is data/materialize.py
restricted to ONE view, with the same code path:
  * FROZEN.json verification (split manifest, normalize.py, both released datasets) = materialize.verify_frozen;
  * splits/load_split.py iter_split(split, tier='strict'): every yielded record is checked against the split
    manifest (uid, source, n_words, quality_tier);
  * hp.normalize.normalize() version 1.0.1 on every text;
  * the same record fields {uid, group, cluster, source, variety, tier, text}, the same bootstrap-cluster rule
    (materialize.boot_cluster), documents in dataset order, one JSON line each, UTF-8, '\\n' line ends.

Equivalence proof, run BEFORE any test row is read: the same row function applied to the validation split
must reproduce data/dev_strict.jsonl byte for byte (sha256 recorded in data/data_manifest.json).

Writes (F:/Hindko/_tokenizer/data/):
  test_strict.jsonl     the strict test split (split == 'test', quality_tier == 'strict'), normalised
  test_manifest.json    the same statistics block as data_manifest.json 'views' (sizes, per source / variety /
                        tier, groups, bootstrap clusters) + provenance. data_manifest.json is NOT modified: its
                        sha256 is recorded in the earlier LM bundles.
No metric of any tokenizer or model is computed here.
"""
import datetime
import hashlib
import json
import os
import sys
import time

TOK = r"F:\Hindko\_tokenizer"
HERE = os.path.join(TOK, "data")
sys.dont_write_bytecode = True          # read-only import of materialize.py; no __pycache__ in data/
sys.path.insert(0, HERE)
import materialize as M                 # noqa: E402  (verify_frozen, boot_cluster, sha256_file, FROZEN, sys.path)

VIEW = "test_strict"


def rows(split, tier, normalize, iter_split, man):
    """materialize.main().rows() for one view (the per-uid cache there only avoids re-normalising the
    same text for the permissive and strict tiers; it does not change any output)."""
    out, changed = [], set()
    for uid, text, rec in iter_split(split, tier=tier):
        norm = normalize(text)
        if norm != text:
            changed.add(uid)
        g = man[uid]["group"]
        out.append({"uid": uid, "group": g, "cluster": M.boot_cluster(g), "source": rec["source"],
                    "variety": rec["language_variety"], "tier": rec["quality_tier"], "text": norm})
    return out, changed


def serialise(rs):
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rs).encode("utf-8")


def stats(rs, changed, path):
    """The statistics block of materialize.main() for one view."""
    st = {"file": os.path.basename(path), "sha256": M.sha256_file(path), "file_bytes": os.path.getsize(path)}
    agg = lambda sel: {"docs": len(sel), "chars": sum(len(r["text"]) for r in sel),
                       "bytes_utf8": sum(len(r["text"].encode("utf-8")) for r in sel),
                       "words_whitespace": sum(len(r["text"].split()) for r in sel),
                       "line_breaks": sum(r["text"].count("\n") for r in sel)}
    st["total"] = agg(rs)
    st["by_source"] = {s: agg([r for r in rs if r["source"] == s]) for s in sorted({r["source"] for r in rs})}
    st["by_variety"] = {v: agg([r for r in rs if r["variety"] == v]) for v in sorted({r["variety"] for r in rs})}
    st["by_tier"] = {v: agg([r for r in rs if r["tier"] == v]) for v in sorted({r["tier"] for r in rs})}
    st["groups"] = len({r["group"] for r in rs})
    st["bootstrap_clusters"] = {s: len({r["cluster"] for r in rs if r["source"] == s})
                                for s in sorted({r["source"] for r in rs})}
    st["docs_changed_by_normalize"] = sum(1 for r in rs if r["uid"] in changed)
    tc = st["total"]["chars"]
    st["book_char_share"] = round(st["by_source"].get("book", {"chars": 0})["chars"] / tc, 6) if tc else None
    return st


def main():
    t0 = time.time()
    out_p = os.path.join(HERE, VIEW + ".jsonl")
    if os.path.exists(out_p):
        raise SystemExit("%s already exists: the test view is materialised once" % out_p)
    M.verify_frozen()
    from hp.normalize import normalize, NORMALIZATION_VERSION
    from load_split import iter_split, load_manifest
    assert NORMALIZATION_VERSION == M.FROZEN["normalize"]["version"] == "1.0.1"
    man = load_manifest()
    dm = json.load(open(os.path.join(HERE, "data_manifest.json"), encoding="utf-8"))

    # (1) equivalence proof on the validation split (no test row read yet)
    dev, _ = rows("validation", "strict", normalize, iter_split, man)
    dev_sha = hashlib.sha256(serialise(dev)).hexdigest()
    want = dm["views"]["dev_strict"]["sha256"]
    if dev_sha != want or M.sha256_file(os.path.join(HERE, "dev_strict.jsonl")) != want:
        raise SystemExit("row function does not reproduce dev_strict.jsonl (%s vs %s) - stop" % (dev_sha, want))
    print("equivalence: validation/strict re-materialised = dev_strict.jsonl byte for byte (%s)" % dev_sha[:12], flush=True)

    # (2) the strict test view
    rs, changed = rows("test", "strict", normalize, iter_split, man)
    bad = [r["uid"] for r in rs if man[r["uid"]]["split"] != "test" or man[r["uid"]]["quality_tier"] != "strict"
           or r["tier"] != "strict"]
    if bad or len({r["uid"] for r in rs}) != len(rs):
        raise SystemExit("manifest check failed for %d rows (or duplicate uids)" % len(bad))
    n_man = sum(1 for v in man.values() if v["split"] == "test" and v["quality_tier"] == "strict")
    if n_man != len(rs):
        raise SystemExit("manifest lists %d strict test uids, dataset yielded %d" % (n_man, len(rs)))
    with open(out_p, "wb") as f:
        f.write(serialise(rs))
    st = stats(rs, changed, out_p)
    print(VIEW, json.dumps(st["total"]), "clusters", st["bootstrap_clusters"], flush=True)
    out = {"what": "Normalised (hp.normalize %s) strict TEST view of the Hindko corpus: the one-shot final test of "
                   "the tokenizer study (PLAN 7.6). Made exactly like data_manifest.json's dev_strict." % NORMALIZATION_VERSION,
           "created_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "split_manifest_sha256": M.FROZEN["split_manifest"]["sha256"],
           "normalize_version": NORMALIZATION_VERSION, "normalize_sha256": M.FROZEN["normalize"]["sha256"],
           "datasets_sha256": {k: M.FROZEN["datasets"][k]["sha256"] for k in ("permissive", "strict")},
           "materialize_test_py_sha256": M.sha256_file(os.path.abspath(__file__)),
           "materialize_py_sha256": M.sha256_file(os.path.join(HERE, "materialize.py")),
           "materialize_py_sha256_in_data_manifest": dm["materialize_py_sha256"],
           "record_fields": ["uid", "group", "cluster", "source", "variety", "tier", "text"],
           "equivalence_check": {"what": "the same row function re-materialised validation/strict; its bytes equal "
                                         "data/dev_strict.jsonl", "sha256": dev_sha, "equal": True},
           "manifest_check": {"strict_test_uids_in_split_manifest": n_man, "rows_written": len(rs),
                              "every_row_split_test_tier_strict": True},
           "expected_from_split_facts": {"docs": 491, "raw_MB": 1.451, "clusters": {"newspaper": 15, "book": 6, "web": 6}},
           "records_changed_by_normalize_total": len(changed),
           "views": {VIEW: st}, "seconds": round(time.time() - t0, 1),
           "discipline": "materialised once, for the one-shot final test bundle (colab/build_bundle.py --final-test) "
                         "and the one-shot intrinsic competitor table (eval/run_test_competitors.py)"}
    with open(os.path.join(HERE, "test_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("done", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
