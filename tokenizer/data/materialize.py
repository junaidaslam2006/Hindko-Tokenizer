# -*- coding: utf-8 -*-
"""PLAN Stage 0, step 2: materialise the NORMALISED text of the train and dev views.

Inputs (read-only): the released datasets through splits/load_split.py (iter_split), with the
split manifest and normalize.py verified against F:/Hindko/_tokenizer/FROZEN.json first.
Every text goes through hp.normalize.normalize() (canonical data form, v1.0.1).

Writes (F:/Hindko/_tokenizer/data/):
  train_D1.jsonl          permissive train (all tiers)                         PLAN 2.1 D1
  train_D2.jsonl          strict train only                                    PLAN 2.1 D2
  train_D3.jsonl          D1 with book characters down-sampled to <= 50%       PLAN 2.1 D3
  dev_strict.jsonl        strict validation = THE evaluation set
  dev_permissive.jsonl    permissive validation (reporting only)
  train_D{1,2,3}.lines.txt  every non-empty line of every document, in document order, one per
                          line (what the SentencePiece trainer reads; PLAN 1.1: SentencePiece is
                          trained on lines and encodes documents line by line with a newline piece)
  data_manifest.json      sizes, per-source / per-variety breakdowns, sha256 of every file, D3 draw
No test-split file is written, and nothing is computed on test (iter_split skips test rows
before yielding them).

JSONL record: {uid, group, cluster, source, variety, tier, text}
  group    the manifest group (split unit; PLAN 1.2)
  cluster  the bootstrap cluster of PLAN 6: the group, except per-record web groups
           ('web:<site>:record:<uid>') collapse to their site 'web:<site>'
  variety  language_variety of the record; tier = quality_tier
Documents keep their boundaries (one JSON record per document; text contains its line breaks).

D3 draw (seed 20260926): book documents of D1 are sorted by uid, shuffled with
random.Random(20260926), and kept in that order while the kept book characters stay
<= the non-book characters of D1 (so book <= 50% of D3 characters). Kept documents are
written in D1 order.
"""
import collections
import datetime
import hashlib
import json
import os
import random
import sys
import time

TOK = r"F:\Hindko\_tokenizer"
HERE = os.path.join(TOK, "data")
sys.path.insert(0, os.path.join(TOK, "splits"))
sys.path.insert(0, r"F:\Hindko\_pipeline")
FROZEN = json.load(open(os.path.join(TOK, "FROZEN.json"), encoding="utf-8"))
D3_SEED = 20260926
D3_MAX_BOOK_SHARE = 0.5


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def verify_frozen():
    got = sha256_file(FROZEN["split_manifest"]["path"])
    if got != FROZEN["split_manifest"]["sha256"]:
        raise SystemExit("split manifest changed since FROZEN.json: %s" % got)
    got = sha256_file(FROZEN["normalize"]["path"])
    if got != FROZEN["normalize"]["sha256"]:
        raise SystemExit("normalize.py changed since FROZEN.json: %s" % got)
    for k in ("permissive", "strict"):
        got = sha256_file(FROZEN["datasets"][k]["path"])
        if got != FROZEN["datasets"][k]["sha256"]:
            raise SystemExit("dataset %s changed since FROZEN.json" % k)


def boot_cluster(group):
    return group.split(":record:")[0] if group.startswith("web:") and ":record:" in group else group


def main():
    t0 = time.time()
    verify_frozen()
    from hp.normalize import normalize, NORMALIZATION_VERSION
    from load_split import iter_split, load_manifest
    assert NORMALIZATION_VERSION == FROZEN["normalize"]["version"]
    man = load_manifest()
    cache = {}          # uid -> (raw sha1, normalised text); strict texts equal permissive texts
    n_changed = collections.Counter()

    def rows(split, tier):
        out = []
        for uid, text, rec in iter_split(split, tier=tier):
            assert split != "test"
            key = hashlib.sha1(text.encode("utf-8")).hexdigest()
            hit = cache.get(uid)
            if hit is None or hit[0] != key:
                norm = normalize(text)
                cache[uid] = (key, norm)
                if norm != text:
                    n_changed[uid] = 1
            norm = cache[uid][1]
            g = man[uid]["group"]
            out.append({"uid": uid, "group": g, "cluster": boot_cluster(g), "source": rec["source"],
                        "variety": rec["language_variety"], "tier": rec["quality_tier"], "text": norm})
        return out

    views = {}
    views["train_D1"] = rows("train", "permissive")
    print("D1", len(views["train_D1"]), round(time.time() - t0, 1), "s", flush=True)
    views["train_D2"] = rows("train", "strict")
    print("D2", len(views["train_D2"]), round(time.time() - t0, 1), "s", flush=True)
    views["dev_strict"] = rows("validation", "strict")
    views["dev_permissive"] = rows("validation", "permissive")
    print("dev", len(views["dev_strict"]), len(views["dev_permissive"]), round(time.time() - t0, 1), "s", flush=True)

    # --- D3: book characters <= 50 % of D1 characters, whole documents, fixed seed
    d1 = views["train_D1"]
    book = sorted((r for r in d1 if r["source"] == "book"), key=lambda r: r["uid"])
    nonbook_chars = sum(len(r["text"]) for r in d1 if r["source"] != "book")
    budget = int(nonbook_chars * D3_MAX_BOOK_SHARE / (1 - D3_MAX_BOOK_SHARE))
    rng = random.Random(D3_SEED)
    order = list(book)
    rng.shuffle(order)
    kept, cum = set(), 0
    for r in order:
        n = len(r["text"])
        if cum + n <= budget:
            kept.add(r["uid"]); cum += n
    views["train_D3"] = [r for r in d1 if r["source"] != "book" or r["uid"] in kept]
    d3_info = {"seed": D3_SEED, "rule": "book docs of D1 sorted by uid, shuffled with random.Random(seed); a doc is kept "
                                        "if the kept book chars stay <= non-book chars (book share <= 50%); kept docs "
                                        "written in D1 order",
               "d1_book_docs": len(book), "d1_book_chars": sum(len(r["text"]) for r in book),
               "d1_nonbook_chars": nonbook_chars, "book_char_budget": budget,
               "d3_book_docs_kept": len(kept), "d3_book_chars": cum,
               "d3_book_char_share": round(cum / (cum + nonbook_chars), 6),
               "kept_book_uids_sha256": hashlib.sha256("\n".join(sorted(kept)).encode()).hexdigest()}
    print("D3", d3_info, flush=True)

    # --- write files + stats
    stats = {}
    for name, rs in views.items():
        path = os.path.join(HERE, name + ".jsonl")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        st = {"file": os.path.basename(path), "sha256": sha256_file(path), "file_bytes": os.path.getsize(path)}
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
        st["docs_changed_by_normalize"] = sum(1 for r in rs if r["uid"] in n_changed)
        tc = st["total"]["chars"]
        st["book_char_share"] = round(st["by_source"].get("book", {"chars": 0})["chars"] / tc, 6) if tc else None
        if name.startswith("train_"):
            lp = os.path.join(HERE, name + ".lines.txt")
            nl, mx = 0, 0
            with open(lp, "w", encoding="utf-8", newline="\n") as f:
                for r in rs:
                    for ln in r["text"].split("\n"):
                        if ln:
                            assert "\r" not in ln
                            f.write(ln + "\n"); nl += 1; mx = max(mx, len(ln.encode("utf-8")))
            st["lines_txt"] = {"file": os.path.basename(lp), "sha256": sha256_file(lp), "file_bytes": os.path.getsize(lp),
                               "non_empty_lines": nl, "max_line_bytes": mx}
        stats[name] = st
        print(name, json.dumps(st["total"]), flush=True)
    stats["train_D3"]["d3_draw"] = d3_info
    out = {"what": "Normalised (hp.normalize %s) train/dev views of the Hindko corpus for the tokenizer study" % NORMALIZATION_VERSION,
           "created_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "split_manifest_sha256": FROZEN["split_manifest"]["sha256"],
           "normalize_version": NORMALIZATION_VERSION, "normalize_sha256": FROZEN["normalize"]["sha256"],
           "materialize_py_sha256": sha256_file(os.path.abspath(__file__)),
           "record_fields": ["uid", "group", "cluster", "source", "variety", "tier", "text"],
           "test_split": "not written, not read beyond iter_split's row filter",
           "records_changed_by_normalize_total": len(n_changed),
           "views": stats, "seconds": round(time.time() - t0, 1)}
    with open(os.path.join(HERE, "data_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("done", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()
