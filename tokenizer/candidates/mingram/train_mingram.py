# -*- coding: utf-8 -*-
"""Train MinGram as reimplemented from the paper (Land 2026, arXiv 2606.27019); PLAN.md A10, Stage 2.

    python train_mingram.py --out runs/mingram_P1_D1_16384 [--ablation] [--tie hf|paper] [--no-cache]

Inputs: data/train_D1.jsonl (permissive train, already hp.normalize 1.0.1; normalize() is applied again and
the number of changed documents recorded). Loaded through eval/harness.load_docs, which verifies the frozen
split manifest and REFUSES any file containing a test-split uid. Nothing here reads dev except the report-only
N_em ablation (--ablation), which encodes dev_strict to report bytes/token; it selects nothing.

Writes into --out:
  seed_bpe.json          the BPE seed (HF BpeTrainer, char-level, P1 Split), ceil(1.15 n) learned tokens
  mingram_model.json     pieces (id, piece, kind, score_q) + per-piece counts (BPE init, each EM pass), params
  tokenizer.json         the HF-native candidate (stock tokenizers Unigram, scores -C + log p)
  vocab.tsv              id, piece (escaped), kind, score, log p, final EM count, BPE-init count, BPE merge rank
  train_log.json         parameters, sizes, per-phase timings, sha256 of every input and output
"""
from __future__ import annotations

import argparse
import collections
import datetime
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--data", default="train_D1")
ap.add_argument("--total", type=int, default=16384, help="total vocabulary incl. 64 specials, <unk>, 256 bytes")
ap.add_argument("--f-num", type=int, default=115, help="overshoot factor f = f_num / 100 (exact integer ceil)")
ap.add_argument("--em", type=int, default=2)
ap.add_argument("--tie", default="hf", choices=["hf", "paper"])
ap.add_argument("--min-frequency", type=int, default=2, help="BpeTrainer min_frequency of the seed")
ap.add_argument("--threads", type=int, default=2)
ap.add_argument("--ablation", action="store_true", help="report-only N_em in {0,1,2,3} (dev_strict bytes/token)")
ap.add_argument("--no-cache", action="store_true", help="recompute the pretoken type counts (G4 retrain)")
ARGS = ap.parse_args()
os.environ["RAYON_NUM_THREADS"] = str(ARGS.threads)
os.environ["TOKENIZERS_PARALLELISM"] = "true" if ARGS.threads > 1 else "false"
os.environ["OMP_NUM_THREADS"] = "1"

sys.path.insert(0, HERE)
import mingram as M  # noqa: E402
H = M.H
sys.path.insert(0, r"F:\Hindko\_pipeline")
from hp.normalize import normalize  # noqa: E402

WORK = os.path.join(HERE, "work")


def ceil_frac(n, num, den=100):
    return -(-n * num // den)


def esc(s):
    return s.replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n").replace("\r", "\\r")


def main():
    t_all = time.time()
    out = ARGS.out if os.path.isabs(ARGS.out) else os.path.join(HERE, ARGS.out)
    os.makedirs(out, exist_ok=True)
    os.makedirs(WORK, exist_ok=True)
    tm = {}
    frozen = json.load(open(H.FROZEN_PATH, encoding="utf-8"))
    man_sha = M.sha256_file(frozen["split_manifest"]["path"])
    assert man_sha == "76582d3a1e0afefe64cdbf892f2214e0b677143dbec7082ffaa8fe3fb4f94aa2", man_sha
    norm_sha = M.sha256_file(frozen["normalize"]["path"])
    assert norm_sha == frozen["normalize"]["sha256"], "normalize.py differs from FROZEN.json"

    # ---------------------------------------------------------------- data + pretoken types
    t0 = time.time()
    docs, dinfo = H.load_docs(ARGS.data)          # refuses test-split uids
    assert dinfo["test_docs"] == 0
    texts = [normalize(d["text"]) for d in docs]
    n_changed = sum(1 for d, t in zip(docs, texts) if d["text"] != t)
    del docs
    tm["load_normalize_s"] = round(time.time() - t0, 1)
    t0 = time.time()
    cache = os.path.join(WORK, "types_%s_%s.json" % (ARGS.data, dinfo["sha256"][:16]))
    if os.path.exists(cache) and not ARGS.no_cache:
        c = json.load(open(cache, encoding="utf-8"))
        assert c["p1"] == M.P1 and c["normalize_sha256"] == norm_sha and c["data_sha256"] == dinfo["sha256"]
        types = c["types"]
        cache_used = True
    else:
        tc = M.type_counts(texts)
        types = dict(sorted(tc.items(), key=lambda kv: (-kv[1], kv[0])))
        cache_used = False
        if not ARGS.no_cache:
            json.dump({"p1": M.P1, "normalize_sha256": norm_sha, "data_sha256": dinfo["sha256"], "types": types},
                      open(cache, "w", encoding="utf-8"), ensure_ascii=False)
    tm["pretokenize_s"] = round(time.time() - t0, 1)
    n_pretokens = sum(types.values())
    alphabet = sorted({ch for s in types for ch in s})
    A = len(alphabet)
    n_special, n_unk, n_bytes = len(M.SPECIALS), 1, 256
    n = ARGS.total - n_special - n_unk - n_bytes - A
    n_seed = ceil_frac(n, ARGS.f_num)
    maxlen_type = max(len(s) for s in types)
    M.log("train %s: %d docs (%d changed by normalize), %d pretokens, %d types, alphabet %d, n=%d, seed=%d"
          % (ARGS.data, len(texts), n_changed, n_pretokens, len(types), A, n, n_seed))

    # ---------------------------------------------------------------- 1. BPE seed
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, Regex
    t0 = time.time()
    bpe = Tokenizer(models.BPE())
    bpe.pre_tokenizer = pre_tokenizers.Split(Regex(M.P1), behavior="isolated", invert=False)
    trn = trainers.BpeTrainer(vocab_size=A + n_seed, min_frequency=ARGS.min_frequency, show_progress=False,
                              special_tokens=[])
    bpe.train_from_iterator(texts, trainer=trn)
    tm["bpe_seed_train_s"] = round(time.time() - t0, 1)
    seed_path = os.path.join(out, "seed_bpe.json")
    bpe.save(seed_path, pretty=True)
    sj = json.loads(bpe.to_str())
    svocab = sj["model"]["vocab"]
    merges = [tuple(x) if not isinstance(x, str) else tuple(x.split(" ", 1)) for x in sj["model"]["merges"]]
    bpe_rank = {a + b: r for r, (a, b) in enumerate(merges)}
    seed_chars = sorted(s for s in svocab if len(s) == 1)
    seed_learned = [s for s in svocab if len(s) > 1]
    assert seed_chars == alphabet, "BPE alphabet differs from the pretoken alphabet"
    assert all(s in bpe_rank for s in seed_learned)
    M.log("BPE seed: %d chars + %d learned (%d merges) in %.1fs" % (len(seed_chars), len(seed_learned), len(merges),
                                                                   tm["bpe_seed_train_s"]))
    del texts

    # ---------------------------------------------------------------- 2. init log p from BPE token frequencies
    t0 = time.time()
    sj2 = json.loads(bpe.to_str())
    sj2["pre_tokenizer"] = None
    enc = Tokenizer.from_str(json.dumps(sj2, ensure_ascii=False))
    tl = list(types)
    bpe_cnt = collections.defaultdict(int)
    bpe_ntok = 0
    for i in range(0, len(tl), 20000):
        chunk = tl[i:i + 20000]
        for s, e in zip(chunk, enc.encode_batch(chunk, add_special_tokens=False)):
            f = types[s]
            toks = e.tokens
            assert "".join(toks) == s, s
            bpe_ntok += f * len(toks)
            for t in toks:
                bpe_cnt[t] += f
    pieces_all = list(alphabet) + seed_learned
    lp, st, lpf = M.mstep(bpe_cnt, pieces_all)
    tm["bpe_init_counts_s"] = round(time.time() - t0, 1)
    iters = [{"iter": 0, "what": "BPE-seed encoding of the train pretokens", "train_tokens": bpe_ntok,
              "zero_count_learned": sum(1 for p in seed_learned if bpe_cnt.get(p, 0) == 0), **st}]
    M.log("init: BPE seed encodes train in %d tokens; %d learned seed tokens have count 0" % (
        bpe_ntok, iters[0]["zero_count_learned"]))
    hist_lp = [lp]
    hist_lpf = [lpf]
    hist_cnt = [dict(bpe_cnt)]

    # ---------------------------------------------------------------- 3. hard EM on the minimum-token path
    n_iter = max(ARGS.em, 3) if ARGS.ablation else ARGS.em
    for it in range(1, n_iter + 1):
        t0 = time.time()
        minq = min(lp.values())
        lat = M.Lattice(lp, minq - int(M.HF_UNK_PENALTY * (1 << M.Q_BITS)))
        cnt, ntok = M.estep(types, lat, ARGS.tie)
        lp, st, lpf = M.mstep(cnt, pieces_all)
        dt = round(time.time() - t0, 1)
        tm["em%d_s" % it] = dt
        iters.append({"iter": it, "what": "hard EM (min-token path, tie=%s)" % ARGS.tie, "train_tokens": ntok,
                      "zero_count_learned": sum(1 for p in seed_learned if cnt.get(p, 0) == 0),
                      "zero_count_chars": sum(1 for p in alphabet if cnt.get(p, 0) == 0), "seconds": dt, **st})
        hist_lp.append(lp)
        hist_lpf.append(lpf)
        hist_cnt.append(cnt)
        M.log("EM %d: train tokens %d, zero-count learned %d, %.1fs" % (it, ntok, iters[-1]["zero_count_learned"], dt))

    # ---------------------------------------------------------------- 4. one flat prune to n
    def prune(k_em):
        c_last = hist_cnt[k_em]
        c_prev = hist_cnt[k_em - 1] if k_em >= 1 else {}
        key = lambda p: (c_last.get(p, 0), c_prev.get(p, 0), bpe_cnt.get(p, 0), -bpe_rank[p])
        order = sorted(seed_learned, key=key)
        n_drop = len(seed_learned) - n
        drop, keep = order[:n_drop], order[n_drop:]
        b_last, b_first = key(drop[-1]), key(keep[0])
        info = {"n_em": k_em, "seed_learned": len(seed_learned), "dropped": n_drop, "kept_learned": len(keep),
                "prune_key": "(count after last EM pass, count after the pass before, BPE-init count, -BPE merge rank)"
                             " ascending; the paper's 'lowest log p' is the first component",
                "last_dropped_count": b_last[0], "first_kept_count": b_first[0],
                "tokens_with_boundary_count": sum(1 for p in seed_learned if c_last.get(p, 0) == b_first[0]),
                "boundary_count_split_by_tiebreak": b_last[0] == b_first[0],
                "kept_zero_count": sum(1 for p in keep if c_last.get(p, 0) == 0),
                "kept_prob_mass": sum(c_last.get(p, 0) for p in keep + list(alphabet)) / max(1, sum(c_last.values()))}
        return keep, drop, info

    keep, drop, prune_info = prune(ARGS.em)
    lp_final = hist_lp[ARGS.em]
    score_q = {p: lp_final[p] for p in list(alphabet) + keep}
    c_final = hist_cnt[ARGS.em]
    extra = {"trained_on": {"data": ARGS.data, "path": dinfo["path"], "sha256": dinfo["sha256"], "docs": dinfo["docs"]},
             "params": {"total": ARGS.total, "n_learned": n, "f": ARGS.f_num / 100, "seed_learned_target": n_seed,
                        "n_em": ARGS.em, "tie": ARGS.tie, "bpe_min_frequency": ARGS.min_frequency,
                        "alphabet": A, "scores": "final EM pass log p, not renormalised after the prune "
                                                 "(renormalising cannot change any encoding)"}}
    model = M.build_model(alphabet, keep, score_q, extra)
    for p in model["pieces"]:
        s = p["piece"]
        if p["kind"] in ("char", "learned"):
            p["logp"] = hist_lpf[ARGS.em][s]
            p["count_em"] = int(c_final.get(s, 0))
            p["count_bpe_init"] = int(bpe_cnt.get(s, 0))
            p["bpe_rank"] = bpe_rank.get(s)
            p["counts_by_iter"] = [int(h.get(s, 0)) for h in hist_cnt[:ARGS.em + 1]]
    model_path = os.path.join(out, "mingram_model.json")
    with open(model_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(model, f, ensure_ascii=False, indent=0)
    tok_path = os.path.join(out, "tokenizer.json")
    M.export_hf(model, tok_path)
    sc_check = M.check_hf_scores(model, tok_path)
    M.log("model: %d pieces; HF score round trip: %s" % (model["vocab_size"], sc_check))
    assert model["vocab_size"] == ARGS.total, model["vocab_size"]
    assert sc_check["score_or_piece_mismatch"] == 0 and sc_check["special_ids_0_63"]

    with open(os.path.join(out, "vocab.tsv"), "w", encoding="utf-8", newline="\n") as f:
        f.write("id\tpiece\tkind\thf_score\tlogp\tcount_em_final\tcount_bpe_init\tbpe_merge_rank\n")
        for p in model["pieces"]:
            f.write("%d\t%s\t%s\t%r\t%.6f\t%s\t%s\t%s\n" % (
                p["id"], esc(p["piece"]), p["kind"], M.model_float_score(p["score_q"]), p["score_q"] / (1 << M.Q_BITS),
                p.get("count_em", ""), p.get("count_bpe_init", ""), "" if p.get("bpe_rank") is None else p["bpe_rank"]))

    # ---------------------------------------------------------------- final model on train (min-token count check)
    t0 = time.time()
    ref = M.MinGramRef(model_path, "hf")
    tr_tok = 0
    tr_cnt = collections.defaultdict(int)
    for s, fq in types.items():
        ids = ref._encode_pretoken(s)
        tr_tok += fq * len(ids)
        for i in ids:
            tr_cnt[i] += fq
    tr_bytes = sum(len(s.encode("utf-8")) * fq for s, fq in types.items())
    learned_ids = [p["id"] for p in model["pieces"] if p["kind"] == "learned"]
    tm["final_train_encode_s"] = round(time.time() - t0, 1)
    final_train = {"train_tokens": tr_tok, "train_bytes": tr_bytes, "train_bytes_per_token": tr_bytes / tr_tok,
                   "learned_zero_count": sum(1 for i in learned_ids if tr_cnt.get(i, 0) == 0),
                   "learned_lt20": sum(1 for i in learned_ids if tr_cnt.get(i, 0) < 20),
                   "note": "pretoken-type encoding with the reference encoder = whole-document encoding (P1 pretokens "
                           "are encoded independently; train has no special-block strings, G3)"}
    M.log("final model on train: %.4f bytes/token, learned count 0: %d, <20: %d" % (
        final_train["train_bytes_per_token"], final_train["learned_zero_count"], final_train["learned_lt20"]))

    # ---------------------------------------------------------------- report-only ablation of N_em
    ablation = None
    if ARGS.ablation:
        t0 = time.time()
        ddocs, ddinfo = H.load_docs("dev_strict")
        dtexts = [normalize(d["text"]) for d in ddocs]
        dtypes = M.type_counts(dtexts)
        dbytes = sum(len(t.encode("utf-8")) for t in dtexts)
        ablation = {"what": "REPORT ONLY, selects nothing (PLAN pre-registers N_em = 2): prune after k EM passes "
                            "and encode; tie=%s" % ARGS.tie, "dev": ddinfo["name"], "dev_sha256": ddinfo["sha256"],
                    "rows": []}
        for k in range(0, n_iter + 1):
            kk, _, pinfo = prune(k)
            lpk = hist_lp[k]
            table = {p: lpk[p] for p in list(alphabet) + kk}
            minq = min(table.values())
            lat = M.Lattice(table, minq - int(M.HF_UNK_PENALTY * (1 << M.Q_BITS)))
            dt_ = 0
            for s, fq in dtypes.items():
                for i, j, p in lat.best_hf(s):
                    dt_ += fq * (1 if p is not None else len(s[i:j].encode("utf-8")))
            trk = 0
            for s, fq in types.items():
                trk += fq * len(lat.best_hf(s))
            same_as_final = sorted(kk) == sorted(keep)
            ablation["rows"].append({"n_em": k, "dev_tokens": dt_, "dev_bytes": dbytes, "dev_bytes_per_token": dbytes / dt_,
                                     "train_tokens": trk, "train_bytes_per_token": tr_bytes / trk,
                                     "kept_zero_count": pinfo["kept_zero_count"],
                                     "vocab_identical_to_n_em_%d" % ARGS.em: same_as_final,
                                     "learned_overlap_with_final": len(set(kk) & set(keep))})
            M.log("ablation N_em=%d: dev %.4f bytes/token, train %.4f" % (k, dbytes / dt_, tr_bytes / trk))
        tm["ablation_s"] = round(time.time() - t0, 1)

    logd = {"label": M.LABEL, "mingram_version": M.MINGRAM_VERSION,
            "run_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "python": sys.version.split()[0], "tokenizers": __import__("tokenizers").__version__,
            "regex": __import__("regex").__version__, "threads": ARGS.threads, "argv": sys.argv[1:],
            "inputs": {"split_manifest_sha256": man_sha, "normalize_version": frozen["normalize"]["version"],
                       "normalize_sha256": norm_sha, "data": dinfo, "docs_changed_by_normalize": n_changed,
                       "pretoken_types_cache_used": cache_used},
            "code_sha256": {"mingram.py": M.sha256_file(os.path.join(HERE, "mingram.py")),
                            "train_mingram.py": M.sha256_file(os.path.abspath(__file__)),
                            "harness.py": M.sha256_file(os.path.join(M.TOK, "eval", "harness.py"))},
            "sizes": {"docs": dinfo["docs"], "pretokens": n_pretokens, "pretoken_types": len(types),
                      "longest_type_chars": maxlen_type, "alphabet": A, "specials": n_special, "unk": n_unk,
                      "byte_pieces": n_bytes, "n_learned": n, "seed_learned_target": n_seed,
                      "seed_learned_actual": len(seed_learned), "seed_merges": len(merges), "total": model["vocab_size"]},
            "iterations": iters, "prune": prune_info, "final_on_train": final_train, "hf_score_check": sc_check,
            "ablation": ablation, "timings_s": tm,
            "outputs_sha256": {os.path.basename(p): M.sha256_file(p) for p in
                               (seed_path, model_path, tok_path, os.path.join(out, "vocab.tsv"))}}
    tm["total_s"] = round(time.time() - t_all, 1)
    with open(os.path.join(out, "train_log.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(H.clean(logd), f, ensure_ascii=False, indent=1)
    M.log("done in %.0fs -> %s" % (tm["total_s"], out))
    M.log("tokenizer.json sha256 %s" % logd["outputs_sha256"]["tokenizer.json"])


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
