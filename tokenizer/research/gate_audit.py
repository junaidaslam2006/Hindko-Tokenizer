# -*- coding: utf-8 -*-
"""Run PLAN.md's tokenizer health gates on the pilot tokenizers (added 2026-09-26 after review).

The review found that the original gates 2-4 reject the plan's own baseline (A1 byte-level
BPE-P1-16k). This script measures exactly what each gate sees, so the rewritten gates in
PLAN.md 4.3 rest on numbers. Pilot split only (md5(uid) % 20 == 0 held out; NOT group-disjoint).

Parts (select with argv; default = all):
  audit    gates on the saved pilot tokenizers (HF byte-level BPE P1 16k/32k, SP Unigram 16k/32k)
  minfreq  retrain A1-16k with min_frequency 2 (determinism vs saved file), 20, 100, 400, 1000
  a2       train + audit char-level BPE with byte_fallback (A2) at 16k
  prune    leaf-prune-to-fixed-size (train 16k+m, iteratively delete the rarest leaf merges)
Writes research/gates/*.json. Reads the released dataset read-only.
"""
import hashlib, json, math, os, sys, time, collections
import numpy as np

THREADS = 3
os.environ["RAYON_NUM_THREADS"] = str(THREADS)
os.environ["TOKENIZERS_PARALLELISM"] = "true"
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, Regex
import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as spb

SRC = r"F:\Hindko\hindko_dataset_permissive.jsonl"
HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.join(HERE, "pilot")
OUT = os.path.join(HERE, "gates"); os.makedirs(OUT, exist_ok=True)
P1 = r" ?[\p{L}\p{M}\x{200C}\x{200D}]+| ?\p{N}| ?[^\s\p{L}\p{N}\p{M}]+|\s+(?!\S)|\s+"
MIN_SUPPORT = 20


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def load_split():
    tr, te = [], []
    with open(SRC, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            h = int(hashlib.md5(str(r["uid"]).encode()).hexdigest(), 16)
            (te if h % 20 == 0 else tr).append(r["text"])
    return tr, te


def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1)) + list(range(ord("\xae"), ord("\xff") + 1))
    cs = bs[:]; n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


U2B = {u: b for b, u in bytes_to_unicode().items()}


def tok_bytes(s):
    return bytes(U2B[c] for c in s)


def is_valid_utf8(b):
    try:
        b.decode("utf-8"); return True
    except UnicodeDecodeError:
        return False


def hf_counts(tok, docs, V):
    cnt = np.zeros(V, dtype=np.int64); n = 0
    for i in range(0, len(docs), 1000):
        for e in tok.encode_batch(docs[i:i + 1000], add_special_tokens=False):
            ids = np.asarray(e.ids, dtype=np.int64); n += ids.size
            cnt += np.bincount(ids, minlength=V)
    return cnt, n


def heldout_stats(tok, te, byte_level=True):
    nb = sum(len(t.encode("utf-8")) for t in te); nt = 0; rt = 0
    for i in range(0, len(te), 500):
        encs = tok.encode_batch(te[i:i + 500], add_special_tokens=False)
        for t, e in zip(te[i:i + 500], encs):
            nt += len(e.ids)
            if tok.decode(e.ids, skip_special_tokens=False) != t:
                rt += 1
    return {"heldout_bytes_per_token": round(nb / nt, 4), "heldout_tokens": nt, "roundtrip_fail_docs": rt,
            "heldout_docs": len(te)}


def profile(freqs, keys, leaves=None):
    """freqs: dict key -> train frequency for learned tokens."""
    f = np.array([freqs[k] for k in keys])
    out = {"learned_tokens": len(keys), "train_freq_eq0": int((f == 0).sum()),
           "train_freq_lt20": int((f < 20).sum()), "train_freq_lt100": int((f < 100).sum()),
           "pct_lt20": round(100 * (f < 20).mean(), 2), "pct_lt100": round(100 * (f < 100).mean(), 2),
           "median_train_freq": int(np.median(f))}
    if leaves is not None:
        lf = np.array([k in leaves for k in keys])
        out["lt20_leaves"] = int(((f < 20) & lf).sum())
        out["lt20_intermediate"] = int(((f < 20) & ~lf).sum())
        out["eq0_intermediate"] = int(((f == 0) & ~lf).sum())
        out["learned_leaves"] = int(lf.sum())
    return out


def audit_hf_bytelevel(path, tr, te, name):
    tok = Tokenizer.from_file(path)
    j = json.loads(open(path, encoding="utf-8").read())
    vocab = j["model"]["vocab"]; merges = j["model"]["merges"]
    V = tok.get_vocab_size(with_added_tokens=True)
    specials = {a["id"] for a in j["added_tokens"]}
    base = set(bytes_to_unicode().values())
    learned = sorted((i, s) for s, i in vocab.items() if i not in specials and s not in base)
    comps = set()
    for a, b in merges:
        comps.add(a); comps.add(b)
    leaves = {s for _, s in learned if s not in comps}
    t0 = time.time(); cnt, ntok = hf_counts(tok, tr, V); log(name, "train encode", round(time.time() - t0, 1), "s")
    freqs = {s: int(cnt[i]) for i, s in learned}
    prof = profile(freqs, [s for _, s in learned], leaves)
    partial = []; selftok_fail = []
    for i, s in learned:
        b = tok_bytes(s)
        if not is_valid_utf8(b):
            partial.append({"id": i, "bytes": b.hex(" "), "train_freq": int(cnt[i]), "leaf": s in leaves})
            continue
        ids = tok.encode(b.decode("utf-8"), add_special_tokens=False).ids
        if ids != [i]:
            selftok_fail.append({"id": i, "text": b.decode("utf-8"), "encodes_to": ids, "train_freq": int(cnt[i])})
    zero = [{"id": i, "text_or_hex": (tok_bytes(s).decode("utf-8") if is_valid_utf8(tok_bytes(s)) else tok_bytes(s).hex(" ")),
             "leaf": s in leaves} for i, s in learned if cnt[i] == 0]
    res = {"tokenizer": name, "file": os.path.relpath(path, HERE), "vocab_total": V, "train_tokens": ntok,
           "support_profile": prof,
           "partial_utf8_tokens": len(partial), "partial_utf8_list": partial,
           "partial_utf8_with_train_freq_0": sum(1 for p in partial if p["train_freq"] == 0),
           "selftok_fail_complete_utf8": len(selftok_fail), "selftok_fail_list": selftok_fail[:50],
           "zero_freq_examples": zero[:40]}
    res.update(heldout_stats(tok, te))
    return res


def sp_counts(sp, texts, V):
    cnt = np.zeros(V, dtype=np.int64); ntok = 0
    for i in range(0, len(texts), 2000):
        for ids in sp.encode(texts[i:i + 2000], num_threads=THREADS):
            a = np.asarray(ids, dtype=np.int64); ntok += a.size; cnt += np.bincount(a, minlength=V)
    return cnt, ntok


def audit_sp(path, tr_lines, te, name, tr_docs=None):
    """Support is counted on WHOLE training documents (what an LM sees) and, for reference, on the
    non-empty lines the pilot SentencePiece models were trained on. The two differ because a word after
    a newline gets no dummy-prefix U+2581 inside a whole document."""
    sp = spm.SentencePieceProcessor(model_file=path)
    V = sp.get_piece_size()
    normal = [i for i in range(V) if not (sp.is_control(i) or sp.is_unknown(i) or sp.is_byte(i))]
    t0 = time.time()
    cnt_lines, ntok_lines = sp_counts(sp, tr_lines, V)
    cnt, ntok = sp_counts(sp, tr_docs, V)
    log(name, "train encode (lines + docs)", round(time.time() - t0, 1), "s")
    multi = [i for i in normal if len(sp.id_to_piece(i).replace("\u2581", " ")) > 1]
    mset = set(multi)
    single = [i for i in normal if i not in mset]
    freqs = {i: int(cnt[i]) for i in normal}
    freqs_lines = {i: int(cnt_lines[i]) for i in normal}
    # self-tokenization with the dummy prefix switched off (a piece without U+2581 is word-internal)
    m = spb.ModelProto(); m.ParseFromString(open(path, "rb").read())
    m.normalizer_spec.add_dummy_prefix = False
    sp2 = spm.SentencePieceProcessor(model_proto=m.SerializeToString())
    fail = []
    for i in normal:
        txt = sp.id_to_piece(i).replace("\u2581", " ")
        ids = sp2.encode(txt)
        if ids != [i]:
            fail.append({"id": i, "piece": sp.id_to_piece(i), "encodes_to": ids, "train_freq": int(cnt[i])})
    nb = sum(len(t.encode("utf-8")) for t in te); nt = 0; rt = 0
    for t in te:
        ids = sp.encode(t); nt += len(ids); rt += sp.decode(ids) != t
    return {"tokenizer": name, "file": os.path.relpath(path, HERE), "vocab_total": V, "train_tokens": ntok,
            "support_counted_on": "whole training documents (as an LM sees them)",
            "support_profile_all_normal_pieces": profile(freqs, normal),
            "support_profile_multichar_pieces": profile(freqs, multi),
            "byte_pieces_used_in_train_docs": int(sum(cnt[i] for i in range(V) if sp.is_byte(i))),
            "train_tokens_lines": ntok_lines,
            "support_profile_all_normal_pieces_on_training_lines": profile(freqs_lines, normal),
            "support_profile_multichar_pieces_on_training_lines": profile(freqs_lines, multi),
            "single_char_pieces": len(single),
            "partial_utf8_tokens": 0, "note_partial": "pieces are Unicode strings; only the 256 <0xNN> byte pieces are sub-character, by design",
            "selftok_fail": len(fail), "selftok_fail_list": fail[:30],
            "heldout_bytes_per_token": round(nb / nt, 4), "roundtrip_fail_docs": rt, "heldout_docs": len(te)}


def train_bytelevel(tr, V, min_frequency=2):
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(P1), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    tok.decoder = decoders.ByteLevel()
    trn = trainers.BpeTrainer(vocab_size=V, min_frequency=min_frequency, show_progress=False,
                              initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), special_tokens=["<|endoftext|>"])
    t0 = time.time(); tok.train_from_iterator(tr, trainer=trn)
    return tok, time.time() - t0


def part_minfreq(tr):
    saved = json.loads(open(os.path.join(PILOT, "hf_bpe_P1_16000.json"), encoding="utf-8").read())
    rows = []
    for mf in (2, 20, 100, 400, 1000):
        tok, dt = train_bytelevel(tr, 16000, mf)
        j = json.loads(tok.to_str())
        mg = j["model"]["merges"]
        common = 0
        for a, b in zip(mg, saved["model"]["merges"]):
            if a != b: break
            common += 1
        row = {"min_frequency": mf, "train_s": round(dt, 1), "merges": len(mg), "vocab": len(j["model"]["vocab"]),
               "identical_to_saved_pilot_merges": mg == saved["model"]["merges"],
               "identical_vocab": j["model"]["vocab"] == saved["model"]["vocab"],
               "common_merge_prefix": common}
        if mf == 2:
            row["json_identical_to_saved_pilot_file"] = j == saved
        log("minfreq", row); rows.append(row)
    return rows


def part_a2(tr, te):
    """Char-level BPE (A2): P1 split, no ByteLevel; 256 <0xNN> byte-fallback tokens added afterwards."""
    tok = Tokenizer(models.BPE(byte_fallback=True))
    tok.pre_tokenizer = pre_tokenizers.Split(Regex(P1), behavior="isolated")
    trn = trainers.BpeTrainer(vocab_size=16000 - 256, min_frequency=2, show_progress=False,
                              special_tokens=["<|endoftext|>"], limit_alphabet=1000)
    t0 = time.time(); tok.train_from_iterator(tr, trainer=trn); dt = time.time() - t0
    j = json.loads(tok.to_str())
    old = sorted(j["model"]["vocab"].items(), key=lambda kv: kv[1])
    new_vocab = {}
    assert old[0] == ("<|endoftext|>", 0), old[0]
    new_vocab[old[0][0]] = 0  # <|endoftext|>
    for b in range(256):
        new_vocab["<0x%02X>" % b] = len(new_vocab)
    for s, _ in old[1:]:
        new_vocab[s] = len(new_vocab)
    j["model"]["vocab"] = new_vocab; j["model"]["byte_fallback"] = True
    j["decoder"] = {"type": "Sequence", "decoders": [{"type": "ByteFallback"}, {"type": "Fuse"}]}
    path = os.path.join(OUT, "a2_charbpe_bytefallback_P1_16000.json")
    open(path, "w", encoding="utf-8").write(json.dumps(j, ensure_ascii=False))
    tok = Tokenizer.from_file(path)
    V = tok.get_vocab_size(with_added_tokens=True)
    alphabet = {s for s in new_vocab if len(s) == 1}
    learned = sorted((i, s) for s, i in new_vocab.items() if i > 256 and s not in alphabet)
    comps = set()
    for a, b in j["model"]["merges"]:
        comps.add(a); comps.add(b)
    leaves = {s for _, s in learned if s not in comps}
    cnt, ntok = hf_counts(tok, tr, V)
    freqs = {s: int(cnt[i]) for i, s in learned}
    fail = []
    for i, s in learned:
        ids = tok.encode(s, add_special_tokens=False).ids
        if ids != [i]:
            fail.append({"id": i, "text": s, "encodes_to": ids})
    res = {"tokenizer": "A2 char-level BPE + byte_fallback, P1", "file": os.path.relpath(path, HERE), "train_s": round(dt, 1),
           "vocab_total": V, "alphabet_chars": len(alphabet), "train_tokens": ntok,
           "support_profile": profile(freqs, [s for _, s in learned], leaves),
           "partial_utf8_tokens": 0, "selftok_fail": len(fail), "selftok_fail_list": fail[:30],
           "byte_fallback_tokens_used_in_train": int(cnt[1:257].sum())}
    res.update(heldout_stats(tok, te))
    log("a2", {k: v for k, v in res.items() if k != "selftok_fail_list"})
    return res


def rebuild(j, drop):
    """Remove learned leaf tokens `drop` (token strings) and the merges that create them; re-index ids."""
    m = j["model"]
    keep = [(s, i) for s, i in sorted(m["vocab"].items(), key=lambda kv: kv[1]) if s not in drop]
    m["vocab"] = {s: k for k, (s, _) in enumerate(keep)}
    m["merges"] = [[a, b] for a, b in m["merges"] if (a + b) not in drop]
    for at in j["added_tokens"]:
        at["id"] = m["vocab"][at["content"]]
    return j


def part_prune(tr, te, m_extra, target=16000):
    tok, dt = train_bytelevel(tr, target + m_extra)
    j = json.loads(tok.to_str())
    base = set(bytes_to_unicode().values())
    it = 0; hist = []
    while True:
        tok = Tokenizer.from_str(json.dumps(j, ensure_ascii=False))
        V = tok.get_vocab_size(with_added_tokens=True)
        excess = V - target
        cnt, _ = hf_counts(tok, tr, V)
        vocab = j["model"]["vocab"]
        comps = set()
        for a, b in j["model"]["merges"]:
            comps.add(a); comps.add(b)
        special = {a["content"] for a in j["added_tokens"]}
        rank = {a + b: r for r, (a, b) in enumerate(j["model"]["merges"])}
        leaves = [(int(cnt[i]), -rank[s], s) for s, i in vocab.items()
                  if s not in base and s not in special and s not in comps]
        leaves.sort()
        hist.append({"iter": it, "vocab": V, "excess": excess,
                     "leaves_lt20": sum(1 for f, _, _ in leaves if f < MIN_SUPPORT)})
        log("prune m=%d" % m_extra, hist[-1])
        if excess <= 0:
            break
        k = min(excess, max(100, math.ceil(excess / 2)))
        drop = {s for _, _, s in leaves[:k]}
        j = rebuild(j, drop); it += 1
    path = os.path.join(OUT, "a1_leafpruned_P1_%d_from_%d.json" % (target, target + m_extra))
    open(path, "w", encoding="utf-8").write(json.dumps(j, ensure_ascii=False))
    res = audit_hf_bytelevel(path, tr, te, "A1 BPE-P1 leaf-pruned to %d from %d" % (target, target + m_extra))
    res["prune_iterations"] = hist; res["train_s_initial"] = round(dt, 1)
    return res


def main():
    parts = sys.argv[1:] or ["audit", "minfreq", "a2", "prune"]
    tr, te = load_split()
    log("pilot split: train", len(tr), "heldout", len(te))
    if "audit" in parts:
        rows = []
        for V in (16000, 32000):
            r = audit_hf_bytelevel(os.path.join(PILOT, "hf_bpe_P1_%d.json" % V), tr, te, "A1 byte-level BPE-P1 %d" % V)
            log({k: v for k, v in r.items() if not k.endswith("_list") and k != "zero_freq_examples"}); rows.append(r)
        tr_lines = [ln for t in tr for ln in t.split("\n") if ln.strip()]
        for V in (16000, 32000):
            r = audit_sp(os.path.join(PILOT, "sp_unigram_%d.model" % V), tr_lines, te, "A4 SP Unigram %d" % V, tr)
            log({k: v for k, v in r.items() if not k.endswith("_list")}); rows.append(r)
        json.dump({"split": "pilot md5(uid)%20 (train 17,333 / held-out 950 docs); not group-disjoint",
                   "min_support": MIN_SUPPORT, "rows": rows},
                  open(os.path.join(OUT, "gate_audit_pilot.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    if "minfreq" in parts:
        json.dump({"what": "HF BpeTrainer min_frequency vs merges, BPE-P1, pilot train, target vocab 16000",
                   "rows": part_minfreq(tr)},
                  open(os.path.join(OUT, "minfreq_test.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    if "a2" in parts:
        json.dump(part_a2(tr, te), open(os.path.join(OUT, "a2_audit.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    if "prune" in parts:
        rows = [part_prune(tr, te, m) for m in (2000, 4000)]
        json.dump({"what": "leaf-prune-to-fixed-size: train BPE-P1 at 16000+m, then repeatedly delete the "
                           "lowest-train-frequency learned leaf tokens (and their merges) until 16000; frequencies "
                           "re-counted after each batch", "rows": rows},
                  open(os.path.join(OUT, "leafprune_test.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
