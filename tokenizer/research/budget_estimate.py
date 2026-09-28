# -*- coding: utf-8 -*-
"""Wall-clock budget for PLAN.md Stages 3-4 from MEASURED concurrent throughput (added 2026-09-26
after review). Inputs: bench_concurrent.json (3 simultaneous 1-thread processes = one 3-seed wave),
split_facts.json (split sizes), pilot bytes/token. The 8k bytes/token is measured here by training
HF BPE-P1 at 8,192 on the pilot split (pilot/hf_bpe_P1_8192.json). Writes budget_estimate.json.
"""
import hashlib, json, os, time
os.environ["RAYON_NUM_THREADS"] = "3"
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, Regex

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = r"F:\Hindko\hindko_dataset_permissive.jsonl"
P1 = r" ?[\p{L}\p{M}\x{200C}\x{200D}]+| ?\p{N}| ?[^\s\p{L}\p{N}\p{M}]+|\s+(?!\S)|\s+"


def bpt_8k():
    tr, te = [], []
    for line in open(SRC, encoding="utf-8"):
        r = json.loads(line)
        (te if int(hashlib.md5(str(r["uid"]).encode()).hexdigest(), 16) % 20 == 0 else tr).append(r["text"])
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Split(Regex(P1), behavior="isolated"),
                                                 pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(tr, trainer=trainers.BpeTrainer(vocab_size=8192, min_frequency=2, show_progress=False,
                            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), special_tokens=["<|endoftext|>"]))
    tok.save(os.path.join(HERE, "pilot", "hf_bpe_P1_8192.json"))
    nb = sum(len(t.encode("utf-8")) for t in te)
    nt = sum(len(e.ids) for e in tok.encode_batch(te, add_special_tokens=False))
    return round(nb / nt, 3)


def main():
    bench = json.load(open(os.path.join(HERE, "bench_concurrent.json")))
    sf = json.load(open(os.path.join(HERE, "split_facts.json"), encoding="utf-8"))
    rate = {(r["d"], r["V"]): (r["conc3_train_tok_s_mean"], r["conc3_eval_tok_s_mean"]) for r in bench["results"]}
    dev_bytes = sf["split_sizes"]["validation/strict"]["bytes_MB"] * 1e6
    train_bytes_full = sf["split_sizes"]["train/permissive"]["bytes_MB"] * 1e6
    bpt = {8192: bpt_8k(), 16384: 6.009, 32768: 6.318}   # 16k/32k: pilot 16,000/32,000 held-out values
    OVERHEAD_MIN = 1.0   # model init, loading pre-encoded token streams, checkpoints (estimate)

    def wave(d, V, train_bytes, bytes_per_token, subset_evals=3, subset_bytes=0.5e6):
        tr_tok_s, ev_tok_s = rate[(d, V)]
        train_min = train_bytes / bytes_per_token / tr_tok_s / 60
        # full strict dev, windows of ctx with stride ctx/2 -> ~2 forward positions per token
        eval_min = (2 * dev_bytes + subset_evals * 2 * subset_bytes) / bytes_per_token / ev_tok_s / 60
        return round(train_min, 1), round(eval_min, 1), round(train_min + eval_min + OVERHEAD_MIN, 1)

    # Stage 3 screening waves, in PLAN 2.2 rank order (bytes/token of unbuilt candidates = conservative stand-ins)
    s3 = [("LR sweep on A1-16k (3 LRs x 1 seed)", 16384, bpt[16384]),
          ("1 A1 BPE-P1 16k (baseline)", 16384, bpt[16384]),
          ("2 SuperBPE 16k t/T=0.9 (A1 bytes/token used: upper bound on time)", 16384, bpt[16384]),
          ("3 Unigram 16k (pilot SP value)", 16384, 6.168),
          ("4 MinGram 16k (Unigram value used; estimate)", 16384, 6.168),
          ("5 SuperBPE 16k t/T=0.8 (A1 value used: upper bound)", 16384, bpt[16384]),
          ("6a A1 BPE-P1 8k", 8192, bpt[8192]),
          ("6b A1 BPE-P1 32k", 32768, bpt[32768]),
          ("7 PickyBPE 16k (A1 value used)", 16384, bpt[16384])]
    rows3 = []
    for name, V, b in s3:
        t, e, w = wave(128, V, 10e6, b)
        rows3.append({"wave": name, "V": V, "bytes_per_token": b, "train_min": t, "eval_min": e, "wave_min": w})
    tot3 = round(sum(r["wave_min"] for r in rows3) / 60, 2)
    # Stage 4: d=192 on the full permissive train split, 1 epoch
    rows4 = []
    for name, V, b in [("A1 16k baseline", 16384, bpt[16384]), ("finalist 16k", 16384, bpt[16384]),
                       ("finalist 32k (if any)", 32768, bpt[32768]), ("finalist 8k (if any)", 8192, bpt[8192])]:
        t, e, w = wave(192, V, train_bytes_full, b)
        rows4.append({"wave": name, "V": V, "train_min": t, "eval_min": e, "wave_min": w})
    out = {"created": time.strftime("%Y-%m-%d %H:%M:%S"),
           "inputs": {"throughput": "bench_concurrent.json (per-process tok/s with 3 simultaneous 1-thread processes)",
                      "rates_train_eval_tok_s": {"d%d_V%d" % k: v for k, v in rate.items()},
                      "dev_strict_bytes": dev_bytes, "train_permissive_bytes": train_bytes_full,
                      "bytes_per_token": bpt, "screening_train_bytes": 10e6, "overhead_min_per_wave": OVERHEAD_MIN},
           "stage3_waves": rows3, "stage3_total_h": tot3,
           "stage3_total_h_without_lr_wave": round(sum(r["wave_min"] for r in rows3[1:]) / 60, 2),
           "stage4_waves": rows4,
           "note": "machine shared with other agents; throughput varied by up to ~25% between measurements "
                   "(e.g. d=192/16k concurrent: 670 tok/s here vs 832 in the reviewer's run)"}
    json.dump(out, open(os.path.join(HERE, "budget_estimate.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
