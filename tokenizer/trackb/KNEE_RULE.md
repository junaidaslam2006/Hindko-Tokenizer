# Track B: knee rule for the number of new tokens k (pre-registered)

Written 2026-09-26, 17:15 local time, **before any dev metric of an extended tokenizer was computed**. At that
moment only the Qwen3 extension had been built (training on train_D1 only) and its merges printed; no dev_strict
encoding with an extended tokenizer existed. `evaluate.py` and `report.py` apply this rule mechanically. The file's
sha256 is recorded in `results/knee.json`.

Grid: k in {1,024, 2,048, 4,096, 8,192, 16,384} new tokens (PLAN 9.2, "1k ... 16k", read as powers of two like the
PLAN's own vocabulary sizes), plus k = 0 (the unextended base) as the reference point.

1. **Eligibility** of a k (all must hold):
   - G1: `decode(encode(doc)) == doc` for all 836 dev_strict documents.
   - G2: 0 new whole-character tokens fail self-tokenization, `encode(decode([id])) == [id]` (partial-UTF-8
     byte tokens are exempt and reported, as in PLAN 4.3).
   - Non-interference: the extended tokenizer encodes every document of the English, Python-code and
     other-script check sets with exactly the base's ids.
   - Equivalence: the HF-native encoding equals the reference continued-BPE encoding (base encoding, then the new
     merges in learned order) on every dev_strict document.
2. **Support cap** (PLAN 9.2: "prefer the k at which the share of new tokens with < 100 train occurrences stays
   small"): at most **10%** of the k new tokens may occur fewer than 100 times when train_D1 is encoded with the
   extended tokenizer itself.
3. **Compression knee** (Kneedle, Satopaa et al. 2011, on a linear k axis, because embedding rows cost linearly in
   k): with x = k / 16,384 and y = (1 - T_k / T_0) / (1 - T_16384 / T_0), where T_k is the number of tokens of
   dev_strict under the k-extension, the knee is argmax over the grid of (y - x).
4. **Choice:** k* = min(Kneedle knee, largest eligible k that meets the support cap). If no k meets the cap,
   k* = 1,024 and the report says so.

The thresholds (10%, < 100 occurrences) and the linear axis are choices, not measurements; the full curves are
published so that a user with other priorities can pick another k from `sweep/`.
