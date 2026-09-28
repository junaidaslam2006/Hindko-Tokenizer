# gemma4-hindko-cbpe2048

The **gemma-4** tokenizer (Gemma 4 (E2B-31B); same pieces and merges as Gemma 3, different control tokens) extended with **2,048 Hindko tokens** by continued
BPE (Purason et al. 2026, reimplemented from the paper) on the Hindko train split. Built for continued pretraining
(CPT) of a gemma-4 checkpoint on Hindko; details, curves and the allowed claims are in `TRACKB.md` of the Track B
work folder.

- New token ids: 262,144 … 264,191; `len(tokenizer)` = 264,192. All special
  tokens keep their ids. The new tokens are ordinary BPE model tokens with merges (not added tokens).
- Text in which no new merge applies is encoded exactly as by the base tokenizer (verified on English, Python code
  and other-script check sets, see EXTENSION.json `non_interference`). Arabic-script text (Hindko, Urdu) gets
  fewer tokens; every new token is a concatenation of consecutive base tokens.
- Hindko dev_strict: 4.640 → 5.995 bytes/token
  (22.6% fewer tokens), fertility 1.765 → 1.357, STRR
  49.6% → 74.0%.
- License: the base tokenizer's (apache-2.0). Base: google/gemma-4-E2B-it.

## Use

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("gemma4-hindko-cbpe2048")        # this folder
```

The model's embedding matrix must be resized to at least 264,192 rows and the new rows initialised
before CPT: `python init_embeddings.py --model <base checkpoint> --tokenizer <this folder> --out <dir>`
(reference implementation, unit-tested on tiny random-weight models only; see TRACKB.md).

Files and sha256: `EXTENSION.json`. New tokens with their base-token decompositions and frequencies: `new_tokens.jsonl`.
