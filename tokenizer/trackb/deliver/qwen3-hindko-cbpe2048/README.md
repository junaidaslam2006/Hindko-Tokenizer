# qwen3-hindko-cbpe2048

The **qwen-3** tokenizer (Qwen3 (0.6B-235B), Qwen2/2.5 share the encodings) extended with **2,048 Hindko tokens** by continued
BPE (Purason et al. 2026, reimplemented from the paper) on the Hindko train split. Built for continued pretraining
(CPT) of a qwen-3 checkpoint on Hindko; details, curves and the allowed claims are in `TRACKB.md` of the Track B
work folder.

- New token ids: 151,669 … 153,716; `len(tokenizer)` = 153,717. All special
  tokens keep their ids. The new tokens are ordinary BPE model tokens with merges (not added tokens).
- Text in which no new merge applies is encoded exactly as by the base tokenizer (verified on English, Python code
  and other-script check sets, see EXTENSION.json `non_interference`). Arabic-script text (Hindko, Urdu) gets
  fewer tokens; every new token is a concatenation of consecutive base tokens.
- Hindko dev_strict: 2.684 → 5.162 bytes/token
  (48.0% fewer tokens), fertility 2.998 → 1.595, STRR
  3.1% → 59.9%.
- License: the base tokenizer's (apache-2.0). Base: Qwen/Qwen3-0.6B.

## Use

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("qwen3-hindko-cbpe2048")        # this folder
```

The model's embedding matrix must be resized to at least 153,717 rows and the new rows initialised
before CPT: `python init_embeddings.py --model <base checkpoint> --tokenizer <this folder> --out <dir>`
(reference implementation, unit-tested on tiny random-weight models only; see TRACKB.md).

Files and sha256: `EXTENSION.json`. New tokens with their base-token decompositions and frequencies: `new_tokens.jsonl`.
