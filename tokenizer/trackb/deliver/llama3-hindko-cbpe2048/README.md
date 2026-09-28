# llama3-hindko-cbpe2048

The **llama-3** tokenizer (Llama 3/3.1/3.2/3.3 (also Alif-1.0, Qalb-1.0)) extended with **2,048 Hindko tokens** by continued
BPE (Purason et al. 2026, reimplemented from the paper) on the Hindko train split. Built for continued pretraining
(CPT) of a llama-3 checkpoint on Hindko; details, curves and the allowed claims are in `TRACKB.md` of the Track B
work folder.

- New token ids: 128,256 … 130,303; `len(tokenizer)` = 130,304. All special
  tokens keep their ids. The new tokens are ordinary BPE model tokens with merges (not added tokens).
- Text in which no new merge applies is encoded exactly as by the base tokenizer (verified on English, Python code
  and other-script check sets, see EXTENSION.json `non_interference`). Arabic-script text (Hindko, Urdu) gets
  fewer tokens; every new token is a concatenation of consecutive base tokens.
- Hindko dev_strict: 2.736 → 5.470 bytes/token
  (50.0% fewer tokens), fertility 2.981 → 1.504, STRR
  19.0% → 64.3%.
- License: the base tokenizer's (llama3.1). Base: NousResearch/Meta-Llama-3.1-8B-Instruct (byte-identical mirror of meta-llama/Llama-3.1-8B-Instruct).

## Use

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("llama3-hindko-cbpe2048")        # this folder
```

The model's embedding matrix must be resized to at least 130,304 rows and the new rows initialised
before CPT: `python init_embeddings.py --model <base checkpoint> --tokenizer <this folder> --out <dir>`
(reference implementation, unit-tested on tiny random-weight models only; see TRACKB.md).

Files and sha256: `EXTENSION.json`. New tokens with their base-token decompositions and frequencies: `new_tokens.jsonl`.
