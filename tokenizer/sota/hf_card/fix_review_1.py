"""Apply review-1 wording fixes to README.template.md, render_readme.py SOURCES map and build_hf_upload.py,
and patch hf_upload/eval/competitors_summary.json method.metrics in place. Idempotent-safe: each old string
must occur exactly once (or the new string must already be present)."""
import json
from pathlib import Path

HERE = Path(r"F:\Hindko\_tokenizer\sota\hf_card")
OUT = Path(r"F:\Hindko\hf_upload")


def patch(path, pairs):
    t = path.read_text(encoding="utf-8")
    for old, new in pairs:
        if new in t and old not in t:
            continue
        n = t.count(old)
        assert n == 1, f"{path.name}: {n} matches for {old[:70]!r}"
        t = t.replace(old, new)
    path.write_text(t, encoding="utf-8", newline="\n")
    print("patched", path.name)


METRICS = ("bytes/token = UTF-8 bytes / tokens; fertility = harness definition: sum over whitespace words of the "
           "number of tokens overlapping the word, divided by words (standalone '\u2581' and newline tokens belong "
           "to no word; the plain ratio tokens / whitespace words is reported separately as tokens_per_word, "
           "released 1.161 vs fertility 1.135); every tokenizer scored with the same definition; each tokenizer "
           "with its own native encoder, no BOS/EOS/CLS (eval/harness.py)")

# ---------------------------------------------------------------- README template
patch(HERE / "README.template.md", [
    # major: intro selection wording (AMENDMENT_1 s5)
    ("and held-out bits per byte decided, under a pre-registered protocol with a sealed test split.",
     "and held-out bits per byte decided under a pre-registered protocol with a sealed test split; this tokenizer "
     "is in the statistically tied top set and was chosen within it by a documented amendment (the pre-registered "
     "pick was MinGram 48k)."),
    # major: citation note
    ("Selected by held-out bits-per-byte of language models under a pre-registered protocol}",
     "In the statistically tied top set by held-out bits-per-byte of language models under a pre-registered "
     "protocol; chosen within that set by a documented amendment (the pre-registered pick was MinGram 48k)}"),
    # minor: Amendment 1 timing
    ("Amendment 1, written before any test number existed, chose",
     "Amendment 1, written before any test LM number existed, chose"),
    # minor: finalists composition
    ("18 finalists (17 ranked: 12 pre-registered and 6 built in a second, exploratory round)",
     "18 finalists (17 ranked): 12 in the first round (10 pre-registered plus 2 reference arms added before the "
     "first LM run) and 6 built in a second, exploratory round;"),
    # minor: fertility label
    ("| **Tokens per word (fertility)** | **1.135** on the test split |",
     "| **Fertility (tokens per word, harness definition)** | **1.135** on the test split (plain tokens / "
     "whitespace words: 1.161) |"),
    # minor: 2024 token count
    ("so `\u06f2\u06f0\u06f2\u06f4` is one token while `2024` is four.",
     "so `\u06f2\u06f0\u06f2\u06f4` is one token while `2024` at a word start is five "
     "(a standalone `\u2581` plus four digits)."),
    # minor: split exception
    ("assigns whole newspaper editions, books and web sites to one side.",
     "assigns whole newspaper editions, books and web sites to one side, with one exception: 230 leak-prone "
     "records from 40 evaluation groups were moved to train, so those groups are not 100 % held out."),
    # minor: standalone snippets
    ("tk = Tokenizer.from_pretrained(\"junaid008/hindko-tokenizer\", token=\"hf_...\")\nenc = tk.encode(text)",
     "tk = Tokenizer.from_pretrained(\"junaid008/hindko-tokenizer\", token=\"hf_...\")\n"
     "text = \"\u0627\u06cc\u06c1\u06c1 \u0628\u06c1\u062a \u06a9\u06c1\u0679 \u0644\u0648\u06a9 "
     "\u062c\u0627\u0646\u0691\u062f\u06cc\u0646 \u06a9\u06c1 \u0688\u0631\u0627\u0645\u06d2 \u062f\u06cc "
     "\u067e\u06cc\u062f\u0627\u0626\u0634 \u06a9\u0633\u0631\u0627\u06ba \u062a\u06d2 \u06a9\u062a\u06be\u06d2 "
     "\u06c1\u0648\u0626\u06cc \u0622\u0626\u06cc\u06d4\"\n"
     "enc = tk.encode(text)"),
])

# the normalize block: fetch quickstart.py from the private repo, define tok, literal input
t = (HERE / "README.template.md").read_text(encoding="utf-8")
i = t.index("from quickstart import normalize, fold_arabic_yeh_kaf")
s = t.rindex("```python", 0, i)
e = t.index("```", i)
old_block = t[s:e + 3]
new_block = (
    "```python\n"
    "import os, sys\n"
    "from huggingface_hub import hf_hub_download\n"
    "from transformers import AutoTokenizer\n"
    "\n"
    "qs = hf_hub_download(\"junaid008/hindko-tokenizer\", \"examples/quickstart.py\", token=\"hf_...\")\n"
    "sys.path.insert(0, os.path.dirname(qs))\n"
    "from quickstart import normalize, fold_arabic_yeh_kaf\n"
    "\n"
    "tok = AutoTokenizer.from_pretrained(\"junaid008/hindko-tokenizer\", token=\"hf_...\")\n"
    "raw_text = \"\u0627\u06cc\u06c1\u06c1  \u0628\u06c1\u062a \u06a9\u06c1\u0679 \u0644\u0648\u06a9\u200b\u06d4\\r\\n\"   "
    "# any input: double space, ZWSP, CRLF are cleaned\n"
    "ids = tok(normalize(raw_text))[\"input_ids\"]\n"
    "```"
)
if old_block != new_block:
    assert "raw_text" in old_block and old_block.count("\n") <= 4, old_block
    t = t[:s] + new_block + t[e + 3:]
    (HERE / "README.template.md").write_text(t, encoding="utf-8", newline="\n")
    print("patched normalize block")

# ---------------------------------------------------------------- SOURCES map in render_readme.py
patch(HERE / "render_readme.py", [
    ("- 18 LM candidates, 17 ranked, 12 + 6: release README s6; analysis/DECISION.md",
     "- 18 LM candidates, 17 ranked, 12 (10 pre-registered + 2 reference arms added before the first LM run) + 6 "
     "exploratory: eval/dev_lm_confirm_ranking.csv -> origin; release README s6; analysis/DECISION.md\n"
     "- Released = tied top set, chosen by amended rule; pre-registered pick MinGram 48k: analysis/AMENDMENT_1.md s5\n"
     "- plain tokens/word 1.161 (199,390 / 171,769): release eval/test_intrinsic_released.json -> overall.tokens_per_word\n"
     "- `2024` = 5 tokens at word start (standalone piece + 4 digits): measured with tokenizer.json; VOCAB_AUDIT.md\n"
     "- 230 leak-prone records from 40 evaluation groups moved to train: release README s4"),
])

# ---------------------------------------------------------------- build script + published JSON
patch(HERE / "build_hf_upload.py", [
    ('    "method": cf["method"],\n',
     '    "method": {**cf["method"], "metrics": ' + json.dumps(METRICS, ensure_ascii=False) + '},\n'),
])
p = OUT / "eval" / "competitors_summary.json"
d = json.loads(p.read_text(encoding="utf-8"))
d["method"]["metrics"] = METRICS
with open(p, "w", encoding="utf-8", newline="\n") as f:
    json.dump(d, f, ensure_ascii=False, indent=1)
    f.write("\n")
print("patched", p.name)
