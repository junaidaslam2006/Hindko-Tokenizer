"""Usability check: run every ```python block of hf_upload/README.md Quickstart standalone (each in a fresh
process) and all in sequence, with the private repo id redirected to the local hf_upload folder."""
import json, re, subprocess, sys, os
from pathlib import Path

OUT = Path(r"F:\Hindko\hf_upload")
readme = (OUT / "README.md").read_text(encoding="utf-8")
qs = readme[readme.index("## Quickstart"):readme.index("## Intended uses")]
blocks = re.findall(r"```python\n(.*?)```", qs, re.S)

PRE = f"""
import os, huggingface_hub
LOCAL = {str(OUT)!r}
def _dl(repo_id, filename, **kw):
    return os.path.join(LOCAL, filename)
huggingface_hub.hf_hub_download = _dl
import tokenizers
_ff = tokenizers.Tokenizer.from_file
class _T:
    @staticmethod
    def from_pretrained(repo, **kw):
        return _ff(os.path.join(LOCAL, "tokenizer.json"))
tokenizers.Tokenizer.from_pretrained = _T.from_pretrained
"""


def prep(code):
    return code.replace('"junaid008/hindko-tokenizer"', "LOCAL").replace(', token="hf_..."', "")


def run(code):
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1", TRANSFORMERS_VERBOSITY="error", HF_HUB_OFFLINE="1")
    r = subprocess.run([sys.executable, "-c", PRE + code], capture_output=True, text=True, encoding="utf-8",
                       env=env, timeout=300)
    err = (r.stderr.strip().splitlines() or [""])[-1]
    return r.returncode == 0, err, r.stdout


res = {"n_blocks": len(blocks), "standalone": []}
for i, b in enumerate(blocks):
    ok, err, _ = run(prep(b))
    res["standalone"].append({"block": i + 1, "first_line": b.splitlines()[0], "ok": ok, "error": None if ok else err})

# sequence run + extra assertions (chat output vs README comment; sp_encode == tokenizer.json on test-like text)
chat_expected = re.search(r"# ('<\|bos\|>.*?')\n", qs).group(1)
seq = "\n".join(prep(b) for b in blocks)
seq = seq.replace("tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)",
                  "_chat = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)")
seq += f"""
import ast
assert _chat == ast.literal_eval({chat_expected!r}), repr(_chat)
from tokenizers import Tokenizer as _Tk
_tk = _Tk.from_file(os.path.join(LOCAL, "tokenizer.json"))
_s = text + "\\n" + text + "\\n\\n" + "سال 2024 وچ ۲۰۲۴"
assert sp_encode(_s) == _tk.encode(_s).ids
assert _tk.encode("2024").tokens == ["\u2581", "2", "0", "2", "4"], _tk.encode("2024").tokens
assert _tk.encode("۲۰۲۴").tokens == ["\u2581۲۰۲۴"]
print("normalized ids:", len(ids), "->", repr(tok.decode(ids)))
print("SEQ_OK")
"""
ok, err, out = run(seq)
res["sequence"] = {"ok": ok and "SEQ_OK" in out, "error": None if ok else err, "stdout": out.strip()}
print(json.dumps(res, ensure_ascii=False, indent=1))
