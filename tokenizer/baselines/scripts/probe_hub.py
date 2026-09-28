"""Probe the Hugging Face hub (anonymous, no login) for candidate tokenizer repos.

Records: commit sha, gated flag, licence, and which tokenizer files exist (with sizes).
Output: baselines/scripts/_probe_hub.json  (metadata only; nothing is downloaded here).
"""
import json
import os
import sys

os.environ.setdefault("HF_HOME", r"F:\Hindko\_tokenizer\hf_cache")
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

from huggingface_hub import HfApi

TOK_FILES = {
    "tokenizer.json", "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json",
    "vocab.json", "merges.txt", "vocab.txt", "spiece.model", "sentencepiece.bpe.model",
    "tiktoken.model", "tekken.json", "added_tokens.json", "o200k_base.tiktoken",
    "tokenizer.tok.json", "sentencepiece.model", "generation_config.json", "config.json",
}

api = HfApi()


def probe(repo):
    out = {"repo": repo}
    try:
        i = api.model_info(repo, files_metadata=True)
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"
        return out
    out["sha"] = i.sha
    out["gated"] = i.gated
    out["private"] = i.private
    lic = None
    if i.card_data is not None:
        lic = i.card_data.get("license")
        if lic in ("other", None) and i.card_data.get("license_name"):
            lic = f"other:{i.card_data.get('license_name')}"
    if lic is None:
        lic = next((t.split(":", 1)[1] for t in (i.tags or []) if t.startswith("license:")), None)
    out["license"] = lic
    out["last_modified"] = str(i.last_modified)
    out["created_at"] = str(getattr(i, "created_at", None))
    files = {}
    for s in i.siblings or []:
        base = s.rfilename.split("/")[-1]
        if base in TOK_FILES or base.endswith((".tiktoken", ".model")) or "token" in base.lower():
            files[s.rfilename] = {"size": s.size, "blob_id": s.blob_id,
                                  "lfs_sha256": (s.lfs.sha256 if s.lfs else None)}
    out["files"] = dict(sorted(files.items()))
    return out


def main():
    repos = [l.strip() for l in open(sys.argv[1], encoding="utf-8") if l.strip() and not l.startswith("#")]
    res = []
    for r in repos:
        p = probe(r)
        res.append(p)
        flag = p.get("error") or f"gated={p['gated']} lic={p['license']} " + " ".join(f"{k}:{v['size']}" for k, v in p['files'].items() if 'config' not in k)
        print(f"{r}: {flag}", flush=True)
    outp = sys.argv[2]
    json.dump(res, open(outp, "w", encoding="utf-8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
