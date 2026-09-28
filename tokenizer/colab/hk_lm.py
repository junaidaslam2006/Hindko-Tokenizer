# -*- coding: utf-8 -*-
"""hk_lm.py - the tokenizer LM arbiter of PLAN.md section 5, device-agnostic (torch + numpy only).

One self-contained module, used unchanged on the Colab GPU (via run_all.py) and on the local CPU:

  * GPT "hk-tiny" (PLAN 5): pre-LayerNorm, GELU MLP 4x, learned positions, tied input/output
    embeddings, no dropout, bias-free linear layers, chunked output-layer loss.
      screen  d=128 L=4 H=4   (pre-registered, Stage 3)          0.79M non-embedding params
      confirm d=192 L=4 H=4   (pre-registered, Stage 4)          1.77M
      large   d=384 L=6 H=6   ADDITION beyond the pre-registration (PLAN 8(ii) "strengthen the
                              claim"), report-only, never used by the decision rule   10.6M
  * byte-matched data (PLAN 5 "fairness controls"):
      - documents = pre-encoded token arrays + per-document UTF-8 byte counts (bundle);
      - document order = permutation of the document index drawn from (run seed, epoch) ALONE, so
        every candidate sees the documents in the same order;
      - each document is [BOS] + tokens + [EOT]; documents are joined with <|endoftext|>
        (if a tokenizer has no BOS distinct from EOT, the EOT before a document is its BOS);
      - budget in BYTES: steps = round(B / (8 * 1536)) is identical for every candidate; a
        candidate reads the first steps * 8 * ctx(c) + 1 tokens of its stream, i.e. the first
        ~B bytes of the shuffled stream (to within the rounding of ctx);
      - ctx_tokens(c) = round(1536 / train_bytes_per_token(c)); 8 sequences per optimiser step.
  * optimiser exactly as PLAN 5: AdamW(0.9, 0.95, eps 1e-8), weight decay 0.1 on matrices only,
    clip 1.0, 50 warm-up steps, cosine to 10 %, peak LR a parameter. GPU: bf16 autocast if the
    GPU has native bf16 (compute capability >= 8), else fp16 + GradScaler. Evaluation: fp32 always.
  * evaluation exactly as PLAN 5: every dev_strict document scored independently, windows of ctx
    tokens with stride ctx/2, BOS given, every token predicted including the document's EOT;
    per-document NLL (bits) and bytes stored; 25/50/75/100 % curve on a fixed 0.5 MB dev subset.
  * bpb = sum(bits) / sum(bytes) (PLAN 4.1). Bytes = UTF-8 bytes of the canonical text, no separators.

CLI (local reproduction on CPU):
  python hk_lm.py --bundle DIR --cand ID --stage screen --seed 1 --lr 3e-3 --device cpu --threads 2 --out r.json
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import platform
import random
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # before any CUDA context exists

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROTOCOL = "hk_lm/1.0"
PLAN_REF = "PLAN.md section 5 (arbiter), 4.1 (bpb), 6 (seeds / power), 7 (decision rule)"

# ---------------------------------------------------------------- protocol constants (PLAN 5)
CTX_BYTES = 1536                 # context in bytes -> ctx_tokens(c) = round(1536 / bytes_per_token(c))
SEQS_PER_STEP = 8                # sequences per optimiser step (~12 KB of text per step)
BYTES_PER_STEP = CTX_BYTES * SEQS_PER_STEP
WARMUP_STEPS = 50
FINAL_LR_FRAC = 0.10             # cosine decay to 10 % of the peak
BETAS = (0.9, 0.95)
ADAM_EPS = 1e-8
WEIGHT_DECAY = 0.1               # matrices (dim >= 2) only; LayerNorm gains/biases are not decayed
GRAD_CLIP = 1.0
LR_GRID = (1e-3, 3e-3, 6e-3)     # the pre-registered 3-point sweep on the baseline (screen budget, 1 seed)
SCREEN_BUDGET_BYTES = 10_000_000 # "10 MB" = 1e7 bytes (PLAN 5: ~1.66M tokens at 6.009 bytes/token)
CURVE_FRACS = (0.25, 0.50, 0.75) # + 1.00, taken from the full dev evaluation
INIT_STD = 0.02                  # GPT-2 init; residual projections scaled by 1/sqrt(2 L)
IGNORE = -100
PARITY_STEPS = 100
PARITY_LR = 3e-3

MODELS = {
    "screen":  {"d": 128, "L": 4, "H": 4, "preregistered": True,
                "role": "Stage 3 screening arbiter (PLAN 5)"},
    "confirm": {"d": 192, "L": 4, "H": 4, "preregistered": True,
                "role": "Stage 4 confirmation arbiter (PLAN 5)"},
    "large":   {"d": 384, "L": 6, "H": 6, "preregistered": False,
                "role": "ADDITION beyond the pre-registration: the larger-model strengthening step "
                        "suggested in PLAN 8(ii). Report-only; the decision rule (PLAN 7) never uses it."},
}

# stage -> training recipe. budget_bytes=None means `epochs` passes over the whole train split.
STAGES = {
    "screen":  {"model": "screen",  "budget_bytes": SCREEN_BUDGET_BYTES, "epochs": None, "curve": True,
                "eval": "all", "preregistered": True},
    "confirm": {"model": "confirm", "budget_bytes": None, "epochs": 1, "curve": True,
                "eval": "all", "preregistered": True},
    # 'large': 2 epochs. A 10.6M-non-embedding model on ~7.5M tokens is data-limited after one pass;
    # up to ~4 epochs of repeated data are nearly as good as fresh data (Muennighoff et al. 2023), and
    # the dev set is disjoint, so a second pass only adds signal. Every candidate gets 2 passes.
    "large":   {"model": "large",   "budget_bytes": None, "epochs": 2, "curve": True,
                "eval": "all", "preregistered": False},
    # fixed tiny config for the CPU <-> GPU sanity check (PARITY.json)
    "parity":  {"model": "screen",  "budget_bytes": PARITY_STEPS * BYTES_PER_STEP, "epochs": None,
                "curve": False, "eval": "parity", "preregistered": False},
}


# ---------------------------------------------------------------- small utilities
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


_SELF_SHA = {}


def this_file_sha256() -> str:
    p = os.path.abspath(__file__)
    if p not in _SELF_SHA:
        _SELF_SHA[p] = sha256_file(p)
    return _SELF_SHA[p]


def _finite_or_none(x):
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def write_json_atomic(path: str, obj) -> None:
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp%d" % os.getpid()
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, allow_nan=False)
    os.replace(tmp, path)


def lr_tag(lr: float) -> str:
    """Lossless, filename-safe tag of a peak LR (part of every result key).

    The shortest scientific notation that parses back to exactly the same float, so two different
    LRs never share a key: 3e-3 -> '3e-3', 2.5e-3 -> '2.5e-3', 2.9e-3 -> '2.9e-3', 5.8e-3 -> '5.8e-3'.
    The grid values keep their short form ('1e-3', '3e-3', '6e-3', '5e-4', '2e-3')."""
    x = float(lr)
    if not math.isfinite(x) or x <= 0.0:
        raise ValueError("peak LR must be a finite positive number, got %r" % (lr,))
    for p in range(17):              # %.16e (17 significant digits) always round-trips a double
        s = "%.*e" % (p, x)
        if float(s) == x:
            break
    mant, exp = s.split("e")
    return "%se%d" % (mant, int(exp))


def ctx_tokens(train_bytes_per_token: float) -> int:
    return int(round(CTX_BYTES / float(train_bytes_per_token)))


def steps_for_budget(budget_bytes: float) -> int:
    return max(1, int(round(float(budget_bytes) / BYTES_PER_STEP)))


def lr_at(step: int, total: int, peak: float) -> float:
    """Linear warm-up over 50 steps, then cosine from the peak to 10 % of it at the last step."""
    if step < WARMUP_STEPS:
        return peak * (step + 1) / WARMUP_STEPS
    prog = (step - WARMUP_STEPS) / max(1, total - WARMUP_STEPS)
    return peak * (FINAL_LR_FRAC + (1.0 - FINAL_LR_FRAC) * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog))))


def doc_order(n_docs: int, seed: int, epoch: int) -> np.ndarray:
    """Permutation of the document index from (seed, epoch) alone: identical for every candidate."""
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([int(seed), int(epoch), 0x484B])))
    return rng.permutation(n_docs)


def sequence_order(n_seq: int, seed: int) -> np.ndarray:
    """Order in which the packed sequences are presented (seeded; depends only on (seed, n_seq))."""
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([int(seed), 0x5E0, 0x484B])))
    return rng.permutation(n_seq)


# ---------------------------------------------------------------- bundle access
class Bundle:
    """Read-only view of an extracted bundle directory (written by build_bundle.py)."""

    def __init__(self, root: str, verify: bool = False):
        self.root = os.path.abspath(root)
        mp = os.path.join(self.root, "manifest.json")
        with open(mp, "rb") as f:
            raw = f.read()
        self.manifest_sha256 = sha256_bytes(raw)
        self.m = json.loads(raw.decode("utf-8"))
        self._arr = {}
        self._cand = {c["id"]: c for c in self.m["candidates"]}
        if verify:
            self.verify()

    # -- integrity
    def verify(self) -> int:
        bad = []
        for name, info in sorted(self.m["files"].items()):
            p = os.path.join(self.root, info["path"])
            if not os.path.isfile(p) or os.path.getsize(p) != info["bytes"] or sha256_file(p) != info["sha256"]:
                bad.append(name)
        if bad:
            raise RuntimeError("bundle files fail sha256/size verification: %s" % ", ".join(bad))
        return len(self.m["files"])

    # -- arrays and metadata
    def array(self, name: str) -> np.ndarray:
        if name not in self._arr:
            info = self.m["files"][name]
            self._arr[name] = np.load(os.path.join(self.root, info["path"]), allow_pickle=False)
        return self._arr[name]

    def text_file(self, name: str) -> str:
        info = self.m["files"][name]
        with open(os.path.join(self.root, info["path"]), encoding="utf-8") as f:
            return f.read()

    @property
    def candidate_ids(self):
        return [c["id"] for c in self.m["candidates"]]

    @property
    def baseline_id(self) -> str:
        return self.m["baseline_id"]

    def cand(self, cid: str) -> dict:
        return self._cand[cid]

    def train_bytes(self) -> np.ndarray:
        return self.array("train_bytes")

    def dev_bytes(self) -> np.ndarray:
        return self.array("dev_bytes")

    def dev_docs(self) -> list:
        if "_dev_docs" not in self._arr:
            self._arr["_dev_docs"] = json.loads(self.text_file("dev_docs"))
        return self._arr["_dev_docs"]

    def tokens(self, cid: str, split: str):
        c = self.cand(cid)
        return self.array(c["arrays"][split + "_tokens"]), self.array(c["arrays"][split + "_offsets"])

    def eval_doc_index(self, which) -> np.ndarray:
        n = len(self.dev_bytes())
        if which == "all":
            return np.arange(n, dtype=np.int64)
        if which == "subset":
            return self.array("dev_subset_idx").astype(np.int64)
        if which == "parity":
            return self.array("dev_parity_idx").astype(np.int64)
        if isinstance(which, (list, tuple)) and which and which[0] == "first":
            return np.arange(min(n, int(which[1])), dtype=np.int64)
        raise ValueError("unknown eval doc set %r" % (which,))


# ---------------------------------------------------------------- runtime / determinism
class Runtime:
    def __init__(self, device="auto", threads=None, deterministic=True, amp="auto", attn="auto", log=print):
        torch.set_flush_denormal(True)   # measured: without it CPU steps were 2-11x slower (PLAN 5)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if threads:
            torch.set_num_threads(int(threads))
        self.threads = torch.get_num_threads()
        self.gpu_name = None
        self.capability = None
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False     # fp32 evaluation must be real fp32
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            self.gpu_name = torch.cuda.get_device_name(self.device)
            self.capability = tuple(torch.cuda.get_device_capability(self.device))
            native_bf16 = self.capability[0] >= 8
            if amp == "auto":
                amp = "bf16" if native_bf16 else "fp16"
        else:
            if amp == "auto":
                amp = "fp32"
        if amp not in ("fp32", "bf16", "fp16"):
            raise ValueError("amp must be fp32|bf16|fp16")
        if amp == "fp16" and self.device.type != "cuda":
            raise ValueError("fp16 + GradScaler is a GPU-only mode")
        self.amp = amp
        self.amp_dtype = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}[amp]
        self.deterministic = "off"
        self.attn = "sdpa" if attn == "auto" else attn
        if deterministic:
            torch.use_deterministic_algorithms(True)
            self.deterministic = "strict"
            self._probe(log)

    def _probe(self, log):
        """Run one tiny forward/backward with the real model code under deterministic mode. If an
        op refuses (no deterministic kernel), fall back to math attention, then to warn-only."""
        for attempt in ("as_is", "math", "warn_only"):
            if attempt == "math":
                self.attn = "math"
            if attempt == "warn_only":
                torch.use_deterministic_algorithms(True, warn_only=True)
                self.deterministic = "warn_only"
            try:
                # exercise exactly the ops of a real run: train_step (autocast, chunked CE with
                # ignore_index, GradScaler, clipping, AdamW) and the fp32 evaluation path
                torch.manual_seed(0)
                m = GPT(64, 32, 1, 2, 16, attn=self.attn).to(self.device)
                opt = make_optimizer(m, 1e-3, self.device)
                scaler = _make_scaler() if self.amp == "fp16" else None
                buf = torch.randint(0, 64, (2, 17), device=self.device)
                y = buf[:, 1:].masked_fill(buf[:, 1:] == 3, IGNORE)
                train_step(m, opt, scaler, buf[:, :-1], y, self)
                with torch.no_grad():
                    h = m.hidden(buf[:, :-1]).reshape(-1, 32)
                    F.cross_entropy(h @ m.tok.weight.t(), y.reshape(-1), ignore_index=IGNORE,
                                    reduction="none").view(2, -1).double().sum(dim=1).cpu()
                return
            except RuntimeError as e:  # pragma: no cover - depends on the GPU/torch build
                log("[runtime] deterministic probe failed (%s): %s" % (attempt, str(e).splitlines()[0][:200]))
        raise RuntimeError("model cannot run on this device even in warn-only deterministic mode")

    def info(self) -> dict:
        return {"device": self.device.type, "gpu_name": self.gpu_name,
                "cuda_capability": list(self.capability) if self.capability else None,
                "amp_dtype": self.amp, "eval_dtype": "fp32", "deterministic": self.deterministic,
                "attn_impl": self.attn, "threads": self.threads, "torch": torch.__version__,
                "numpy": np.__version__, "python": platform.python_version(),
                "cuda": torch.version.cuda, "tf32": False if self.device.type == "cuda" else None,
                "flush_denormal": True}

    def ce_chunk(self) -> int:
        # chunked output-layer loss; 512 rows on CPU (PLAN 5), whole batch at once on GPU (same math)
        return 512 if self.device.type == "cpu" else 8192

    def eval_batch(self) -> int:
        return 16 if self.device.type == "cpu" else 64


def _autocast(device, amp_dtype):
    if amp_dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=amp_dtype)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)


# ---------------------------------------------------------------- model
class Block(nn.Module):
    def __init__(self, d, n_head, attn="sdpa"):
        super().__init__()
        self.n_head = n_head
        self.attn = attn
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.ln2 = nn.LayerNorm(d)
        self.fc = nn.Linear(d, 4 * d, bias=False)
        self.out = nn.Linear(4 * d, d, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        h = self.n_head
        q, k, v = self.qkv(self.ln1(x)).split(C, dim=2)
        q = q.view(B, T, h, C // h).transpose(1, 2)
        k = k.view(B, T, h, C // h).transpose(1, 2)
        v = v.view(B, T, h, C // h).transpose(1, 2)
        if self.attn == "math":
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(C // h))
            mask = torch.ones(T, T, dtype=torch.bool, device=x.device).triu(1)
            att = att.float().masked_fill(mask, float("-inf")).softmax(dim=-1).to(v.dtype)
            y = att @ v
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).reshape(B, T, C))
        return x + self.out(F.gelu(self.fc(self.ln2(x))))


class GPT(nn.Module):
    """Pre-LN GPT, learned positions, tied embeddings, no dropout (PLAN 5)."""

    def __init__(self, n_vocab, d, n_layer, n_head, n_ctx, attn="sdpa"):
        super().__init__()
        self.n_vocab, self.d, self.n_layer, self.n_ctx = n_vocab, d, n_layer, n_ctx
        self.tok = nn.Embedding(n_vocab, d)
        self.pos = nn.Embedding(n_ctx, d)
        self.blocks = nn.ModuleList([Block(d, n_head, attn) for _ in range(n_layer)])
        self.lnf = nn.LayerNorm(d)

    def init_weights(self, seed: int) -> None:
        """Deterministic, device-independent init from a CPU generator (so CPU and GPU runs start
        from bit-identical weights)."""
        g = torch.Generator().manual_seed(int(seed))
        with torch.no_grad():
            for name, p in self.named_parameters():
                if p.dim() >= 2:
                    std = INIT_STD / math.sqrt(2 * self.n_layer) if name.endswith(("proj.weight", "out.weight")) \
                        else INIT_STD
                    p.copy_((torch.randn(p.shape, generator=g) * std).to(p.device, p.dtype))
                elif name.endswith("weight"):
                    p.fill_(1.0)
                else:
                    p.zero_()

    def hidden(self, idx):
        T = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        for b in self.blocks:
            x = b(x)
        return self.lnf(x)

    def param_counts(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        emb = self.tok.weight.numel()
        pos = self.pos.weight.numel()
        return {"total": total, "token_embedding_tied": emb, "position_embedding": pos,
                "non_embedding": total - emb - pos}


def build_model(model_name: str, n_vocab: int, ctx: int, seed: int, attn: str = "sdpa") -> GPT:
    cfg = MODELS[model_name]
    _seed_everything(seed)
    m = GPT(n_vocab, cfg["d"], cfg["L"], cfg["H"], ctx, attn=attn)
    m.init_weights(seed)
    return m


def make_optimizer(model: GPT, lr: float, device) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [{"params": decay, "weight_decay": WEIGHT_DECAY},
              {"params": no_decay, "weight_decay": 0.0}]
    kw = {"lr": lr, "betas": BETAS, "eps": ADAM_EPS}
    if device.type == "cuda":
        try:
            return torch.optim.AdamW(groups, fused=True, **kw)
        except (TypeError, RuntimeError):
            pass
    return torch.optim.AdamW(groups, **kw)


def _make_scaler():
    try:
        return torch.amp.GradScaler("cuda")
    except (AttributeError, TypeError):  # older torch
        return torch.cuda.amp.GradScaler()


def train_step(model, opt, scaler, x, y, rt: Runtime):
    """One optimiser step with the chunked output-layer loss (hidden state detached, logits built
    chunk by chunk, gradient pushed back through the body once). Returns the loss as a 0-d tensor."""
    with _autocast(rt.device, rt.amp_dtype):
        h = model.hidden(x)
    h = h.reshape(-1, h.shape[-1])
    hd = h.detach().requires_grad_(True)
    W = model.tok.weight
    yv = y.reshape(-1)
    n_valid = (yv != IGNORE).sum().clamp_min(1).to(torch.float32)
    total = torch.zeros((), device=x.device, dtype=torch.float32)
    step = rt.ce_chunk()
    for i in range(0, yv.numel(), step):
        with _autocast(rt.device, rt.amp_dtype):
            lg = hd[i:i + step] @ W.t()
        loss = F.cross_entropy(lg.float(), yv[i:i + step], ignore_index=IGNORE, reduction="sum") / n_valid
        (scaler.scale(loss) if scaler is not None else loss).backward()
        total = total + loss.detach()
    h.backward(hd.grad)
    if scaler is not None:
        scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    if scaler is not None:
        scaler.step(opt)
        scaler.update()
    else:
        opt.step()
    opt.zero_grad(set_to_none=True)
    return total


# ---------------------------------------------------------------- data pipeline
def build_train_stream(bundle: Bundle, cid: str, seed: int, n_needed: int):
    """Token stream = documents in the seed's order (epoch 0, then epoch 1, ...), each as
    [BOS] doc [EOT] (or doc [EOT] after a leading EOT when BOS == EOT). Returns (stream int64,
    info dict with byte accounting for the first n_needed tokens)."""
    c = bundle.cand(cid)
    bos, eot = int(c["bos_id"]), int(c["eot_id"])
    toks, offs = bundle.tokens(cid, "train")
    dbytes = bundle.train_bytes().astype(np.int64)
    n_docs = len(dbytes)
    lens = np.diff(offs).astype(np.int64)
    extra = 2 if bos != eot else 1
    parts = [np.array([eot], dtype=np.int64)] if bos == eot else []
    total = len(parts)
    unit_end, unit_bytes = [], []
    epoch = 0
    while total < n_needed:
        order = doc_order(n_docs, seed, epoch)
        cum = total + np.cumsum(lens[order] + extra)
        k = int(np.searchsorted(cum, n_needed, side="left")) + 1   # docs needed in this epoch
        k = min(k, n_docs)
        for i in order[:k]:
            i = int(i)
            if bos != eot:
                parts.append(np.array([bos], dtype=np.int64))
            parts.append(toks[offs[i]:offs[i + 1]].astype(np.int64))
            parts.append(np.array([eot], dtype=np.int64))
        unit_end.append(cum[:k])
        unit_bytes.append(dbytes[order[:k]])
        total = int(cum[k - 1])
        epoch += 1
    stream = np.concatenate(parts)
    assert len(stream) == total >= n_needed
    ue = np.concatenate(unit_end)
    ub = np.concatenate(unit_bytes)
    starts = np.concatenate([[1 if bos == eot else 0], ue[:-1]])
    full = ue <= n_needed
    partial = (~full) & (starts < n_needed)
    frac = np.where(partial, (n_needed - starts) / np.maximum(1, ue - starts), 0.0)
    bytes_seen = float(ub[full].sum() + (ub * frac).sum())
    info = {"stream_tokens_built": int(len(stream)), "docs_fully_seen": int(full.sum()),
            "docs_partially_seen": int(partial.sum()), "epochs_touched": epoch,
            "bytes_seen": round(bytes_seen, 1),
            "epochs_seen": round(bytes_seen / float(dbytes.sum()), 6)}
    return stream, info


def eval_windows(seq_len_targets: int, ctx: int):
    """Windows (a, end, first_scored) over a document with m = seq_len_targets targets: inputs
    s[a:end] predict s[a+1:end+1]; positions >= first_scored (window-relative) are scored.
    Windows of ctx tokens, stride ctx/2; every target is scored exactly once, each after the first
    window with >= ctx/2 tokens of context (PLAN 5)."""
    stride = max(1, ctx // 2)
    m = seq_len_targets
    out, a, done = [], 0, 0
    while True:
        end = min(a + ctx, m)
        out.append((a, end, done - a))
        done = end
        if end >= m:
            return out
        a += stride


_EVAL_CACHE = {}


def _eval_batches(bundle: Bundle, cid: str, doc_idx: np.ndarray, ctx: int, batch: int):
    key = (bundle.manifest_sha256, cid, ctx, batch, hashlib.sha1(np.ascontiguousarray(doc_idx).tobytes()).hexdigest())
    if key in _EVAL_CACHE:
        return _EVAL_CACHE[key]
    c = bundle.cand(cid)
    bos, eot = int(c["bos_id"]), int(c["eot_id"])
    toks, offs = bundle.tokens(cid, "dev")
    rows = []   # (doc_pos, x[ctx], y[ctx])
    ntok = np.zeros(len(doc_idx), dtype=np.int64)
    for p, i in enumerate(doc_idx):
        i = int(i)
        s = np.concatenate([[bos], toks[offs[i]:offs[i + 1]].astype(np.int64), [eot]])
        m = len(s) - 1
        ntok[p] = m
        for a, end, first in eval_windows(m, ctx):
            x = np.zeros(ctx, dtype=np.int64)
            y = np.full(ctx, IGNORE, dtype=np.int64)
            L = end - a
            x[:L] = s[a:end]
            y[first:L] = s[a + 1 + first:end + 1]
            rows.append((p, x, y))
    batches = []
    for j in range(0, len(rows), batch):
        chunk = rows[j:j + batch]
        batches.append((np.array([r[0] for r in chunk], dtype=np.int64),
                        torch.from_numpy(np.stack([r[1] for r in chunk])),
                        torch.from_numpy(np.stack([r[2] for r in chunk]))))
    res = (batches, ntok, len(rows))
    if len(_EVAL_CACHE) > 8:
        _EVAL_CACHE.clear()
    _EVAL_CACHE[key] = res
    return res


@torch.no_grad()
def evaluate(model: GPT, bundle: Bundle, cid: str, doc_idx: np.ndarray, ctx: int, rt: Runtime) -> dict:
    """Per-document NLL in bits (fp32 forward, float64 accumulation), PLAN 5 windows."""
    batches, ntok, n_windows = _eval_batches(bundle, cid, doc_idx, ctx, rt.eval_batch())
    nats = np.zeros(len(doc_idx), dtype=np.float64)
    W = model.tok.weight
    chunk = 512 if rt.device.type == "cpu" else 4096
    for pos, xb, yb in batches:
        x = xb.to(rt.device)
        y = yb.to(rt.device).reshape(-1)
        h = model.hidden(x).reshape(-1, W.shape[1])
        nll = torch.empty(y.numel(), dtype=torch.float32, device=rt.device)
        for i in range(0, y.numel(), chunk):
            nll[i:i + chunk] = F.cross_entropy(h[i:i + chunk] @ W.t(), y[i:i + chunk],
                                               ignore_index=IGNORE, reduction="none")
        per_win = nll.view(x.shape[0], -1).double().sum(dim=1).cpu().numpy()
        for p, v in zip(pos.tolist(), per_win.tolist()):
            nats[p] += v
    bits = nats / math.log(2.0)
    dbytes = bundle.dev_bytes()[doc_idx].astype(np.int64)
    sb = int(dbytes.sum())
    return {"n_docs": int(len(doc_idx)), "n_windows": int(n_windows), "bits": bits, "bytes": dbytes,
            "ntok": ntok, "sum_bits": float(bits.sum()), "sum_bytes": sb,
            "bpb": float(bits.sum() / sb) if sb else float("nan")}


def bpb_of(bits, nbytes) -> float:
    """PLAN 4.1: bpb = sum over docs of bits / sum over docs of UTF-8 bytes."""
    return float(np.sum(np.asarray(bits, dtype=np.float64)) / np.sum(np.asarray(nbytes, dtype=np.int64)))


@torch.no_grad()
def brute_force_doc_bits(model: GPT, seq: np.ndarray, ctx: int, rt: Runtime) -> float:
    """Independent check: score s[1..m] token by token, one forward pass per target, with the
    context the protocol assigns to target j: s[a_j:j], a_j = stride * max(0, ceil((j - ctx)/stride)).
    For a document with m <= ctx this is simply the full prefix (no windows at all)."""
    stride = max(1, ctx // 2)
    s = torch.as_tensor(np.asarray(seq, dtype=np.int64))
    m = len(s) - 1
    tot = 0.0
    for j in range(1, m + 1):
        w = max(0, -(-(j - ctx) // stride))
        a = w * stride
        x = s[a:j].unsqueeze(0).to(rt.device)
        h = model.hidden(x)[0, -1]
        logp = torch.log_softmax((h @ model.tok.weight.t()).double(), dim=-1)
        tot -= float(logp[int(s[j])])
    return tot / math.log(2.0)


# ---------------------------------------------------------------- run recipe + identity (resume safety)
def _recipe(stage: str, overrides: dict = None):
    """The effective training recipe of `stage` after `overrides` (tests / smoke only)."""
    ov = dict(overrides or {})
    st = dict(STAGES[stage])
    st.setdefault("curve_eval", "subset")
    for k in ("budget_bytes", "epochs", "curve", "eval", "curve_eval"):
        if k in ov:
            st[k] = ov[k]
    return st, ov


def _plan_budget(bundle: Bundle, st: dict, ov: dict):
    """Budget in bytes -> step count, identical for every candidate. Returns (budget, steps, train_total_bytes)."""
    train_total_bytes = int(bundle.train_bytes().astype(np.int64).sum())
    if st["budget_bytes"] is not None:
        budget = int(st["budget_bytes"])
    else:
        budget = int(st["epochs"]) * train_total_bytes
    steps = steps_for_budget(budget)
    if "max_steps" in ov:
        steps = min(steps, int(ov["max_steps"]))
    return budget, steps, train_total_bytes


def _json_norm(x):
    """What x looks like after a JSON round trip (tuples -> lists), for comparing with stored records."""
    return json.loads(json.dumps(x, ensure_ascii=False, allow_nan=False))


# Fields that must match before a stored result may stand in for a requested run. The runtime fields
# are 'soft': run_all.py accepts a device/dtype difference only under --allow-mixed.
IDENTITY_FIELDS = ("protocol", "bundle_manifest_sha256", "hk_lm_sha256", "stage", "stage_recipe", "candidate",
                   "seed", "lr", "model.name", "data.budget_bytes", "data.epochs", "data.steps",
                   "data.ctx_tokens", "overrides", "runtime.device", "runtime.amp_dtype")
IDENTITY_SOFT = ("runtime.device", "runtime.amp_dtype")


def run_identity(bundle: Bundle, cid: str, stage: str, seed: int, lr: float, device: str, amp: str,
                 label: str = None, overrides: dict = None) -> dict:
    """The identity (IDENTITY_FIELDS) of the record run_one() would write for these arguments."""
    st, ov = _recipe(stage, overrides)
    ov.pop("keep_model", None)
    budget, steps, _ = _plan_budget(bundle, st, ov)
    return {"protocol": PROTOCOL, "bundle_manifest_sha256": bundle.manifest_sha256,
            "hk_lm_sha256": this_file_sha256(), "stage": label or stage, "stage_recipe": stage,
            "candidate": cid, "seed": int(seed), "lr": float(lr), "model.name": st["model"],
            "data.budget_bytes": int(budget), "data.epochs": st["epochs"], "data.steps": int(steps),
            "data.ctx_tokens": int(bundle.cand(cid)["ctx_tokens"]), "overrides": _json_norm(ov or None),
            "runtime.device": str(device), "runtime.amp_dtype": str(amp)}


def record_identity(rec: dict) -> dict:
    """The same fields read from a stored result record (missing -> None)."""
    out = {}
    for k in IDENTITY_FIELDS:
        v = rec
        for part in k.split("."):
            v = v.get(part) if isinstance(v, dict) else None
        if k == "overrides" and isinstance(v, dict):
            v = {kk: vv for kk, vv in v.items() if kk != "keep_model"} or None
        out[k] = v
    return out


def identity_mismatches(rec: dict, want: dict) -> list:
    """[(field, stored, requested)] for every identity field where a stored record differs from the
    requested run. Exact comparison: LRs round-trip exactly through JSON, so 5.8e-3 != 6e-3."""
    got = record_identity(rec)
    bad = []
    for k in IDENTITY_FIELDS:
        a, b = got.get(k), want.get(k)
        if k == "lr":
            same = a is not None and b is not None and float(a) == float(b)
        else:
            same = _json_norm(a) == _json_norm(b)
        if not same:
            bad.append((k, a, b))
    return bad


# ---------------------------------------------------------------- one run
def run_one(bundle: Bundle, cid: str, stage: str, seed: int, lr: float, rt: Runtime, label: str = None,
            out_path: str = None, overrides: dict = None, log=print) -> dict:
    """Train + evaluate one (stage, candidate, seed, lr). Writes the result record to out_path (atomic)
    and returns it. `overrides` (tests / smoke only): budget_bytes, max_steps, eval, curve, epochs."""
    st, ov = _recipe(stage, overrides)
    mcfg = MODELS[st["model"]]
    c = bundle.cand(cid)
    label = label or stage
    tag = "[%s|%s|s%d|lr%s]" % (label, cid, seed, lr_tag(lr))
    t_start = time.time()

    # budget -> identical step count for every candidate
    budget, steps, train_total_bytes = _plan_budget(bundle, st, ov)
    ctx = int(c["ctx_tokens"])
    n_seq = steps * SEQS_PER_STEP
    n_needed = n_seq * ctx + 1
    stream, sinfo = build_train_stream(bundle, cid, seed, n_needed)
    seq_starts = sequence_order(n_seq, seed).astype(np.int64) * ctx
    bos, eot = int(c["bos_id"]), int(c["eot_id"])
    mask_bos = bos != eot

    model = build_model(st["model"], int(c["n_vocab"]), ctx, seed, attn=rt.attn).to(rt.device)
    opt = make_optimizer(model, lr, rt.device)
    scaler = _make_scaler() if rt.amp == "fp16" else None
    stream_t = torch.from_numpy(stream).to(rt.device)
    starts_t = torch.from_numpy(seq_starts).to(rt.device)
    ar = torch.arange(ctx + 1, device=rt.device)
    losses = torch.zeros(steps, dtype=torch.float32, device=rt.device)
    curve_steps = {}
    if st["curve"]:
        for fr in CURVE_FRACS:
            s_ = max(1, int(round(fr * steps)))
            if s_ < steps:
                curve_steps.setdefault(s_, fr)
    subset_idx = bundle.eval_doc_index(st["curve_eval"])
    curve = []
    log_every = max(1, steps // 10)
    status = "ok"
    t_train0 = time.time()
    t_curve = 0.0
    steps_done = 0
    for step in range(steps):
        lr_s = lr_at(step, steps, lr)
        for g in opt.param_groups:
            g["lr"] = lr_s
        idx = starts_t[step * SEQS_PER_STEP:(step + 1) * SEQS_PER_STEP, None] + ar[None, :]
        buf = stream_t[idx]
        x = buf[:, :-1]
        y = buf[:, 1:]
        if mask_bos:
            y = y.masked_fill(y == bos, IGNORE)
        losses[step] = train_step(model, opt, scaler, x, y, rt)
        done = step + 1
        steps_done = done
        if done in curve_steps:
            tc = time.time()
            ev = evaluate(model, bundle, cid, subset_idx, ctx, rt)
            t_curve += time.time() - tc
            curve.append({"frac": curve_steps[done], "step": done, "bpb": ev["bpb"],
                          "sum_bits": ev["sum_bits"], "sum_bytes": ev["sum_bytes"]})
        if done % log_every == 0 or done == steps:
            lv = float(losses[step])
            log("%s step %d/%d loss %.4f lr %.2e %.1fs" % (tag, done, steps, lv, lr_s, time.time() - t_train0))
            if not math.isfinite(lv) and rt.amp != "fp16":
                status = "diverged"
                log("%s DIVERGED (non-finite loss) - stopping this run" % tag)
                break
    # the n_seq sequences tile the prefix exactly: targets are stream[1 : n_seq*ctx + 1]
    n_masked = int((stream[1:n_needed] == bos).sum()) if mask_bos else 0
    loss_list = [_finite_or_none(v) for v in losses[:steps_done].detach().cpu().tolist()]
    tail = [v for v in loss_list[max(0, len(loss_list) - 20):] if v is not None]
    if status == "ok" and (not tail or not all(math.isfinite(v) for v in tail) or np.mean(tail) > 50):
        status = "diverged"
    t_train = time.time() - t_train0 - t_curve

    # --- evaluation (fp32)
    dev = None
    r3 = None
    t_ev0 = time.time()
    if status == "ok":
        ev_idx = bundle.eval_doc_index(st["eval"])
        ev = evaluate(model, bundle, cid, ev_idx, ctx, rt)
        dev = ev
        if st["curve"]:
            sub_pos = np.searchsorted(ev_idx, subset_idx)
            if len(sub_pos) and np.all(sub_pos < len(ev_idx)) and np.all(ev_idx[np.minimum(sub_pos, len(ev_idx) - 1)] == subset_idx):
                sb = ev["bits"][sub_pos]
                curve.append({"frac": 1.0, "step": steps, "bpb": bpb_of(sb, ev["bytes"][sub_pos]),
                              "sum_bits": float(sb.sum()), "sum_bytes": int(ev["bytes"][sub_pos].sum())})
            else:
                e2 = evaluate(model, bundle, cid, subset_idx, ctx, rt)
                curve.append({"frac": 1.0, "step": steps, "bpb": e2["bpb"], "sum_bits": e2["sum_bits"],
                              "sum_bytes": e2["sum_bytes"]})
        # PLAN 4.3 R3: the 50 lowest-norm (tied) output embeddings with their train frequencies
        with torch.no_grad():
            norms = model.tok.weight.detach().float().norm(dim=1).cpu().numpy()
        tt, _ = bundle.tokens(cid, "train")
        freq = np.bincount(tt.astype(np.int64), minlength=int(c["n_vocab"]))
        low = np.argsort(norms, kind="stable")[:50]
        r3 = [{"id": int(i), "norm": float(norms[i]), "train_freq": int(freq[i])} for i in low]
    t_eval = time.time() - t_ev0 + t_curve

    pc = model.param_counts()
    docs = bundle.dev_docs()
    rec = {
        "protocol": PROTOCOL, "plan_ref": PLAN_REF, "status": status,
        "stage": label, "stage_recipe": stage, "candidate": cid, "seed": int(seed), "lr": float(lr),
        "preregistered": bool(st.get("preregistered", False)) and bool(mcfg["preregistered"]),
        "model": {"name": st["model"], "d": mcfg["d"], "L": mcfg["L"], "H": mcfg["H"],
                  "role": mcfg["role"], "n_vocab": int(c["n_vocab"]), "ctx_tokens": ctx,
                  "params": pc, "tied_embeddings": True, "dropout": 0.0, "mlp": "GELU 4x",
                  "norm": "pre-LN", "positions": "learned", "linear_bias": False,
                  "init": "normal(0, 0.02); residual projections 0.02/sqrt(2L); CPU generator seeded by run seed"},
        "optimizer": {"name": "AdamW", "betas": list(BETAS), "eps": ADAM_EPS, "weight_decay": WEIGHT_DECAY,
                      "weight_decay_on": "parameters with dim >= 2 (all matrices incl. tied embedding and positions)",
                      "grad_clip": GRAD_CLIP, "warmup_steps": WARMUP_STEPS, "schedule": "cosine to 10% of peak",
                      "peak_lr": float(lr)},
        "data": {"budget_bytes": budget, "epochs": st["epochs"], "bytes_per_step": BYTES_PER_STEP,
                 "seqs_per_step": SEQS_PER_STEP, "steps": steps, "steps_done": steps_done,
                 "ctx_tokens": ctx, "train_bytes_per_token": c["train_bytes_per_token"],
                 "tokens_seen": int(n_seq * ctx), "targets_masked_bos": n_masked,
                 "doc_order": "PCG64(SeedSequence([seed, epoch, 0x484B])).permutation(n_train_docs); identical for all candidates",
                 "sequence_order": "PCG64(SeedSequence([seed, 0x5E0, 0x484B])).permutation(steps*8)",
                 "bos_id": bos, "eot_id": eot, "bos_is_eot": not mask_bos,
                 "train_docs": int(len(bundle.train_bytes())), "train_total_bytes": train_total_bytes,
                 **sinfo},
        "runtime": dict(rt.info(), ce_chunk_rows=rt.ce_chunk(), eval_window_batch=rt.eval_batch()),
        "wall_time_s": {"train": round(t_train, 2), "eval": round(t_eval, 2),
                        "total": round(time.time() - t_start, 2)},
        "train_loss": loss_list,
        "curve": curve,
        "r3_lowest_norm_embeddings": r3,
        "bundle_manifest_sha256": bundle.manifest_sha256,
        "hk_lm_sha256": this_file_sha256(),
        "overrides": ov or None,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if dev is not None:
        ev_idx = bundle.eval_doc_index(st["eval"])
        rec["dev"] = {"doc_set": st["eval"] if isinstance(st["eval"], str) else list(st["eval"]),
                      "n_docs": dev["n_docs"], "n_windows": dev["n_windows"],
                      "window": {"ctx": ctx, "stride": max(1, ctx // 2), "bos_given": True, "eot_predicted": True},
                      "bpb": dev["bpb"], "sum_bits": dev["sum_bits"], "sum_bytes": dev["sum_bytes"],
                      "doc_index": ev_idx.tolist(),
                      "uids": [docs[int(i)]["uid"] for i in ev_idx],
                      "bits": dev["bits"].tolist(), "bytes": dev["bytes"].tolist(), "ntok": dev["ntok"].tolist()}
        rec["bpb"] = dev["bpb"]
        log("%s dev bpb %.5f (%d docs, %.3f MB) train %.1fs eval %.1fs" %
            (tag, dev["bpb"], dev["n_docs"], dev["sum_bytes"] / 1e6, t_train, t_eval))
    else:
        rec["bpb"] = None
    rec["subset"] = {"doc_set": st["curve_eval"] if isinstance(st["curve_eval"], str) else list(st["curve_eval"]),
                     "doc_index": subset_idx.tolist(), "bytes": int(bundle.dev_bytes()[subset_idx].sum())}
    if out_path:
        write_json_atomic(out_path, rec)
    if ov.get("keep_model"):          # tests only: hand the trained model back (never serialised)
        rec = dict(rec, _model=model)
    del model, opt, stream_t
    if rt.device.type == "cuda":
        torch.cuda.empty_cache()
    return rec


# ---------------------------------------------------------------- FLOP-based runtime estimate
def train_flops_per_token(model_name: str, n_vocab: int, ctx: int) -> float:
    """~6 N per token for the matmuls (N incl. the tied output layer) + causal attention."""
    m = MODELS[model_name]
    d, L = m["d"], m["L"]
    n_mat = 12 * d * d * L + n_vocab * d
    attn = 6 * L * ctx * d          # fwd 2*(QK + AV)*ctx*d/2 causal ~ 2 L ctx d; x3 fwd+bwd
    return 6.0 * n_mat + attn


def eval_flops_per_token(model_name: str, n_vocab: int, ctx: int) -> float:
    return train_flops_per_token(model_name, n_vocab, ctx) / 3.0


# ---------------------------------------------------------------- CLI
def _main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--cand", required=True)
    ap.add_argument("--stage", default="screen", choices=sorted(STAGES))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--amp", default="auto", choices=["auto", "fp32", "bf16", "fp16"])
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--eval", default=None, help="all | subset | parity | first:N")
    ap.add_argument("--no-curve", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    b = Bundle(a.bundle, verify=a.verify)
    rt = Runtime(a.device, threads=a.threads, amp=a.amp)
    ov = {}
    if a.max_steps:
        ov["max_steps"] = a.max_steps
    if a.eval:
        ov["eval"] = ("first", int(a.eval.split(":")[1])) if a.eval.startswith("first:") else a.eval
    if a.no_curve:
        ov["curve"] = False
    rec = run_one(b, a.cand, a.stage, a.seed, a.lr, rt, out_path=a.out, overrides=ov)
    print(json.dumps({"bpb": rec["bpb"], "steps": rec["data"]["steps"], "ctx": rec["data"]["ctx_tokens"],
                      "status": rec["status"]}))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    _main()
