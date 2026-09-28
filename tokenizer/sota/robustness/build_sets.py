# -*- coding: utf-8 -*-
"""Build every evaluation set of the robustness study, deterministically (seed 20260927), into sets/.

  1. OUT-OF-CORPUS Hindko: survey samples on disk (F:\\Hindko\\_web\\survey\\_samples, ...\\samples,
     ...\\corpora_work\\samples), each split into documents and checked against the released corpus with two
     indexes (corpus_ngrams.py: word 8-grams; corpus_chargrams.py: spacing-insensitive character 24-grams).
     A document is KEPT as out-of-corpus Hindko when all hold:
        >= 20 key words; word-8-gram overlap < 10 %; char-24-gram overlap < 20 %;
        Arabic-script share of letters >= 0.6; U+FFFD share < 0.1 %;
        Hindko/Punjabi function-word rate > Urdu function-word rate (so Urdu prose about Hindko is dropped).
     In-corpus sources (Common Voice, Omnilingual, the Gandhara Hindko site, hindko.org, ...) are checked too, as
     positive controls of the overlap test; 25 test_strict documents are a further control (expected ~100 %).
  2. OTHER LANGUAGES: deterministic samples of F:\\Hindko\\_web\\ref\\{urd,pnb,skr,pbt}_Arab.jsonl (docs with
     >= 50 % word-8-gram overlap with the Hindko corpus are dropped and counted); 50 English prose paragraphs from
     the METADATA long descriptions of the installed Python packages (first qualifying paragraph per package, by
     package name); 30 Python standard-library modules (fixed list).
  3. CODE-MIXED: 100 lines = one test_strict Hindko sentence each + English words/phrases (Latin script) inserted
     at random word boundaries + (some lines) an English prefix/tail and an Urdu clause (fixed lists below).
  4. EXAMPLES: the 3 card sentences, located in test_strict.
Each set row has text_norm (hp.normalize 1.0.1 applied: the documented input form) and text_raw (as found).
The set files contain third-party text: LOCAL USE ONLY, do not publish them.
"""
import glob
import json
import os
import random
import re
import sys
import sysconfig
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import rcommon as C  # noqa: E402
from corpus_chargrams import chargram_hashes  # noqa: E402

SEED = 20260927
SURVEY = r"F:\Hindko\_web\survey"
S1 = os.path.join(SURVEY, "_samples")
S2 = os.path.join(SURVEY, "samples")
S3 = os.path.join(SURVEY, "corpora_work", "samples")
REF = r"F:\Hindko\_web\ref"
FS, QM, AC = chr(0x06D4), chr(0x061F), chr(0x060C)

# ------------------------------------------------------------------------------ function-word markers
def _w(*cps_lists):
    return {"".join(chr(c) for c in cps) for cps in cps_lists}

# Hindko / Punjabi genitive, postpositions and pronouns (none is an Urdu function word)
HK_MARK = _w((0x062F, 0x0627), (0x062F, 0x06CC), (0x062F, 0x06D2), (0x0646, 0x0648, 0x06BA), (0x062A, 0x06D2),
             (0x0627, 0x0686), (0x0628, 0x0686), (0x0648, 0x0686), (0x0627, 0x06D2), (0x0648, 0x06D2),
             (0x06C1, 0x06A9), (0x0627, 0x0633, 0x0627, 0x06BA), (0x062A, 0x0633, 0x0627, 0x06BA),
             (0x06A9, 0x062C), (0x06A9, 0x062C, 0x06BE), (0x0646, 0x0626, 0x06CC, 0x06BA), (0x06C1, 0x0626, 0x06D2))
# Urdu function words that Hindko does not use in these roles
UR_MARK = _w((0x06A9, 0x0627), (0x06A9, 0x06CC), (0x06A9, 0x06D2), (0x06A9, 0x0648), (0x06C1, 0x06D2),
             (0x06C1, 0x06CC, 0x06BA), (0x0633, 0x06D2), (0x062A, 0x06BE, 0x0627), (0x062A, 0x06BE, 0x06CC),
             (0x062A, 0x06BE, 0x06D2), (0x0627, 0x0648, 0x0631), (0x06CC, 0x06C1), (0x0648, 0x06C1))


def marker_rates(text):
    ws = re.findall(r"[^\W\d_]+", "".join(c for c in text if ord(c) not in C.HARAKAT))
    n = max(len(ws), 1)
    return sum(w in HK_MARK for w in ws) / n, sum(w in UR_MARK for w in ws) / n


def arabic_letter_share(text):
    letters = [c for c in text if unicodedata.category(c).startswith("L")]
    if not letters:
        return 0.0
    ar = sum(1 for c in letters if 0x0600 <= ord(c) <= 0x08FF)
    return ar / len(letters)


# ------------------------------------------------------------------------------ overlap
W8 = np.load(os.path.join(C.SETS, "corpus_8grams.npy"))
C24 = np.load(os.path.join(C.SETS, "corpus_c24.npy"))


def _found(A, h):
    if len(h) == 0:
        return 0
    i = np.searchsorted(A, h)
    i = np.minimum(i, len(A) - 1)
    return int(np.sum(A[i] == h))


def overlap(text):
    w = C.key_words(text)
    h8 = C.ngram_hashes(w, 8)
    hc = chargram_hashes(text)
    return {"key_words": len(w), "w8_n": int(len(h8)), "w8_found": _found(W8, h8),
            "w8_overlap": (_found(W8, h8) / len(h8)) if len(h8) else None,
            "c24_n": int(len(hc)), "c24_found": _found(C24, hc),
            "c24_overlap": (_found(C24, hc) / len(hc)) if len(hc) else None}


# ------------------------------------------------------------------------------ 1. out-of-corpus Hindko
HEADER_RE = re.compile(r"^#### .*$", re.M)


def split_docs(text, mode):
    if mode == "whole":
        return [text.strip()]
    if mode == "lines":
        return [l.strip() for l in text.split("\n") if l.strip()]
    if mode == "header":        # '#### <url>' header lines start a document
        parts = HEADER_RE.split(text)
        return [p.strip() for p in parts if p.strip()]
    raise ValueError(mode)


def read_txt(p):
    with open(p, encoding="utf-8", errors="replace", newline="") as f:
        return f.read().replace("\r\n", "\n")


OOC = [
    # id, path, kind, split, description, expected ('ooc' = candidate; 'in' = known corpus source, control)
    ("gotquestions_southern_hindko", os.path.join(S1, "gq_all.txt"), "txt", "lines",
     "GotQuestions.org articles in Southern Hindko (hnd), one article per line", "ooc"),
    ("surah_yasin_hindko_pdf", os.path.join(S2, "ia_surah_yasin_pdf.txt"), "txt", "header",
     "Surah Yasin with its meaning in Hindko (Internet Archive PDF text layer: joined words, bidi marks)", "ooc"),
    ("chup_di_kahani_pdf", os.path.join(S2, "ia_chup_di_kahani_pdf.txt"), "txt", "header",
     "'Chup di Kahani', Hindko short stories (Internet Archive PDF text layer)", "ooc"),
    ("translatewiki_hno", os.path.join(S1, "opus_tw_hno.txt"), "txt", "whole",
     "MediaWiki interface messages in Northern Hindko (OPUS translatewiki), one file", "ooc"),
    # screened but NOT used: the rows that pass the filters are mostly Punjabi by inspection (Punjabi poetry sites,
    # pnb.wikipedia.org, Punjabi naat lyrics); the function-word filter cannot separate Punjabi from Hindko
    ("fineweb2_pnb_hindko_candidates", os.path.join(S3, "fw2_pnb_hindko_candidates.json"), "json", None,
     "FineWeb-2 pnb_Arab pages flagged as Hindko by the survey (passing rows mostly Punjabi: not used)", "uncertain"),
    ("fineweb2_pnb_hindko_domains", os.path.join(S3, "fw2_pnb_train_hindko_domains.json"), "json", None,
     "FineWeb-2 pnb_Arab rows from Hindko-related domains (passing rows mostly Punjabi poetry: not used)", "uncertain"),
    ("finepdfs_hindko_org", os.path.join(S3, "finepdfs_pnb_hindko_url_rows.json"), "json", None,
     "hindko.org PDFs as OCR'd by FinePDFs (Hindko, but heavily OCR-corrupted letters: a noisy-input set)", "ooc"),
    ("bible_for_children_pdf", os.path.join(S1, "bfc_hindko_01.txt"), "txt", "whole",
     "Bible for Children in Hindko (PDF text layer)", "ooc"),
    ("manglori_tareekh_djvu", os.path.join(S2, "ia_manglori_tareekh.txt"), "txt", "header",
     "History of Hindko language and literature (Internet Archive djvu OCR)", "ooc"),
    # positive controls: sources the corpus is known to contain
    ("ctrl_common_voice_hno", os.path.join(S1, "cv26_hno_unique_sentences.txt"), "txt", "whole",
     "Common Voice 26 Hindko sentences (corpus source common_voice_hno)", "in_corpus"),
    ("ctrl_omnilingual_hno", os.path.join(S1, "omni_hno_transcripts.txt"), "txt", "lines",
     "Omnilingual ASR Hindko transcripts (corpus source omnilingual_asr_hno)", "in_corpus"),
    ("ctrl_gandhara_posts", os.path.join(S2, "ghb_post_bodies.txt"), "txt", "header",
     "gandharahindko.com posts (corpus source web_gandharahindko)", "in_corpus"),
    ("ctrl_hindko_org", os.path.join(S2, "hlcs_hindko_org.txt"), "txt", "header",
     "hindko.org pages (corpus source web_hindko_org)", "in_corpus"),
    ("ctrl_aaprihindko", os.path.join(S2, "aaprihindko_wayback_all.txt"), "txt", "header",
     "aaprihindko.com via the Wayback Machine (corpus source web_aaprihindko)", "in_corpus"),
    ("ctrl_hindkomaza", os.path.join(S2, "hindkomaza.txt"), "txt", "header", "hindkomaza blog (web_hindkomaza)", "in_corpus"),
    ("ctrl_blog_hindko_pk", os.path.join(S2, "blog_hindko-pk.txt"), "txt", "header", "hindko-pk blog", "in_corpus"),
    ("ctrl_blog_hindkopoint", os.path.join(S2, "blog_hindkopoint.txt"), "txt", "header", "hindkopoint blog", "in_corpus"),
    ("urdu_about_hindko", os.path.join(S2, "urdu_about_hindko.txt"), "txt", "header", "Urdu prose about Hindko",
     "not_hindko"),
    ("urduweb_zeerak", os.path.join(S2, "urduweb_zeerak.txt"), "txt", "header",
     "urduweb forum thread comparing Punjabi/Saraiki/Pothwari/Hindko words (word lists, not running Hindko)",
     "not_hindko"),
]


def build_ooc():
    rep, keep_sets = [], {}
    for sid, path, kind, mode, desc, exp in OOC:
        if kind == "txt":
            docs = split_docs(read_txt(path), mode)
        else:
            rows = json.load(open(path, encoding="utf-8"))
            docs = [(r.get("text") or "").replace("\r\n", "\n").strip() for r in rows]
            docs = [d for d in docs if d]
        kept, rows_out, dstats = [], [], []
        for k, d in enumerate(docs):
            ov = overlap(d)
            hk, ur = marker_rates(d)
            asr = arabic_letter_share(d)
            fffd = d.count(chr(0xFFFD)) / max(len(d), 1)
            reasons = []
            if ov["key_words"] < 20:
                reasons.append("<20 key words")
            if ov["w8_overlap"] is not None and ov["w8_overlap"] >= 0.10:
                reasons.append("word-8-gram overlap %.0f%%" % (100 * ov["w8_overlap"]))
            if ov["c24_overlap"] is not None and ov["c24_overlap"] >= 0.20:
                reasons.append("char-24-gram overlap %.0f%%" % (100 * ov["c24_overlap"]))
            if asr < 0.6:
                reasons.append("Arabic-script letter share %.2f" % asr)
            if fffd >= 0.001:
                reasons.append("U+FFFD share %.2f%%" % (100 * fffd))
            if hk <= ur:
                reasons.append("Urdu function words >= Hindko ones (%.3f vs %.3f)" % (ur, hk))
            ok = not reasons
            dstats.append({"doc": k, "chars": len(d), **ov, "hk_rate": round(hk, 4), "ur_rate": round(ur, 4),
                           "arabic_letter_share": round(asr, 3), "fffd_share": round(fffd, 5), "kept": ok,
                           "reasons": reasons})
            if ok:
                rows_out.append({"id": "%s/%d" % (sid, k), "text_raw": d, "text_norm": C.normalize(d)})
        tot8 = sum(x["w8_n"] for x in dstats)
        f8 = sum(x["w8_found"] for x in dstats)
        totc = sum(x["c24_n"] for x in dstats)
        fc = sum(x["c24_found"] for x in dstats)
        k8 = sum(x["w8_n"] for x in dstats if x["kept"])
        kf8 = sum(x["w8_found"] for x in dstats if x["kept"])
        kc = sum(x["c24_n"] for x in dstats if x["kept"])
        kfc = sum(x["c24_found"] for x in dstats if x["kept"])
        r = {"id": sid, "path": path, "sha256": C.sha256_file(path), "description": desc, "expected": exp,
             "docs": len(docs), "docs_kept": len(rows_out), "chars_kept": sum(len(x["text_norm"]) for x in rows_out),
             "w8_overlap_all": f8 / tot8 if tot8 else None, "c24_overlap_all": fc / totc if totc else None,
             "w8_overlap_kept": kf8 / k8 if k8 else None, "c24_overlap_kept": kfc / kc if kc else None,
             "doc_stats": dstats}
        rep.append(r)
        print("%-34s docs %4d kept %4d  w8 %s  c24 %s" % (
            sid, len(docs), len(rows_out),
            "%.3f" % r["w8_overlap_all"] if r["w8_overlap_all"] is not None else "-",
            "%.3f" % r["c24_overlap_all"] if r["c24_overlap_all"] is not None else "-"), flush=True)
        if exp == "ooc" and rows_out:
            keep_sets[sid] = rows_out
    # control: test_strict documents must be found (~100 %)
    test = C.load_test()
    rng = random.Random(SEED)
    ctrl = [test[i] for i in sorted(rng.sample(range(len(test)), 25))]
    c8 = [overlap(d["text"]) for d in ctrl]
    ctrl_rep = {"docs": 25, "w8_overlap": sum(x["w8_found"] for x in c8) / max(1, sum(x["w8_n"] for x in c8)),
                "c24_overlap": sum(x["c24_found"] for x in c8) / max(1, sum(x["c24_n"] for x in c8)),
                "min_doc_w8": min(x["w8_overlap"] for x in c8 if x["w8_overlap"] is not None)}
    print("control test_strict", ctrl_rep)
    return rep, keep_sets, ctrl_rep


# ------------------------------------------------------------------------------ 2. other languages
def build_ref(code, k):
    rows = C.read_jsonl(os.path.join(REF, code + ".jsonl"))
    rng = random.Random("%d-%s" % (SEED, code))
    idx = sorted(rng.sample(range(len(rows)), min(k, len(rows))))
    out, dropped = [], 0
    for i in idx:
        t = (rows[i].get("text") or "").replace("\r\n", "\n").strip()
        if not t:
            continue
        ov = overlap(t)
        if ov["w8_overlap"] is not None and ov["w8_overlap"] >= 0.5:
            dropped += 1
            continue
        out.append({"id": "%s/%d" % (code, i), "text_raw": t, "text_norm": C.normalize(t)})
    return out, {"source": os.path.join(REF, code + ".jsonl"), "sha256": C.sha256_file(os.path.join(REF, code + ".jsonl")),
                 "rows_in_file": len(rows), "sampled": len(idx), "dropped_in_hindko_corpus": dropped, "kept": len(out)}


BAD_START = ("```", "    ", "\t", ">>>", "$", "|", "..", "#", "*", "-", "+", "<", "[", "!", "=", ":", "{", "(", "`")
LICENSE_RE = re.compile(r"(?i)copyright|licen[cs]e|warrant|permission is hereby|\(c\)")


def build_english(n=50):
    sp = sysconfig.get_paths()["purelib"]
    files = sorted(glob.glob(os.path.join(sp, "*.dist-info", "METADATA")), key=lambda p: os.path.basename(os.path.dirname(p)).lower())
    per_pkg, seen = [], set()
    for p in files:
        txt = read_txt(p)
        body = txt.split("\n\n", 1)[1] if "\n\n" in txt else ""
        good = []
        for para in re.split(r"\n\s*\n", body):
            lines = [l for l in para.split("\n") if l.strip()]
            if not lines or any(l.startswith(BAD_START) or l.lstrip().startswith(BAD_START[2:]) for l in lines):
                continue
            s = " ".join(l.strip() for l in lines)
            if not (250 <= len(s) <= 1500) or "http" in s or "::" in s or "`" in s or LICENSE_RE.search(s):
                continue
            if sum(ord(c) < 128 for c in s) / len(s) < 0.995:
                continue
            if sum(c.isalpha() or c == " " for c in s) / len(s) < 0.88 or len(s.split()) < 40 or not s.endswith("."):
                continue
            if s in seen:
                continue
            seen.add(s)
            good.append(s)
        if good:
            per_pkg.append((os.path.basename(os.path.dirname(p)), good))
    out = []
    for rnd in range(2):                     # round robin: 1st paragraph of every package, then 2nd ones
        for pkg, good in per_pkg:
            if rnd < len(good) and len(out) < n:
                out.append({"id": "en/%s/%d" % (pkg, rnd), "text_raw": good[rnd], "text_norm": good[rnd]})
    return out, {"source": sp + r"\*.dist-info\METADATA (long descriptions)", "packages_scanned": len(files),
                 "packages_with_prose": len(per_pkg),
                 "rule": "paragraphs with 250-1500 chars, >=40 words, >=99.5% ASCII, >=88% letters/spaces, ending "
                         "in '.', no code/list/heading markup, no URL, no licence text; round robin over packages "
                         "sorted by dist-info name (1st qualifying paragraph of each, then 2nd), first 50",
                 "kept": len(out)}


PY_MODULES = ["abc.py", "bisect.py", "calendar.py", "colorsys.py", "copy.py", "csv.py", "dataclasses.py",
              "difflib.py", "fnmatch.py", "fractions.py", "functools.py", "glob.py", "heapq.py", "keyword.py",
              "linecache.py", "numbers.py", "operator.py", "queue.py", "random.py", "shlex.py", "shutil.py",
              "statistics.py", "string.py", "textwrap.py", "timeit.py", "tokenize.py", "uuid.py", "weakref.py",
              "json/decoder.py", "json/encoder.py"]


def build_code():
    lib = sysconfig.get_paths()["stdlib"]
    out = []
    for m in PY_MODULES:
        t = read_txt(os.path.join(lib, m))
        out.append({"id": "py/" + m, "text_raw": t, "text_norm": t})     # code: never normalized
    return out, {"source": lib, "modules": PY_MODULES, "python": sys.version.split()[0], "kept": len(out)}


# ------------------------------------------------------------------------------ 3. code-mixed
EN_WORDS = ["meeting", "office", "mobile", "online", "class", "exam", "result", "university", "WhatsApp", "Facebook",
            "YouTube", "video", "link", "update", "message", "news", "match", "cricket", "team", "government",
            "school", "teacher", "doctor", "hospital", "report", "project", "deadline", "computer", "internet",
            "traffic", "weekend", "problem", "busy", "important", "actually", "basically", "seriously", "please",
            "sorry", "OK"]
EN_PREFIX = ["Actually,", "By the way,", "Honestly,", "So basically,", "Listen,", "Guys,"]
EN_TAIL = ["Thank you!", "Please share it.", "What do you think?", "OK?", "See you tomorrow.", "Good luck!",
           "No problem.", "That is the point."]
UR_CLAUSES = [
    "مجھے لگتا ہے کہ یہ بہت ضروری ہے۔",
    "آپ کا کیا خیال ہے؟",
    "کوئی بات نہیں۔",
    "بہت شکریہ۔",
    "میں آپ کو بعد میں بتاؤں گا۔",
    "یہ بالکل ٹھیک ہے۔",
    "ہمیں اس پر غور کرنا چاہیے۔",
    "جلدی کریں، وقت کم ہے۔",
    "اصل میں بات کچھ اور ہے۔",
    "سب لوگ وہاں موجود تھے۔",
    "یہ خبر سب کو معلوم ہے۔",
    "ان شاء اللہ سب ٹھیک ہو جائے گا۔",
]
ARABIC_ONLY = re.compile("^[" + "".join(chr(c) for c in range(0x0600, 0x0700)) + chr(0x0768) +
                         "".join(chr(c) for c in range(0x08A0, 0x0900)) + " ]+$")


def sentences_of(doc):
    out = []
    for line in doc["text"].split("\n"):
        for s in re.split("(?<=[%s%s])\\s*" % (FS, QM), line):
            s = s.strip()
            w = s.split()
            if 8 <= len(w) <= 25 and s.endswith((FS, QM)) and ARABIC_ONLY.match(s[:-1]):
                out.append(s)
    return out


def build_codemixed(test, n=100):
    for u in UR_CLAUSES:                      # guard against encoding damage of the literals above
        assert all(0x0600 <= ord(c) <= 0x06FF or c == " " for c in u), u
    rng = random.Random("%d-codemix" % SEED)
    by_doc = []
    for d in sorted((d for d in test if d["variety"] == "hindko"), key=lambda d: d["uid"]):
        ss = sentences_of(d)
        if ss:
            by_doc.append((d["uid"], ss))
    pick = sorted(rng.sample(range(len(by_doc)), n))
    rows = []
    for j in pick:
        uid, ss = by_doc[j]
        base = ss[rng.randrange(len(ss))]
        words = base.split()
        k = rng.choice([1, 1, 2, 2, 3])
        pos = sorted(rng.sample(range(1, len(words)), min(k, len(words) - 1)), reverse=True)
        ins = []
        for p in pos:
            e = rng.choice(EN_WORDS)
            words.insert(p, e)
            ins.append(e)
        line = " ".join(words)
        pre = tail = ur = None
        if rng.random() < 0.25:
            pre = rng.choice(EN_PREFIX)
            line = pre + " " + line
        if rng.random() < 0.25:
            tail = rng.choice(EN_TAIL)
            line = line + " " + tail
        if rng.random() < 0.35:
            ur = rng.choice(UR_CLAUSES)
            line = line + " " + ur
        rows.append({"id": "mix/%s" % uid, "uid": uid, "base": base, "text_raw": line, "text_norm": line,
                     "english_inserted": ins[::-1], "english_prefix": pre, "english_tail": tail, "urdu_clause": ur})
    return rows


# ------------------------------------------------------------------------------ 4. examples
EXAMPLES = [
    ("e1c68c155a177618", "book", "Very few people know how and where drama was born."),
    ("28ef7bef82c534b1", "newspaper", "If we do not reflect on these words of our elders today, we will be left with "
                                      "nothing but regret."),
    ("5d343e5acaa07e1c", "newspaper", "In the same way, the literature and culture of the other languages spoken in "
                                      "Khyber Pakhtunkhwa were also represented."),
]
EXAMPLE_PREFIX = ["ایہہ بہت کہٹ لوک", "اج اساں اگر آپڑیں", "اسی طراں خیبر پختونخوا"]


def build_examples(test):
    by = {d["uid"]: d for d in test}
    out = []
    for (uid, src, gloss), pre in zip(EXAMPLES, EXAMPLE_PREFIX):
        cand = [s for s in sentences_of(by[uid]) + [x.strip() for x in re.split("(?<=[%s%s])" % (FS, QM), by[uid]["text"])]
                if s.startswith(pre)]
        assert cand, (uid, pre)
        s = cand[0]
        assert 10 <= len(s.split()) <= 20, (uid, len(s.split()))
        out.append({"id": "ex/%s" % uid, "uid": uid, "source": src, "text_raw": s, "text_norm": s,
                    "words": len(s.split()), "gloss_en_approx": gloss})
    return out


def main():
    os.makedirs(C.SETS, exist_ok=True)
    manifest = {"seed": SEED, "note": "LOCAL USE ONLY: these files contain third-party text; do not publish them.",
                "normalize": {"version": "1.0.1", "sha256": C.sha256_file(os.path.join(C.PIPELINE, "hp", "normalize.py"))},
                "corpus_indexes": {"w8": json.load(open(os.path.join(C.SETS, "corpus_8grams.json"), encoding="utf-8")),
                                   "c24": json.load(open(os.path.join(C.SETS, "corpus_c24.json"), encoding="utf-8"))},
                "sets": {}}
    rep, keep_sets, ctrl = build_ooc()
    manifest["ooc_screening"] = rep
    manifest["ooc_control_test_strict"] = ctrl
    for sid, rows in keep_sets.items():
        C.write_jsonl(os.path.join(C.SETS, "ooc_%s.jsonl" % sid), rows)
        manifest["sets"]["ooc_" + sid] = {"group": "ooc_hindko", "docs": len(rows)}
    for code, k, name in (("urd_Arab", 300, "urdu"), ("pnb_Arab", 300, "punjabi_shahmukhi"),
                          ("skr_Arab", 200, "saraiki"), ("pbt_Arab", 200, "pashto")):
        rows, info = build_ref(code, k)
        C.write_jsonl(os.path.join(C.SETS, "lang_%s.jsonl" % name), rows)
        manifest["sets"]["lang_" + name] = {"group": "other_language", **info}
    rows, info = build_english()
    C.write_jsonl(os.path.join(C.SETS, "lang_english.jsonl"), rows)
    manifest["sets"]["lang_english"] = {"group": "other_language", **info}
    rows, info = build_code()
    C.write_jsonl(os.path.join(C.SETS, "lang_python_code.jsonl"), rows)
    manifest["sets"]["lang_python_code"] = {"group": "other_language", **info}
    test = C.load_test()
    mixed_nat = [{"id": "mixednat/" + d["uid"], "text_raw": d["text"], "text_norm": d["text"]}
                 for d in test if d["variety"] == "mixed"]
    C.write_jsonl(os.path.join(C.SETS, "mixed_natural_test.jsonl"), mixed_nat)
    manifest["sets"]["mixed_natural_test"] = {"group": "code_mixed", "docs": len(mixed_nat),
                                              "source": "test_strict documents with variety 'mixed' (Hindko + Urdu)"}
    cm = build_codemixed(test)
    C.write_jsonl(os.path.join(C.SETS, "codemixed.jsonl"), cm)
    manifest["sets"]["codemixed"] = {"group": "code_mixed", "docs": len(cm), "en_words": EN_WORDS,
                                     "en_prefix": EN_PREFIX, "en_tail": EN_TAIL, "ur_clauses": UR_CLAUSES,
                                     "rule": "one sentence (8-25 words, Arabic script only, ends in U+06D4/U+061F) from "
                                             "each of 100 randomly chosen hindko-variety test_strict documents; 1-3 "
                                             "English items (choice of [1,1,2,2,3]) inserted at random word "
                                             "boundaries; P=0.25 English prefix; P=0.25 English tail; P=0.35 Urdu clause "
                                             "appended; random.Random('%d-codemix')" % SEED}
    ex = build_examples(test)
    C.write_jsonl(os.path.join(C.SETS, "examples.jsonl"), ex)
    manifest["sets"]["examples"] = {"group": "examples", "docs": len(ex)}
    for name in list(manifest["sets"]):
        p = os.path.join(C.SETS, name + ".jsonl")
        manifest["sets"][name]["file"] = os.path.basename(p)
        manifest["sets"][name]["sha256"] = C.sha256_file(p)
        rows = C.read_jsonl(p)
        manifest["sets"][name]["docs"] = len(rows)
        manifest["sets"][name]["bytes_norm"] = sum(len(r["text_norm"].encode("utf-8")) for r in rows)
        manifest["sets"][name]["bytes_raw"] = sum(len(r["text_raw"].encode("utf-8")) for r in rows)
    json.dump(manifest, open(os.path.join(C.SETS, "sets_manifest.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for k, v in manifest["sets"].items():
        print("%-40s docs %4d  bytes %8d" % (k, v["docs"], v["bytes_norm"]))


if __name__ == "__main__":
    main()
