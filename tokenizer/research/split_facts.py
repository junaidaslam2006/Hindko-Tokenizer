# -*- coding: utf-8 -*-
"""Facts about the newspaper cluster key and the split manifest that PLAN.md 1.2 / 5 / 6 rely on
(added 2026-09-26 after review). Read-only on the dataset and on _tokenizer/splits.
Writes research/split_facts.json.
"""
import collections, hashlib, json, os

DS = r"F:\Hindko\hindko_dataset_permissive.jsonl"
MAN = r"F:\Hindko\_tokenizer\splits\split_manifest.jsonl"
HERE = os.path.dirname(os.path.abspath(__file__))


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def boot_cluster(m):
    """Bootstrap cluster: the manifest group, except per-record web groups -> their site (coarser)."""
    g = m["group"]
    if g.startswith("web:") and ":record:" in g:
        return g.split(":record:")[0]
    return g


def main():
    recs = {}
    with open(DS, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line); recs[r["uid"]] = r
    man = [json.loads(l) for l in open(MAN, encoding="utf-8")]
    out = {"manifest": MAN, "manifest_sha256": sha256(MAN), "records_in_manifest": len(man)}

    # --- newspaper issue field
    news = [r for r in recs.values() if r["source"] == "newspaper"]
    null = [r for r in news if not r.get("issue")]
    b = lambda rs: sum(len(r["text"].encode("utf-8")) for r in rs)
    sn = [r for r in news if r["quality_tier"] == "strict"]; snn = [r for r in sn if not r.get("issue")]
    tree = collections.Counter(r["source_path"].split("/")[1] for r in null)
    prec = collections.Counter(str(r.get("date_precision")) for r in null)
    out["newspaper_issue_null"] = {
        "records": len(null), "of_newspaper_records": len(news),
        "share_of_newspaper_bytes_pct": round(100 * b(null) / b(news), 2),
        "strict_records": len(snn), "strict_bytes_MB": round(b(snn) / 1e6, 3),
        "share_of_strict_newspaper_bytes_pct": round(100 * b(snn) / b(sn), 2),
        "by_source_path_tree": dict(tree), "by_date_precision": dict(prec),
        "distinct_source_path_folders": len({os.path.dirname(r["source_path"]) for r in null})}

    # --- manifest: bytes per split x tier, bootstrap clusters in evaluation splits
    by = collections.defaultdict(lambda: [0, 0])
    clusters = collections.defaultdict(lambda: collections.defaultdict(set))
    news_groups = collections.Counter()
    biggest = collections.defaultdict(collections.Counter)
    for m in man:
        r = recs[m["uid"]]; nb = len(r["text"].encode("utf-8"))
        for tier in ("permissive", "strict"):
            if tier == "permissive" or r["quality_tier"] == "strict":
                by[(m["split"], tier)][0] += 1; by[(m["split"], tier)][1] += nb
        if m["split"] != "train" and r["quality_tier"] == "strict":
            clusters[m["split"]][r["source"]].add(boot_cluster(m))
            biggest[m["split"]][boot_cluster(m)] += nb
        if r["source"] == "newspaper":
            news_groups[m["group"]] += nb
    out["split_sizes"] = {"%s/%s" % k: {"records": v[0], "bytes_MB": round(v[1] / 1e6, 3)} for k, v in sorted(by.items())}
    out["strict_eval_bootstrap_clusters"] = {
        s: {src: len(c) for src, c in sorted(d.items())} | {"total": sum(len(c) for c in d.values())}
        for s, d in sorted(clusters.items())}
    out["strict_eval_largest_clusters"] = {
        s: [{"cluster": k, "bytes_MB": round(v / 1e6, 3)} for k, v in c.most_common(3)] for s, c in sorted(biggest.items())}
    vals = sorted(news_groups.values(), reverse=True)
    out["newspaper_groups"] = {"n": len(vals), "largest_MB": round(vals[0] / 1e6, 3),
                               "median_MB": round(vals[len(vals) // 2] / 1e6, 3),
                               "n_dir_fallback": sum(1 for g in news_groups if g.startswith("news:dir:")),
                               "n_issue": sum(1 for g in news_groups if g.startswith("news:issue:"))}
    json.dump(out, open(os.path.join(HERE, "split_facts.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
