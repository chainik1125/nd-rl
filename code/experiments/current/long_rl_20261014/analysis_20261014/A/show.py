import json, sys, random
d = [json.loads(l) for l in open("feats.jsonl")]
pool, src, lo, hi, solved, n = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], int(sys.argv[6])
xs = [r for r in d if r["pool"] == pool and r["src"].startswith(src) and lo <= r["body"] <= hi and (solved == "any" or r["solved"] == int(solved))]
random.Random(int(sys.argv[7]) if len(sys.argv) > 7 else 0).shuffle(xs)
print(len(xs))
for r in xs[:n]:
    print("=" * 100); print(r["name"], r["src"], "body", r["body"], "solved", r["solved"], r.get("schema", ""), {k: r.get(k) for k in ("n_ok", "n_found", "found_pruned_min")})
    print(r["prompt"])
    if r.get("gold"):
        for ln in r["gold"].split(" ; "):
            if ": PR" not in ln and ln != "QED": print("   ", ln)
    if r.get("found_proof") and "--found" in sys.argv:
        print("  MODEL:")
        for ln in r["found_proof"].split(" ; "):
            if ": PR" not in ln and ln != "QED": print("   ", ln)
