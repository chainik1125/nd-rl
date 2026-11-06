import json, sys, collections
res = json.load(open(sys.argv[1])); n = int(sys.argv[2]) if len(sys.argv) > 2 else 3
for r in res:
    print("#" * 100); print(r["prompt"])
    c = collections.Counter(s["nd"].split(":")[0][:40] if not s["ok"] and not s["nd"].startswith("N") else ("OK" if s["ok"] else "nd-but-lean-fail") for s in r["samples"])
    print(c)
    for s in r["samples"][:n]:
        print("  --", s["ok"], s["nd"][:300] if not s["nd"].startswith("N") else "")
        print("    ", s["raw"][:900])
