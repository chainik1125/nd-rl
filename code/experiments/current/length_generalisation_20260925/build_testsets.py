#!/usr/bin/env python3
"""Assemble the length-generalisation test sets and check that they are disjoint from training data.

    python build_testsets.py [--check-only]

Writes testsets/std.jsonl (lengths 6-18, the main curve), testsets/matched.jsonl (the construction-style
control, when ab3/sealed/matched_L6-14.jsonl exists) and testsets/MANIFEST.json. Sources (common.SOURCES):

  L6-7    ab2/data/reserve.jsonl, only records built fresh for run 2 (origin ab2/gen, ab2/textbook; the old
          ladder records, some of them run 1's RL targets, are skipped), 300 per length
  L8-11   ab2/data/dev_L8-11.jsonl (300 per length, all of it)
  L12-14  ab2/sealed/sealed_L12-14.jsonl, at most 300 per length
  L15-18  ab3/sealed/sealed_L15-16.jsonl and sealed_L17-18.jsonl, at most 300 per length (skipped if missing)
  matched ab3/sealed/matched_L6-14.jsonl, at most 300 per length (skipped if missing)

Draw rule per length (same as ab2's assemble.py draw()): textbook share <= 40% unless the generator records
run out, textbook records round-robin over schemata, all with a fixed seed per (set, length). Draws depend
only on the source file's records at that length, so re-running after a missing file appears leaves the
existing lengths unchanged (the evaluator re-uses cached results by a hash of each length's theorems).

Disjointness is by pkey (premise-order-invariant atom-renaming class, recomputed from `thm`): every test
theorem is checked against the RL pool (ab2 targets_L6-11 = ab3 frozen targets), the pretraining data
(train_cap6, heldout_cap6) and every other test set. Overlapping records are dropped (and counted); the
standard sets are expected to have none. Matched records that overlap a standard test set are dropped from
the matched set.
"""
import argparse, collections, datetime, os, random, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (DRAW_SEED, PER_LEN, SOURCES, TESTSETS, TEXTBOOK_SHARE, TRAIN_FILES, iter_jsonl, pkey,
                    sha256_file, thm_of, write_json, write_jsonl)

KEEP = ('name', 'prompt', 'thm', 'L_true', 'source', 'schema', 'n_prem', 'origin', 'minlen_bound', 'gen_lines', 'family', 'label_kind')


def draw(recs, n, rng):
    """n records: textbook share <= TEXTBOOK_SHARE unless the generator runs out; textbook round-robin over schemata."""
    recs = sorted(recs, key=lambda r: r['name'])
    rng.shuffle(recs)
    g = [r for r in recs if r.get('source', 'gen') != 'textbook']
    t = [r for r in recs if r.get('source', 'gen') == 'textbook']
    n = min(n, len(recs))
    n_t = min(len(t), max(int(TEXTBOOK_SHARE * n), n - len(g)))
    n_g = n - n_t
    sch = collections.defaultdict(list)
    for r in t:
        sch[r.get('schema') or ''].append(r)
    tt, names = [], sorted(sch)
    while len(tt) < n_t:
        for s in names:
            if sch[s] and len(tt) < n_t:
                tt.append(sch[s].pop())
    sel = g[:n_g] + tt
    rng.shuffle(sel)
    return sel


def load_source(src, excl, report):
    """Records of one source file at its lengths, minus records whose pkey is in `excl` (training data)."""
    by = collections.defaultdict(list)
    n_read = n_bad_pkey = 0
    dropped = collections.Counter()
    for r in iter_jsonl(src['file']):
        L = r.get('L_true')
        if L not in src['lengths']:
            continue
        if src.get('origin_prefix') and not str(r.get('origin', '')).startswith(src['origin_prefix']):
            continue
        n_read += 1
        pk = pkey(thm_of(r))
        if r.get('pkey') and r['pkey'] != pk:
            n_bad_pkey += 1
        hit = [k for k, s in excl.items() if pk in s]
        if hit:
            for k in hit:
                dropped[k] += 1
            continue
        r = {k: r.get(k) for k in KEEP}
        r['pkey'] = pk
        by[L].append(r)
    report.update(n_read=n_read, stored_pkey_mismatch=n_bad_pkey, dropped_overlap_training=dict(dropped))
    return by


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check-only', action='store_true', help='report without writing the test files')
    a = ap.parse_args()
    os.makedirs(TESTSETS, exist_ok=True)
    print('loading training pkeys ...', flush=True)
    excl = {}
    for k, fn in TRAIN_FILES.items():
        excl[k] = {pkey(thm_of(r)) for r in iter_jsonl(fn)}
        print(f'  {k}: {len(excl[k])} classes ({fn})', flush=True)
    manifest = {'built_utc': datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
                'draw_seed': DRAW_SEED, 'per_len': PER_LEN, 'textbook_share': TEXTBOOK_SHARE,
                'train_files': {k: {'file': fn, 'sha256': sha256_file(fn), 'n_classes': len(excl[k])}
                                for k, fn in TRAIN_FILES.items()},
                'sources': [], 'units': {}, 'missing': []}
    sets = collections.defaultdict(list)
    for src in SOURCES:
        if not os.path.isfile(src['file']):
            print(f'  missing (skipped): {src["file"]}')
            manifest['missing'].append(src['file'])
            continue
        rep = {'set': src['set'], 'file': src['file'], 'sha256': sha256_file(src['file']), 'lengths': list(src['lengths'])}
        by = load_source(src, excl, rep)
        rep['available_by_len'] = {str(L): len(v) for L, v in sorted(by.items())}
        for L in src['lengths']:
            if not by.get(L):
                continue
            rng = random.Random(f'{DRAW_SEED}-{src["set"]}-{L}')
            sel = draw(by[L], PER_LEN, rng)
            for r in sel:
                r['set'] = src['set']
                r['src_file'] = os.path.basename(src['file'])
            sets[src['set']] += sel
        manifest['sources'].append(rep)
        print(f'  {src["set"]:8s} {os.path.basename(src["file"])}: read {rep["n_read"]}, available {rep["available_by_len"]}, '
              f'dropped (training overlap) {rep["dropped_overlap_training"]}, stored-pkey mismatches {rep["stored_pkey_mismatch"]}')

    # test sets vs each other: std must be pairwise disjoint across lengths and files; matched loses its overlaps
    std = sets.get('std', [])
    c = collections.Counter(r['pkey'] for r in std)
    dup = {k for k, v in c.items() if v > 1}
    if dup:
        print(f'  WARNING: {len(dup)} pkey classes occur twice in the standard sets; keeping the first')
        seen, keep = set(), []
        for r in std:
            if r['pkey'] in seen:
                continue
            seen.add(r['pkey'])
            keep.append(r)
        sets['std'] = std = keep
    manifest['std_internal_duplicates'] = len(dup)
    if sets.get('matched'):
        std_keys = {r['pkey'] for r in std}
        m0 = len(sets['matched'])
        sets['matched'] = [r for r in sets['matched'] if r['pkey'] not in std_keys]
        mc = collections.Counter(r['pkey'] for r in sets['matched'])
        manifest['matched_dropped_overlap_std'] = m0 - len(sets['matched'])
        manifest['matched_internal_duplicates'] = sum(1 for v in mc.values() if v > 1)

    # final disjointness assertion (recomputed on what is written)
    checks = {}
    for s, rows in sets.items():
        keys = [r['pkey'] for r in rows]
        checks[s] = {k: sum(1 for x in keys if x in e) for k, e in excl.items()}
        checks[s]['internal_duplicates'] = len(keys) - len(set(keys))
    if 'matched' in sets:
        sk = {r['pkey'] for r in sets['std']}
        checks['matched']['overlap_std'] = sum(1 for r in sets['matched'] if r['pkey'] in sk)
    manifest['disjointness'] = checks
    assert all(v == 0 for c in checks.values() for v in c.values()), checks

    for s, rows in sorted(sets.items()):
        rows.sort(key=lambda r: (r['L_true'], r['name']))
        by = collections.defaultdict(list)
        for r in rows:
            by[r['L_true']].append(r)
        for L, rs in sorted(by.items()):
            manifest['units'][f'{s}_L{L}'] = {'n': len(rs), 'by_source': dict(collections.Counter(r['source'] for r in rs)),
                                             'n_schemata': len({r['schema'] for r in rs if r['source'] == 'textbook'}),
                                             'src_file': rs[0]['src_file']}
        if not a.check_only:
            write_jsonl(os.path.join(TESTSETS, f'{s}.jsonl'), rows)
    print('units:')
    for u, v in manifest['units'].items():
        print(f'  {u:12s} n={v["n"]:4d} {v["by_source"]} schemata={v["n_schemata"]} ({v["src_file"]})')
    print('disjointness (overlap counts, all must be 0):', checks)
    if not a.check_only:
        write_json(os.path.join(TESTSETS, 'MANIFEST.json'), manifest)
        print('wrote', TESTSETS)


if __name__ == '__main__':
    main()
