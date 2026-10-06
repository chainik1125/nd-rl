#!/usr/bin/env python3
"""Markdown shape / overlap / assembly tables of the three sets from artifacts/dsg/{shape,overlap,assemble,render}_*.json
(for data/dsg/README.md and run_ds_generator.md).  python3 dsg_tables.py"""
import json, os
D = 'artifacts/dsg'
J = lambda fn: json.load(open(fn)) if os.path.exists(fn) else None
RULES = ('AS', 'R', 'ANDI', 'ANDE1', 'ANDE2', 'ORI1', 'ORI2', 'ORE', 'IMPI', 'IMPE', 'NEGI', 'NEGE', 'DN', 'BOTE')
sets = {'C0 (a1)': 'c0_a1', 'G1': 'g1', 'G2': 'g2'}
sh = {k: list(J(f'{D}/shape_{v}.json').values())[0] for k, v in sets.items() if J(f'{D}/shape_{v}.json')}
pc = lambda x, d=1: f'{100 * x:.{d}f}'
print('### Shape table (per set, 155,000 proofs; shares in %; `dsg_shape.py`)\n')
print('| quantity | ' + ' | '.join(sh) + ' |\n|---|' + '---|' * len(sh))
print('| length 2 / 3 / 4 / 5 / 6 | ' + ' | '.join(' / '.join(str(s['len_hist'].get(str(L), 0)) for L in range(2, 7)) for s in sh.values()) + ' |')
print('| box depth 0 / 1 / 2 / 3 | ' + ' | '.join(' / '.join(pc(s['box_depth_hist'].get(str(b), 0) / s['n']) for b in range(4)) for s in sh.values()) + ' |')
for ru in RULES:
    print(f'| proofs with `{ru}` | ' + ' | '.join(pc(s['rule_share'][ru]) for s in sh.values()) + ' |')
print('| proofs with `ORE` (n) | ' + ' | '.join(f"{s['proofs_with_ore']} ({pc(s['ore_share'], 2)})" for s in sh.values()) + ' |')
print('| box inside an `ORE` branch (n) | ' + ' | '.join(f"{s['proofs_with_box_in_ore']} ({pc(s['box_in_ore_share'], 2)})" for s in sh.values()) + ' |')
print('| premises 0 / 1 / 2 / 3 | ' + ' | '.join(' / '.join(pc(s['n_prem_hist'].get(str(b), 0) / s['n']) for b in range(4)) for s in sh.values()) + ' |')
print('| mean premises | ' + ' | '.join(f"{s['mean_n_prem']:.2f}" for s in sh.values()) + ' |')
print('| mean lazy premises (labelled proofs) | ' + ' | '.join('not labelled' if s['mean_lazy_prem'] is None else f"{s['mean_lazy_prem']:.2f} ({s['lazy_labelled']})" for s in sh.values()) + ' |')
print('| contradictory-premise theorems | ' + ' | '.join(pc(s['contra_prem_share'], 2) for s in sh.values()) + ' |')
print('| final rule IMPI / IMPE / DN / ORI / ANDI / other | ' + ' | '.join(' / '.join(pc(s['last_rule_share'].get(k, 0)) for k in ('IMPI', 'IMPE', 'DN')) + ' / ' + pc(s['last_rule_share'].get('ORI1', 0) + s['last_rule_share'].get('ORI2', 0)) + ' / ' + pc(s['last_rule_share'].get('ANDI', 0)) + ' / ' + pc(1 - sum(s['last_rule_share'].get(k, 0) for k in ('IMPI', 'IMPE', 'DN', 'ORI1', 'ORI2', 'ANDI'))) for s in sh.values()) + ' |')
print('| pattern proofs: reductio / derived-ORE / depth-3 | ' + ' | '.join(f"{s['pattern_counts'].get('reductio', 0)} / {s['pattern_counts'].get('derived_ore', 0)} / {s['pattern_counts'].get('depth3', 0)}" for s in sh.values()) + ' |')
print('| mean tokens ND / Lean (`lean_seq`) | ' + ' | '.join(f"{s['mean_nd_tokens']:.1f} / {s['mean_lean_tokens']:.1f}" for s in sh.values()) + ' |')
print('\n### Assembly and render check\n')
print('| set | source pool classes | usable non-depth-3 per length 2 / 3 / 4 / 5 / 6 | fill from G1-knob pool per length | render 3,000 round-trip / 1,000 Lean / 300 negatives |\n|---|---|---|---|---|')
for name, tag in (('G1', 'g1'), ('G2', 'g2')):
    asm, rc = J(f'{D}/assemble_{tag}.json'), J(f'{D}/render_{tag}.json')
    if not asm: continue
    st = asm['stats']; per = asm['per_len']
    print(f"| {name} | {st['main:read']:,} read, {st['main:depth3']:,} depth-3, {st['main:excluded']:,} excluded | " + ' / '.join(f"{per[str(L)]['main'] if per[str(L)]['fill'] else '≥ 31,000'}" for L in range(2, 7)) + ' | ' + ' / '.join(f"{per[str(L)]['fill']:,} ({100 * per[str(L)]['fill_frac']:.0f} %)" if per[str(L)]['fill'] else '0' for L in range(2, 7)) + f" = {asm['fill_total']:,} ({100 * asm['fill_frac']:.1f} %) | {rc['roundtrip_identical']} / {rc['lean_positive_accepted']} / {rc['lean_negative_rejected']} |")
print('\n### Overlap table (set records matching a pool theorem: order-sensitive `thm` / renaming class / premise-order-insensitive class; `dsg_overlap.py`)\n')
ov = {k: list(J(f'{D}/overlap_{v}.json').values())[0]['vs'] for k, v in (('C0 (a1)', 'c0'), ('G1', 'g1'), ('G2', 'g2')) if J(f'{D}/overlap_{v}.json')}
pools = list(next(iter(ov.values())).keys())
print('| pool (n) | ' + ' | '.join(ov) + ' |\n|---|' + '---|' * len(ov))
for p in pools:
    n = next(iter(ov.values()))[p]['pool_n']
    print(f'| `{p}` ({n:,}) | ' + ' | '.join(f"{v[p]['thm']} / {v[p]['class']} / {v[p]['premise_order_insensitive']}" if p in v else '–' for v in ov.values()) + ' |')
