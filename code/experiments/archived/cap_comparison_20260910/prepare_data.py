"""Freeze nested cap6/10/12 training-only pools and explicit sampling weights.

No protected textbook/synthetic evaluation payload is opened. SFT uses complete
synthetic certificates; RL outputs contain prompts only. Short RL overlap with
SFT is intentional and authorized; the long component excludes the entire SFT
union. Probe panels are SFT-exposed diagnostics, never new held-out evidence.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import UTC, datetime
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT.parent / 'r1-sprint-20260909-1625'
EARLY = ROOT.parent / 'r1-cot-20260909'
ENCODER = OLD / 'snapshots/reasoning_v2/run.py'
sys.path.insert(0, str(ROOT / 'code/vendor'))
sys.path.insert(0, str(ENCODER.parent))
from nd_data import canonical_theorem, sequent_parts
from strict_verify import safe_verify
from proof_diagnostics import compress_proof
spec = importlib.util.spec_from_file_location('cap_data_frozen_encoder', ENCODER)
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
FIELDS = ('canonical_theorem_sha256', 'nonempty_premise_context_sha256', 'closed_formula_skeleton_sha256')
SOURCES = {
 'short': (OLD/'data/sft_cap6.jsonl', '281cc6525814fbf936f6add25c965f19f1ce5e1115561d4b5255f61f2bec14e4'),
 'mid': (OLD/'data/cap_extensions/additional_cap10.jsonl', 'fdd6187c1b3b8c1d008d6debce6a3f63e2e3ab8c47386fb45e12b0c9e5563f33'),
 'cap12': (EARLY/'data/cap_extensions/additional_cap12.jsonl', '502f595ca1f5a156fdcd4a21202986525152f5430b40dd87897ca6c7a6e07fee'),
 'long_rl': (EARLY/'data/cap_extensions/rl_long_disjoint.jsonl', 'f2e0eb10d3de8fe9fed3543394c921cdf3872690d2853e2ac7b8da5f32cba523'),
 'book_prompt_container': (EARLY/'data/rl_with_textbook.jsonl', 'a0fb288342897c928595f41349936ce2145f3124c6acae7939d9daae49aa9359'),
 'all_book_guards': (EARLY/'data/all_textbook_guards.json', '408ce2ea9bb136566db7745f268077e9ef79fa033de2ef04791544d4130b8bd4'),
 'heldout_book_guards': (EARLY/'data/heldout_guards.json', '533ed4c15f229159d2a3aacbe745bc0df3ec1175ccba96f579dd9c78866dc2ce'),
 'supervised_guards': (OLD/'data/cap_extensions/supervised_guards.json', '6ea1cc7b12ff42c1e8b7521f92157a1ea5e461e3c7de1479e9bff62fe7bd6df7'),
 'rl_guards': (OLD/'data/cap_extensions/rl_guards.json', 'bcf1ca50f8f5da685cf0e9f153253a4e9a8151d2fabe21edbdb8deb7af5bd37e'),
 'split_protocol': (EARLY/'data/split_protocol.json', 'a1cf987a348cb84d4408bbeeed70b842b9288db17a4da5a7336fc5e49b1a896b'),
}
STRATA = {6: {'short': Fraction(1)}, 10: {'short': Fraction(3,4), 'mid': Fraction(1,4)},
          12: {'short': Fraction(3,4), 'mid': Fraction(1,8), 'high': Fraction(1,8)}}
SEED = 202609101958


def require(ok, reason):
    if not ok: raise ValueError(reason)


def digest(raw): return hashlib.sha256(raw).hexdigest()
def canonical_hash(prompt): return digest(canonical_theorem(prompt).encode())
def dumps(value): return json.dumps(value, sort_keys=True, separators=(',', ':'))


def read_input(key, hashes):
    require(key in SOURCES, 'Only explicitly allowlisted training/guard inputs may be opened')
    path, expected = SOURCES[key]
    require(not any('sealed' in part.lower() for part in path.parts)
            and not any(x in path.name.lower() for x in ('textbook_dev', 'textbook_final', 'synthetic_dev')),
            'Protected payload read denied')
    raw = path.read_bytes()
    require(digest(raw) == expected, 'Frozen authorized input changed: ' + str(path))
    hashes[str(path)] = expected
    return json.loads(raw) if path.suffix == '.json' else [json.loads(x) for x in raw.splitlines() if x.strip()]


def check_guard(row, guard):
    for key, field in zip(core.guard_keys(row['prompt']), FIELDS, strict=True):
        require(key is None or key not in guard[field], 'Protected theorem/context/skeleton in ' + row.get('name', 'row'))


def validate_label(row, lo, hi, guard, origin):
    prompt, body = core.split_record(row)
    require(type(prompt) is str and type(body) is str, 'Literal complete prompt and body required')
    require(body.split().count('QED') == 1 and body.split()[-1] == 'QED', 'Complete certificate must end at its only QED')
    canonical = canonical_hash(prompt)
    require(row.get('canonical_hash') == canonical, 'Canonical source identity mismatch')
    check_guard({'prompt': prompt, 'name': row.get('name', '')}, guard)
    ok, reason, n = safe_verify(prompt, body)
    require(ok, 'Invalid complete source certificate: ' + reason)
    require(type(row.get('n_lines')) is int and row['n_lines'] == n and lo <= n <= hi, 'Source length/cap mismatch')
    compressed = compress_proof(prompt, body)
    require(compressed['raw_valid'] and compressed['raw_written_length'] == n, 'Compression replay mismatch')
    ok, reason, cn = safe_verify(prompt, compressed['compressed_proof'])
    require(ok and cn == compressed['compressed_length'] <= n, 'Compressed source certificate invalid: ' + reason)
    return {'canonical_hash': canonical, 'name': row.get('name', canonical), 'prompt': prompt, 'body': body,
            'role': 'sft_train', 'family': row.get('family', 'unknown'), 'n_lines': n,
            'source_compressed_lines': cn, 'source_dataset': origin,
            'source_record_sha256': digest(dumps(row).encode()),
            'source_full_certificate_sha256': digest(body.encode())}


def unique(rows):
    result = {}
    for row in rows:
        key = row['canonical_hash']
        require(key not in result, 'Repeated canonical class; atom-renamings cannot inflate diversity')
        result[key] = row
    return result


def weighted(buckets, masses):
    require(sum(masses.values()) == 1, 'Sampling bucket masses must sum exactly to one')
    result = []
    for bucket, mass in masses.items():
        rows = buckets[bucket]
        require(bool(rows) and mass > 0, 'Each sampling bucket must be nonempty and positive')
        weight = mass / len(rows)
        for row in rows:
            result.append({**row, 'sampling_bucket': bucket, 'sampling_weight': float(weight),
                           'sampling_weight_rational': f'{weight.numerator}/{weight.denominator}'})
    unique(result)
    require(all(math.isfinite(x['sampling_weight']) and x['sampling_weight'] > 0 for x in result), 'Invalid weight')
    require(abs(math.fsum(x['sampling_weight'] for x in result) - 1) < 1e-12, 'Floating sampling normalization differs')
    return sorted(result, key=lambda x: x['canonical_hash'])


def balanced(rows, count, seed, excluded=frozenset()):
    groups = defaultdict(list)
    for row in rows:
        if row['canonical_hash'] not in excluded: groups[(row['family'], row.get('n_lines', 0))].append(row)
    for key, group in groups.items():
        group.sort(key=lambda x: digest(f'{seed}:{x["canonical_hash"]}'.encode()), reverse=True)
    result = []
    while len(result) < count:
        moved = False
        for key in sorted(groups):
            if groups[key] and len(result) < count: result.append(groups[key].pop()); moved = True
        require(moved, 'Insufficient unique rows for fixed panel')
    return result


def prompt_only(row, role, source):
    # Construct a whitelist; never propagate a body, target, label or proof.
    return {'canonical_hash': canonical_hash(row['prompt']), 'prompt': row['prompt'],
            'name': row.get('name', canonical_hash(row['prompt'])), 'family': row.get('family', 'unknown'),
            'role': role, 'source_dataset': source}


def audit_rl(buckets, supervised_ids, book_ids, guards):
    require(set(buckets) == {'short', 'long', 'book'}, 'Fixed common RL buckets required')
    for bucket, rows in buckets.items():
        unique(rows)
        for row in rows:
            require(set(row) <= {'canonical_hash', 'prompt', 'name', 'family', 'role', 'source_dataset'}, 'RL cannot carry certificate/reference fields')
            require(row['canonical_hash'] == canonical_hash(row['prompt']), 'RL canonical mismatch')
            check_guard(row, guards)
            if bucket == 'short': require(row['canonical_hash'] in supervised_ids, 'Short RL must come from common supervised base')
            if bucket == 'long': require(row['canonical_hash'] not in supervised_ids, 'Long RL overlaps supervised/retention union')
            if bucket == 'book': require(row['canonical_hash'] in book_ids, 'Unauthorized textbook RL class')
    require({r['canonical_hash'] for r in buckets['book']} == book_ids, 'Exact original14 textbook prompt classes required')
    unique([row for rows in buckets.values() for row in rows])


def stats(rows):
    return {'rows': len(rows), 'canonical_classes': len(unique(rows)),
            'raw_length_counts': dict(sorted(Counter(x.get('n_lines') for x in rows if 'n_lines' in x).items())),
            'compressed_length_counts': dict(sorted(Counter(x.get('source_compressed_lines') for x in rows if 'source_compressed_lines' in x).items())),
            'family_counts': dict(sorted(Counter(x['family'] for x in rows).items())),
            'sampling_mass_by_bucket': {k: math.fsum(x['sampling_weight'] for x in rows if x.get('sampling_bucket') == k)
                 for k in sorted({x['sampling_bucket'] for x in rows if 'sampling_bucket' in x})}}


def write_rows(path, rows):
    with path.open('x') as stream:
        for row in rows: stream.write(dumps(row) + '\n')


def build(out):
    require(out.resolve() == (ROOT/'data').resolve(), 'Only owned experiment data directory may be written')
    out.mkdir(parents=True, exist_ok=True)
    require(not (out/'manifest.json').exists() and not list(out.glob('sft_*.jsonl')), 'Frozen data already exists; refuse replacement')
    source_hashes = {str(Path(__file__)): digest(Path(__file__).read_bytes()), str(ENCODER): digest(ENCODER.read_bytes())}
    for p in (ROOT/'code/vendor').rglob('*.py'): source_hashes[str(p)] = digest(p.read_bytes())
    source_hashes[str(ENCODER.parent/'scratch_variants.py')] = digest((ENCODER.parent/'scratch_variants.py').read_bytes())
    inputs = {}
    raw = {key: read_input(key, inputs) for key in SOURCES}
    sg = {k:set(raw['supervised_guards'][k]) for k in FIELDS};rg = {k:set(raw['rl_guards'][k]) for k in FIELDS}
    require([len(sg[k]) for k in FIELDS] == [785,98,31], 'Expected all145 textbook plus640 synthetic canonical guards')
    require([len(rg[k]) for k in FIELDS] == [771,91,28], 'Expected heldout131 textbook plus640 synthetic canonical guards')
    for field in FIELDS:
        require(set(raw['all_book_guards'][field]) <= sg[field], 'Missing textbook supervised guards')
        require(set(raw['heldout_book_guards'][field]) <= rg[field], 'Missing textbook heldout RL guards')
    book_ids = set(raw['all_book_guards'][FIELDS[0]]) - set(raw['heldout_book_guards'][FIELDS[0]])
    require(len(book_ids)==14 and raw['split_protocol']['counts']=={'dev':58,'final':73,'train':14}, 'Textbook allocation changed')
    buckets = {'short':[], 'mid':[], 'high':[]};tokenizer = core.ScratchTokenizer();encoding = {}
    cap12_by = unique(raw['cap12']);mid_by = unique(raw['mid'])
    require(set(mid_by)=={h for h,x in cap12_by.items() if 7<=x['n_lines']<=10}, 'Cap10 is not exact mid-length subset of cap12 source')
    for h,x in mid_by.items():
        require(core.split_record(x)==core.split_record(cap12_by[h]), 'Nested mid source literal certificate differs')
    selected_sources = [('short',raw['short'],2,6), ('mid',raw['mid'],7,10),
                        ('high',[x for x in raw['cap12'] if x['n_lines']>10],11,12)]
    for bucket, rows, lo, cap in selected_sources:
        for i, source in enumerate(rows):
            row = validate_label(source,lo,cap,sg,'short' if bucket=='short' else ('mid' if bucket=='mid' else 'cap12'))
            enc = {mode:core.encode_training(row,tokenizer,mode,16384,max_proof_lines=cap,scratch_style='dependency') for mode in ('direct','cot')}
            encoding[row['canonical_hash']] = {mode:{'prompt_tokens':len(v['prompt_ids']), 'response_tokens':len(v['response_ids']),
                                                     'total_tokens':len(v['prompt_ids'])+len(v['response_ids'])} for mode,v in enc.items()}
            buckets[bucket].append(row)
            if (i+1)%10000==0: print(f'verified {bucket} {i+1}/{len(rows)}',flush=True)
        print(f'verified {bucket} complete: {len(rows)}',flush=True)
    require([len(buckets[x]) for x in ('short','mid','high')]==[60000,3372,1597], 'Source population changed')
    union = unique([x for rows in buckets.values() for x in rows]);union_ids=set(union)
    require(len(union)==64969, 'Supervised strata overlap')
    panel_short=balanced(buckets['short'],512,SEED);panel_ids={x['canonical_hash'] for x in panel_short}
    short_rl=balanced(buckets['short'],512,SEED+1,panel_ids)
    rb={'short':[prompt_only(x,'rl_train','shared_short_supervision') for x in short_rl],
        'long':[prompt_only(x,'rl_train','long_rl') for x in raw['long_rl']],
        'book':[prompt_only(x,'rl_train','book_prompt_container') for x in raw['book_prompt_container'] if x.get('family')=='textbook_train']}
    audit_rl(rb,union_ids,book_ids,rg)
    require(len(rb['long'])==1303, 'Long RL population changed')
    common_rl=weighted(rb,{'short':Fraction(1,2),'long':Fraction(9,20),'book':Fraction(1,20)})
    artifacts={};arms={}
    def save(name,rows):
        write_rows(out/name,rows);artifacts[name]=digest((out/name).read_bytes());return str(out/name)
    for cap in (6,10,12):
        rows=weighted(buckets,STRATA[cap]);path=save(f'sft_cap{cap}.jsonl',rows)
        ids=[x['canonical_hash'] for x in rows]
        arms[str(cap)]={'supervision':path,'retention':path,'retention_is_identical_weighted_supervision_pool':True,
                       'identical_direct_cot_source_membership_and_weights':True,'actual_supervised_cap':cap,
                       'stratum_probabilities':{k:str(v) for k,v in STRATA[cap].items()},**stats(rows),
                       'canonical_membership_sha256':digest(('\n'.join(sorted(ids))+'\n').encode()),
                       'encoding':{mode:{'maximum_total_tokens':max(encoding[h][mode]['total_tokens'] for h in ids),
                                         'maximum_response_tokens':max(encoding[h][mode]['response_tokens'] for h in ids),
                                         'weighted_mean_response_tokens':math.fsum(x['sampling_weight']*encoding[x['canonical_hash']][mode]['response_tokens'] for x in rows),
                                         'weighted_mean_total_tokens':math.fsum(x['sampling_weight']*encoding[x['canonical_hash']][mode]['total_tokens'] for x in rows)} for mode in ('direct','cot')}}
    save('rl_common.jsonl',common_rl)
    mid_panel=balanced(buckets['mid'],128,SEED+2);high_panel=balanced(buckets['high'],64,SEED+3)
    def probe(rows):return [{**prompt_only(x,'synthetic_training_probe',x['source_dataset']), 'reference_lines':x['n_lines'],
                              'source_compressed_lines':x['source_compressed_lines'], 'diagnostic_is_sft_exposed':True} for x in rows]
    save('pretraining_probe_common_short.jsonl',probe(panel_short))
    save('pretraining_probe_cap6_mixed.jsonl',probe(panel_short))
    save('pretraining_probe_cap10_mixed.jsonl',probe(panel_short[:384]+mid_panel))
    save('pretraining_probe_cap12_mixed.jsonl',probe(panel_short[:384]+mid_panel[:64]+high_panel))
    save('source_encoding_audit.jsonl',[{'canonical_hash':h,**encoding[h]} for h in sorted(encoding)])
    for name,guards in [('supervised_guards.json',sg),('rl_guards.json',rg),
                        ('long_rl_union_guards.json',{**rg,FIELDS[0]:rg[FIELDS[0]]|union_ids})]:
        value={k:sorted(v) for k,v in guards.items()};(out/name).write_text(json.dumps(value,sort_keys=True,indent=2)+'\n');artifacts[name]=digest((out/name).read_bytes())
    for cap in (6,10,12):
        if cap>6: require(set(unique(weighted(buckets,STRATA[6 if cap==10 else 10]))) <= set(unique(weighted(buckets,STRATA[cap]))), 'Nested cap membership failed')
    require(not ({x['canonical_hash'] for x in rb['long']} & union_ids), 'Long overlap')
    require({x['canonical_hash'] for x in rb['short']} <= set(unique(buckets['short'])), 'Short RL is not common-base subset')
    require(not ({x['canonical_hash'] for x in rb['short']} & panel_ids), 'Fixed short probe overlaps short RL target component')
    for path,sha in {**inputs,**source_hashes}.items():require(digest(Path(path).read_bytes())==sha,'Input changed during construction')
    manifest={'status':'complete','created_utc':datetime.now(UTC).isoformat(),'seed':SEED,'input_sha256':inputs,'source_sha256':source_hashes,
      'output_sha256':artifacts,'arms':arms,'common_rl':{'path':str(out/'rl_common.jsonl'),**stats(common_rl),
       'probabilities':{'short':'1/2','long':'9/20','book':'1/20'},'component_counts':{k:len(v) for k,v in rb.items()},
       'long_disjoint_from_union_of_all_sft_and_retention':True,'short_overlap_with_shared_sft_intentional':True,
       'book_classes':14,'book_population':145,'book_labels_read_or_emitted':False,'prompt_only':True},
      'probe':{'common_short':str(out/'pretraining_probe_common_short.jsonl'),'rows':512,'same_literal_rows_all_six_runs':True,
               'common_short_disjoint_from_short_rl_component':True,'sft_exposed_training_diagnostic':True,
               'mixed_owncap_counts':{'6':{'short':512},'10':{'short':384,'mid':128},'12':{'short':384,'mid':64,'high':64}},
               'mixed_panels_are_different_populations_not_crosscap_accuracy_comparisons':True},
      'synthetic_dev':{'path':str(EARLY/'data/synthetic_dev.jsonl'),'rows':256,
                       'sha256_copied_from_prior_metadata':'b7f5e2a1de1fd7fb3806703344cab56e338ca642b62b810c5e6a5aedb283596f',
                       'payload_read_or_hashed':False,'exclusion_via_existing_640_synthetic_canonical_guards':True},
      'protected_textbook':{'train':14,'development':58,'old_final':73,'payloads_or_results_read':False,
                           'all145_excluded_from_supervision':True,'heldout131_structural_guards_preserved':True},
      'certificate_audit':{'distinct_full_labels_verified':64969,'compressed_and_reverified':64969,'direct_cot_labels_encoded':129938,
                           'compression_is_diagnostic_only_original_full_certificates_are_training_labels':True},
      'limitations':['Cap increases also introduce eight new synthetic families and more unique classes; not a pure proof-length intervention.',
                     'SFT and retention must honor sampling_weight; uniform row sampling would implement a different experiment.',
                     'Direct/CoT token lengths differ; equal update or example counts do not imply matched compute.',
                     'Reference/raw/compressed certificate lengths are upper bounds, not proven shortest derivations.',
                     'Old development/final benchmarks have historical project exposure; no new payload or result access in this build.',
                     'Common short probe and own-cap probes are supervised training-domain diagnostics, not independent generalization sets.']}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    (out/'README.md').write_text('Frozen synthetic cap comparison data. Every SFT certificate is complete, strictly verified, compressed diagnostically and encoded in both modes.\n\nThe nested unique label sets share all60,000 short rows. Sampling weights implement cap6=100%short; cap10=75%short/25%7–10; cap12=75%short/12.5%7–10/12.5%11–12. Retention uses exactly the same per-cap file and weights. The higher caps add families and diversity as well as lengths.\n\nCommon RL is50%512 shared-short targets (intentional SFT overlap),45%1,303 long targets disjoint from all64,969 supervised classes, and5%the original14 textbook prompt-only classes. No textbook proof label is read. The512 common-short probe is identical across caps and is SFT-exposed; own-cap mixed probes are distinct populations. The old256 synthetic development payload is not opened here; existing640 synthetic guard hashes exclude it.\n\nUse manifest.json as the authoritative source, identity, cap, weight, token-length and limitation record. No heldout textbook payload or result was read.\n')
    print(json.dumps({'status':'complete','manifest':str(out/'manifest.json'),'manifest_sha256':digest((out/'manifest.json').read_bytes()),
                       'rows':{k:v['rows'] for k,v in arms.items()},'rl_rows':len(common_rl),
                       'max_context':{k:v['encoding'] for k,v in arms.items()}},indent=2),flush=True)


class Tests(unittest.TestCase):
    def test_unique_and_weights(self):
        a={'canonical_hash':'a','family':'x'};b={'canonical_hash':'b','family':'y'}
        with self.assertRaises(ValueError):unique([a,a])
        rows=weighted({'s':[a],'l':[b]}, {'s':Fraction(3,4),'l':Fraction(1,4)})
        self.assertEqual([r['sampling_weight'] for r in rows],[.75,.25])
        with self.assertRaises(ValueError):weighted({'s':[a]}, {'s':Fraction(1,2)})
    def test_prompt_whitelist_and_long_overlap(self):
        row={'prompt':'THM P SEQ P PRF','name':'x','family':'x','body':'secret','reference_lines':999}
        out=prompt_only(row,'rl_train','invented');self.assertNotIn('body',out);self.assertNotIn('reference_lines',out)
        guards={k:set() for k in FIELDS}
        with self.assertRaises(ValueError):audit_rl({'short':[],'long':[out],'book':[]},{out['canonical_hash']},set(),guards)
    def test_complete_source_cap_and_canonical_guard(self):
        prompt='THM P SEQ P PRF';body='N1 P : PR ; QED';row={'prompt':prompt,'body':body,'n_lines':1,'canonical_hash':canonical_hash(prompt)}
        guards={k:set() for k in FIELDS};self.assertEqual(validate_label(row,1,1,guards,'invented')['n_lines'],1)
        for bad in [{**row,'body':body+' junk'}, {**row,'n_lines':2}, {**row,'canonical_hash':'wrong'}]:
            with self.assertRaises(ValueError):validate_label(bad,1,1,guards,'invented')
        with self.assertRaises(ValueError):validate_label(row,2,6,guards,'invented')
        blocked={**guards,FIELDS[0]:{row['canonical_hash']}}
        with self.assertRaises(ValueError):validate_label(row,1,1,blocked,'invented')
        self.assertEqual(canonical_hash('THM Q SEQ Q PRF'),row['canonical_hash'])
    def test_unknown_input_denied_before_io(self):
        with self.assertRaises(ValueError):read_input('textbook_final',{})
    def test_fixed_panel_is_unique_repeatable_and_excludes(self):
        rows=[{'canonical_hash':str(i),'family':str(i%2),'n_lines':2+i%3} for i in range(20)]
        a=balanced(rows,8,123,{'0','1'});self.assertEqual(a,balanced(rows,8,123,{'0','1'}));self.assertEqual(len(unique(a)),8)
        self.assertFalse({x['canonical_hash'] for x in a}&{'0','1'})

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,default=ROOT/'data');parser.add_argument('--self-test',action='store_true');args=parser.parse_args()
    if args.self_test: unittest.main(argv=[sys.argv[0]],exit=True)
    else: build(args.out)
