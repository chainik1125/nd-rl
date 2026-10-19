#!/usr/bin/env python3
"""Resumable, deadline-aware wrapper around the fork's minlen.minlen (same search, same labels).

Differences from `minlen.py` main: imap_unordered + flush per record (a killed / deadline-stopped run keeps every
finished label), --deadline (wall-clock seconds; stop submitting and terminate after it), --resume (skip names already
in --out), per-worker address-space limit (--mem_gb; MemoryError -> unresolved with flag `oom`), and input order is
preserved as submission order (so a pre-sorted input labels its priority items first).
Output record = {name, thm, prompt, gen_lines} + minlen() result + secs, exactly like minlen.py.
"""
import argparse, json, os, sys, time, resource, collections
sys.path.insert(0, '/scratch/cowifornia/nd-rl/external/nd-takehome')
import minlen as M

MEM = None


def init(mem_gb):
    if mem_gb:   # limit = the worker's inherited (forked) address space + mem_gb of new allocations
        vm = int(open('/proc/self/statm').read().split()[0]) * resource.getpagesize()
        b = vm + int(mem_gb * 2**30)
        resource.setrlimit(resource.RLIMIT_AS, (b, b))


def work(args):
    rec, bound, tl = args
    t0 = time.time()
    flag = None
    try:
        r = M.minlen(rec['prompt'], bound, tl)
    except MemoryError:
        r, flag = None, 'oom'
    except RecursionError:
        r, flag = None, 'recursion'
    if r is None:        # outside the except block so the search frames (memo) are released before the result is pickled
        import gc; gc.collect()
        r = {'min_lines_ub': None, 'proof': None, 'timeout': False, flag: True, 'bound': bound, 'calls': None}
    r['secs'] = time.time() - t0
    out = {k: rec[k] for k in ('name', 'thm', 'prompt') if k in rec}
    out['gen_lines'] = rec.get('n_lines', rec.get('gen_lines'))
    out.update(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--bound', type=int, required=True); ap.add_argument('--time', type=float, required=True)
    ap.add_argument('--procs', type=int, default=40); ap.add_argument('--deadline', type=float, default=None)
    ap.add_argument('--mem_gb', type=float, default=1.2); ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--resume', action='store_true')
    a = ap.parse_args()
    recs = [json.loads(l) for l in open(a.inp) if l.strip()]
    done = set()
    if a.resume and os.path.exists(a.out):
        done = {json.loads(l)['name'] for l in open(a.out) if l.strip()}
    recs = [r for r in recs if r['name'] not in done]
    if a.limit:
        recs = recs[:a.limit]
    import multiprocessing as mp
    t0 = time.time(); n = 0; hist = collections.Counter(); stopped = False
    pool = mp.Pool(a.procs, initializer=init, initargs=(a.mem_gb,), maxtasksperchild=50)
    with open(a.out, 'a' if a.resume else 'w') as fo:
        it = pool.imap_unordered(work, [(r, a.bound, a.time) for r in recs], chunksize=1)
        while True:
            try:
                out = it.next(timeout=a.time * 3 + 60)   # a lost worker hangs imap forever; stop instead
            except StopIteration:
                break
            except mp.TimeoutError:
                print('NO RESULT for', a.time * 3 + 60, 's: stopping (lost worker?)', flush=True); stopped = True; break
            except mp.pool.MaybeEncodingError as e:
                print('RESULT ENCODING ERROR (skipped):', str(e)[:200], flush=True); continue
            fo.write(json.dumps(out) + '\n'); fo.flush(); n += 1
            hist[out['min_lines_ub'] if out['min_lines_ub'] is not None else ('timeout' if out.get('timeout') else ('oom' if out.get('oom') else 'none'))] += 1
            if n % 500 == 0:
                print(f'{n}/{len(recs)} {time.time()-t0:.0f}s {dict(sorted(hist.items(), key=str))}', flush=True)
            if a.deadline and time.time() - t0 > a.deadline:
                stopped = True; break
    pool.terminate()
    print(f'DONE n={n}/{len(recs)} stopped_at_deadline={stopped} secs={time.time()-t0:.0f} hist={dict(sorted(hist.items(), key=str))}', flush=True)


if __name__ == '__main__':
    main()
