#!/usr/bin/env python3
"""Run 1: generate completions with a local vLLM model for the prompt set (greedy + n samples at T).
  python run_vllm.py --model Qwen/Qwen3-Coder-30B-A3B-Instruct --prompts data/r1/prompts.jsonl --out artifacts/r1/gens_qwen30b.jsonl --n 8 --temperature 0.7
Writes one record per prompt: {id, outputs: [greedy, sample1..sampleN]}. Resumable (ids already in --out are skipped).
--filter form=lean,src=val36 restricts the prompt set; --thinking_off adds enable_thinking=False (Qwen3 hybrid models).
"""
import argparse, json, os, time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True); ap.add_argument('--prompts', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=8); ap.add_argument('--temperature', type=float, default=0.7); ap.add_argument('--max_tokens', type=int, default=1200)
    ap.add_argument('--max_model_len', type=int, default=12288); ap.add_argument('--gpu_mem', type=float, default=0.92); ap.add_argument('--tp', type=int, default=1)
    ap.add_argument('--filter', default=None); ap.add_argument('--limit', type=int, default=None); ap.add_argument('--thinking_off', action='store_true')
    ap.add_argument('--batch', type=int, default=256)
    a = ap.parse_args()
    from vllm import LLM, SamplingParams
    P = [json.loads(l) for l in open(a.prompts) if l.strip()]
    if a.filter:
        for kv in a.filter.split(','):
            k, v = kv.split('='); P = [p for p in P if str(p.get(k)) == v]
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)['id'] for l in open(a.out) if l.strip()}
    P = [p for p in P if p['id'] not in done]
    if a.limit: P = P[:a.limit]
    print(f'{len(P)} prompts to run ({len(done)} done)', flush=True)
    llm = LLM(model=a.model, tensor_parallel_size=a.tp, max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_mem, enable_prefix_caching=True, trust_remote_code=True)
    greedy = SamplingParams(temperature=0.0, max_tokens=a.max_tokens)
    samp = SamplingParams(temperature=a.temperature, top_p=0.95, n=a.n, max_tokens=a.max_tokens, seed=0)
    kw = {'chat_template_kwargs': {'enable_thinking': False}} if a.thinking_off else {}
    t0 = time.time()
    with open(a.out, 'a') as f:
        for s in range(0, len(P), a.batch):
            chunk = P[s:s + a.batch]
            msgs = [p['messages'] for p in chunk]
            g = llm.chat(msgs, greedy, use_tqdm=False, **kw)
            outs = [[o.outputs[0].text] for o in g]
            if a.n > 0:
                sm = llm.chat(msgs, samp, use_tqdm=False, **kw)
                for j, o in enumerate(sm):
                    outs[j] += [c.text for c in o.outputs]
            for p, o in zip(chunk, outs):
                f.write(json.dumps({'id': p['id'], 'model': a.model, 'outputs': o}) + '\n')
            f.flush()
            print(f'{s + len(chunk)}/{len(P)} {time.time() - t0:.0f}s', flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
