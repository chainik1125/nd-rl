#!/usr/bin/env python3
"""Rent one RunPod GPU, run the pipeline there, pull the artifacts back, terminate the pod.

  python pod.py --name run0                          # A40, defaults (5 min pretrain + 5 min RL)
  python pod.py --name run0 -- --pretrain_secs 120   # extra args go to pipeline.py
  python pod.py --list | --kill <id|all>

Adapted from the take-home's pod.py. The pod is terminated in a `finally`, including on Ctrl-C.
Needs RUNPOD_API_KEY (the take-home's .envrc.local has it) and ~/.ssh/id_ed25519.
"""
import argparse, json, os, shlex, subprocess, sys, time, urllib.request

API = 'https://api.runpod.io/graphql'
IMAGE = 'runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04'
HERE = os.path.dirname(os.path.abspath(__file__))
LOCAL_ART = os.path.join(HERE, '..', '..', '..', '..', 'artifacts', 'pretrain_poc')   # repo artifacts/ is gitignored


def gql(query, key):
    req = urllib.request.Request(f'{API}?api_key={key}', method='POST', data=json.dumps({'query': query}).encode(),
                                 headers={'Content-Type': 'application/json', 'User-Agent': 'curl/8.7.1'})
    out = json.load(urllib.request.urlopen(req))
    if 'errors' in out:
        raise SystemExit(json.dumps(out['errors'], indent=2))
    return out['data']


def create(key, gpu, name, pubkey, cloud):
    q = ('mutation { podFindAndDeployOnDemand(input: {'
         f'cloudType: {cloud}, gpuCount: 1, volumeInGb: 0, containerDiskInGb: 40, minVcpuCount: 4, minMemoryInGb: 16, '
         f'gpuTypeId: "{gpu}", name: "{name}", imageName: "{IMAGE}", ports: "22/tcp", volumeMountPath: "/workspace", '
         f'supportPublicIp: true, env: [{{key: "PUBLIC_KEY", value: "{pubkey}"}}]'
         '}) { id costPerHr } }')
    return gql(q, key)['podFindAndDeployOnDemand']


def wait_ssh(key, pod_id, timeout=600):
    q = ('query { pod(input: {podId: "%s"}) { runtime { ports { ip publicPort privatePort isIpPublic } } } }' % pod_id)
    t0 = time.time()
    while time.time() - t0 < timeout:
        rt = gql(q, key)['pod']['runtime']
        for p in (rt or {}).get('ports') or []:
            if p['privatePort'] == 22 and p['isIpPublic']:
                return p['ip'], p['publicPort']
        time.sleep(5)
    raise SystemExit('pod never exposed ssh')


def ssh_args(ip, port):
    return ['-p', str(port), '-o', 'StrictHostKeyChecking=no', '-o', 'UserKnownHostsFile=/dev/null',
            '-o', 'LogLevel=ERROR', '-o', 'ServerAliveInterval=30', f'root@{ip}']


def terminate(key, pod_id):
    gql('mutation { podTerminate(input: {podId: "%s"}) }' % pod_id, key)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', default='nd-poc')
    ap.add_argument('--gpu', default='NVIDIA A40')
    ap.add_argument('--cloud', default='SECURE', choices=['SECURE', 'COMMUNITY'])
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--kill', default=None, metavar='ID|all')
    ap.add_argument('rest', nargs='*', help='args passed to pipeline.py (after --)')
    a = ap.parse_args()
    key = os.environ.get('RUNPOD_API_KEY') or sys.exit('RUNPOD_API_KEY unset')

    pods = gql('query { myself { clientBalance pods { id name desiredStatus costPerHr } } }', key)['myself']
    if a.list:
        print(json.dumps(pods, indent=1))
        return
    if a.kill:
        for p in pods['pods']:
            if a.kill in (p['id'], 'all'):
                terminate(key, p['id'])
                print('terminated', p['id'], p['name'])
        return
    live = [p for p in pods['pods'] if p['desiredStatus'] == 'RUNNING']
    print(f"balance ${pods['clientBalance']:.2f}; running pods: {live}", flush=True)

    pubkey = open(os.path.expanduser('~/.ssh/id_ed25519.pub')).read().strip()
    pod = create(key, a.gpu, a.name, pubkey, a.cloud)
    if pod is None and a.cloud == 'SECURE':
        print('no secure-cloud capacity, trying community', flush=True)
        pod = create(key, a.gpu, a.name, pubkey, 'COMMUNITY')
    if pod is None:
        raise SystemExit(f'no {a.gpu} capacity right now')
    print(f"pod {pod['id']} at ${pod['costPerHr']}/hr", flush=True)
    t0 = time.time()
    try:
        ip, port = wait_ssh(key, pod['id'])
        print(f'ssh -p {port} root@{ip}   (up after {time.time() - t0:.0f}s)', flush=True)
        for _ in range(60):
            if subprocess.run(['ssh', *ssh_args(ip, port), 'true'], capture_output=True).returncode == 0:
                break
            time.sleep(5)
        else:
            raise SystemExit('ssh never accepted the key')
        ssh = 'ssh ' + ' '.join(ssh_args(ip, port))
        subprocess.run(f'tar czf - -C {shlex.quote(HERE)} pipeline.py remote.sh | {ssh} "mkdir -p /workspace/poc && '
                       f'tar xzf - -C /workspace/poc"', shell=True, check=True)
        out = f'/workspace/poc/artifacts/{a.name}'
        cmd = f'mkdir -p /workspace/poc/artifacts && bash /workspace/poc/remote.sh {out} ' + ' '.join(map(shlex.quote, a.rest))
        rc = subprocess.run(['ssh', *ssh_args(ip, port), 'bash', '-lc', shlex.quote(cmd)]).returncode
        os.makedirs(LOCAL_ART, exist_ok=True)
        subprocess.run(f'{ssh} "cd /workspace/poc/artifacts && tar czf - {a.name} {a.name}.log" | tar xzf - -C {shlex.quote(LOCAL_ART)}',
                       shell=True, check=False)
        print(f'artifacts in {os.path.abspath(os.path.join(LOCAL_ART, a.name))}; remote rc {rc}', flush=True)
    finally:
        terminate(key, pod['id'])
        mins = (time.time() - t0) / 60
        print(f'terminated {pod["id"]} after {mins:.1f} min (~${pod["costPerHr"] * mins / 60:.2f})', flush=True)


if __name__ == '__main__':
    main()
