#!/usr/bin/env python3
"""One long-lived RunPod GPU for the autoresearch loop (boot + clone once, run many experiments).

  python pod.py up                      # rent an A40, wait for ssh, clone the fork; saves pod.json
  python pod.py run <name> [args...]    # upload harness.py + pretrain.py, run, pull artifacts/<name>
  python pod.py down                    # terminate the pod in pod.json
  python pod.py list | kill <id|all>

Adapted from pretrain_poc/pod.py, which rented one pod per run. Needs RUNPOD_API_KEY
(~/repos/nd-takehome/.envrc.local has it) and ~/.ssh/id_ed25519.
"""
import json, os, shlex, subprocess, sys, time, urllib.request

API = 'https://api.runpod.io/graphql'
IMAGE = 'runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04'
GPU = 'NVIDIA A40'
HERE = os.path.dirname(os.path.abspath(__file__))
LOCAL_ART = os.path.join(HERE, '..', '..', '..', '..', 'artifacts', 'autoresearch')   # repo artifacts/ is gitignored
STATE = os.path.join(LOCAL_ART, 'pod.json')
REMOTE = '/workspace/autoresearch'


def gql(query, key):
    req = urllib.request.Request(f'{API}?api_key={key}', method='POST', data=json.dumps({'query': query}).encode(),
                                 headers={'Content-Type': 'application/json', 'User-Agent': 'curl/8.7.1'})
    out = json.load(urllib.request.urlopen(req))
    if 'errors' in out:
        raise SystemExit(json.dumps(out['errors'], indent=2))
    return out['data']


def create(key, name, pubkey, cloud):
    q = ('mutation { podFindAndDeployOnDemand(input: {'
         f'cloudType: {cloud}, gpuCount: 1, volumeInGb: 0, containerDiskInGb: 40, minVcpuCount: 4, minMemoryInGb: 16, '
         f'gpuTypeId: "{GPU}", name: "{name}", imageName: "{IMAGE}", ports: "22/tcp", volumeMountPath: "/workspace", '
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


def ssh_args(st):
    return ['-p', str(st['port']), '-o', 'StrictHostKeyChecking=no', '-o', 'UserKnownHostsFile=/dev/null',
            '-o', 'LogLevel=ERROR', '-o', 'ServerAliveInterval=30', f'root@{st["ip"]}']


def ssh_run(st, cmd, check=True):
    return subprocess.run(['ssh', *ssh_args(st), 'bash', '-lc', shlex.quote(cmd)], check=check).returncode


def upload(st, files):
    ssh = 'ssh ' + ' '.join(ssh_args(st))
    subprocess.run(f'tar czf - -C {shlex.quote(HERE)} {" ".join(files)} | {ssh} "mkdir -p {REMOTE} && tar xzf - -C {REMOTE}"',
                   shell=True, check=True)


def terminate(key, pod_id):
    gql('mutation { podTerminate(input: {podId: "%s"}) }' % pod_id, key)


def load_state():
    if not os.path.exists(STATE):
        raise SystemExit(f'no {STATE}; run `pod.py up` first')
    return json.load(open(STATE))


def main():
    cmd, rest = (sys.argv[1] if len(sys.argv) > 1 else 'list'), sys.argv[2:]
    key = os.environ.get('RUNPOD_API_KEY') or sys.exit('RUNPOD_API_KEY unset')
    pods = gql('query { myself { clientBalance pods { id name desiredStatus costPerHr } } }', key)['myself']
    os.makedirs(LOCAL_ART, exist_ok=True)

    if cmd == 'list':
        print(json.dumps(pods, indent=1))
    elif cmd == 'kill':
        for p in pods['pods']:
            if rest[0] in (p['id'], 'all'):
                terminate(key, p['id'])
                print('terminated', p['id'], p['name'])
        if os.path.exists(STATE):
            os.remove(STATE)
    elif cmd == 'down':
        st = load_state()
        terminate(key, st['id'])
        os.remove(STATE)
        mins = (time.time() - st['t0']) / 60
        print(f'terminated {st["id"]} after {mins:.1f} min (~${st["cost"] * mins / 60:.2f})')
    elif cmd == 'up':
        live = [p for p in pods['pods'] if p['desiredStatus'] == 'RUNNING']
        print(f"balance ${pods['clientBalance']:.2f}; running pods: {live}", flush=True)
        if os.path.exists(STATE):
            raise SystemExit(f'{STATE} exists; `pod.py down` first')
        pubkey = open(os.path.expanduser('~/.ssh/id_ed25519.pub')).read().strip()
        pod = create(key, 'autoresearch', pubkey, 'SECURE')
        if pod is None:
            print('no secure-cloud capacity, trying community', flush=True)
            pod = create(key, 'autoresearch', pubkey, 'COMMUNITY')
        if pod is None:
            raise SystemExit(f'no {GPU} capacity right now')
        st = {'id': pod['id'], 'cost': pod['costPerHr'], 't0': time.time()}
        json.dump(st, open(STATE, 'w'))   # saved before waiting, so `down` works if the wait is interrupted
        print(f"pod {pod['id']} at ${pod['costPerHr']}/hr", flush=True)
        st['ip'], st['port'] = wait_ssh(key, pod['id'])
        json.dump(st, open(STATE, 'w'))
        for _ in range(60):
            if subprocess.run(['ssh', *ssh_args(st), 'true'], capture_output=True).returncode == 0:
                break
            time.sleep(5)
        else:
            raise SystemExit('ssh never accepted the key')
        upload(st, ['remote.sh', 'harness.py', 'pretrain.py'])
        ssh_run(st, f'cd {REMOTE} && bash remote.sh /tmp/warm --help > /dev/null 2>&1; nvidia-smi --query-gpu=name --format=csv,noheader')
        print(f'up after {time.time() - st["t0"]:.0f}s: ssh -p {st["port"]} root@{st["ip"]}')
    elif cmd == 'run':
        st = load_state()
        name, args = rest[0], rest[1:]
        upload(st, ['remote.sh', 'harness.py', 'pretrain.py'])
        out = f'{REMOTE}/artifacts/{name}'
        rc = ssh_run(st, f'mkdir -p {REMOTE}/artifacts && bash {REMOTE}/remote.sh {out} ' + ' '.join(map(shlex.quote, args)), check=False)
        ssh = 'ssh ' + ' '.join(ssh_args(st))
        subprocess.run(f'{ssh} "cd {REMOTE}/artifacts && tar czf - {name} {name}.log" | tar xzf - -C {shlex.quote(LOCAL_ART)}',
                       shell=True, check=False)
        print(f'artifacts in {os.path.abspath(os.path.join(LOCAL_ART, name))}; remote rc {rc}', flush=True)
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main()
