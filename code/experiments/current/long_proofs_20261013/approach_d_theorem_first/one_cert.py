"""Official certify() (unmodified minlen) on one prompt: python one_cert.py 'THM ...' FLOOR SECS"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import certify as C
from minlen import parse_thm
p, floor, secs = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
prem, _ = parse_thm(p)
print(json.dumps(C.certify(p, len(prem), floor=floor, secs=secs)), flush=True)
