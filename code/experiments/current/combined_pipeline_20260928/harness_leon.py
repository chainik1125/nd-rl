#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Uploaded to the pod as harness.py for the Leon-RL arm: the frozen harness (uploaded as
harness_frozen.py, byte-identical) with only stage_rl replaced by leon_rl.stage_rl."""

import harness_frozen as H
import leon_rl

H.stage_rl = leon_rl.stage_rl

if __name__ == "__main__":
    H.main()
