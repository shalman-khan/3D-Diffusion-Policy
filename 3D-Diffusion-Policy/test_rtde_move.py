#!/usr/bin/env python3
"""
Quick test: move robot2 wrist3 by +0.1 rad using servoJ, then back.
Run this to verify RTDE servoJ actually moves the physical robot.

Usage:
    python3 test_rtde_move.py --ip 192.168.1.20
"""
import time, argparse
import numpy as np
from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive

parser = argparse.ArgumentParser()
parser.add_argument("--ip",    default="192.168.1.20")
parser.add_argument("--joint", type=int,   default=5,   help="Joint index 0-5 to move")
parser.add_argument("--delta", type=float, default=0.1, help="How far to move (rad)")
args = parser.parse_args()

print(f"Connecting to {args.ip} ...")
rc = RTDEControl(args.ip)
rr = RTDEReceive(args.ip)

q_start = np.array(rr.getActualQ())
print(f"Current joints: {np.round(q_start, 4)}")
print(f"Moving joint {args.joint} by +{args.delta} rad ...")

q_target = q_start.copy()
q_target[args.joint] += args.delta

# Interpolate over 50 RTDE steps (~400ms at 125Hz)
n_steps = 50
for i in range(n_steps):
    alpha = (i + 1) / n_steps
    q_interp = q_start + alpha * (q_target - q_start)
    rc.servoJ(q_interp.tolist(), 0.5, 0.5, 0.008, 0.1, 300)
    time.sleep(0.008)

q_now = np.array(rr.getActualQ())
print(f"After move:     {np.round(q_now, 4)}")
print(f"Actual delta j{args.joint}: {q_now[args.joint] - q_start[args.joint]:.4f} rad")

moved = abs(q_now[args.joint] - q_start[args.joint]) > 0.01
print(f"{'✓ Robot moved — servoJ is working' if moved else '✗ Robot did NOT move — check Remote Control mode and UR program'}")

if moved:
    print("\nReturning to start ...")
    for i in range(n_steps):
        alpha = (i + 1) / n_steps
        q_interp = q_now + alpha * (q_start - q_now)
        rc.servoJ(q_interp.tolist(), 0.5, 0.5, 0.008, 0.1, 300)
        time.sleep(0.008)
    rc.servoStop()

rc.disconnect(); rr.disconnect()
print("Done.")
