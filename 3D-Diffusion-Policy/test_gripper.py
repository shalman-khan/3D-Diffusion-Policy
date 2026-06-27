#!/usr/bin/env python3
"""
test_gripper.py
===============
Standalone test — sends rq_set_pos commands via sendCustomScript (port 30002).
Run this BEFORE policy_executor to confirm gripper control works independently.

Usage:
    python3 test_gripper.py --ip 192.168.1.20
    python3 test_gripper.py --ip 192.168.1.20 --pos 128   # half-open
"""
import argparse
import time
from rtde_control import RTDEControlInterface as RTDEControl

def test_gripper(ip: str, pos: int, delay: float = 2.0):
    print(f"Connecting to {ip} ...")
    rc = RTDEControl(ip)
    print("Connected.")

    for pos_byte in [0, pos, 255, 0]:
        script = f"def grip():\n  rq_set_pos({pos_byte})\nend\ngrip()\n"
        print(f"  Sending rq_set_pos({pos_byte}) ...", end=" ", flush=True)
        ok = rc.sendCustomScript(script)
        print(f"returned {ok}")
        time.sleep(delay)

    rc.disconnect()
    print("Done. If gripper moved: rq_set_pos works.")
    print("If gripper did not move: Robotiq URCap is not active on this controller.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip",    default="192.168.1.20")
    parser.add_argument("--pos",   type=int, default=200, help="Mid test position (0=open, 255=closed)")
    parser.add_argument("--delay", type=float, default=2.0, help="Seconds between commands")
    args = parser.parse_args()
    test_gripper(args.ip, args.pos, args.delay)
