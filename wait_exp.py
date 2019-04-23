import os
import time
import subprocess
from pathlib import Path

def wait_exp(exp_dir):
    while True:
        for fn in os.listdir(exp_dir):
            if not fn.endswith('.run'): continue
            return Path(exp_dir, fn)
        time.sleep(1)


if __name__ == "__main__":
    while True:
        exp_path = wait_exp('exps/running')
        subprocess.call(['python', 'train.py', 'join', exp_path])
