"""Run the frozen full-sample classifier-only learning-rate / epoch grid."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

REPO = Path('/mnt/weka/fgeikyan/rf-perception-papers/cubelearn')
DEFAULT = Path(__file__).resolve().parent
PYTHON = REPO.parent / 'compass/.venv/bin/python'
CACHE = Path('/mnt/weka/fgeikyan/cubelearn_data/har_cache')
FILES = ['train_har.py', 'har_dataset.py', 'network_har.py', 'network.py']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(root, task):
    return [str(PYTHON), '-u', str(root/'source/train_har.py'),
            '--model', 'dat_2dcnn_lstm', '--lr', str(task['lr']),
            '--lpp-lr', str(task['lpp_lr']), '--epochs', str(task['epochs']),
            '--batch-size', str(task['batch_size']), '--seed', str(task['seed']), '--workers', '4',
            '--cache-dir', str(CACHE), '--split-dir', str(root/'source/dataset_split'),
            '--output-dir', str(root/'runs'/task['name'])]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--campaign', type=Path, default=DEFAULT)
    p.add_argument('--submit', action='store_true')
    p.add_argument('--task', type=int)
    args = p.parse_args()
    root = args.campaign.resolve()
    os.umask(0)
    if args.task is not None:
        manifest = json.loads((root/'manifest.json').read_text())
        for name, expected in manifest['sha256'].items():
            if sha(root/name) != expected:
                raise RuntimeError('Frozen source changed: '+name)
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('GPU allocation required')
        task = manifest['tasks'][args.task]
        out = root/'runs'/task['name']
        out.mkdir()  # Refuse overwriting/restarting any existing run.
        cmd = command(root, task)
        (out/'command.json').write_text(json.dumps(cmd, indent=2)+'\n')
        (out/'environment.txt').write_text(f'python={sys.version}\ntorch={torch.__version__}\ncuda={torch.version.cuda}\n')
        with (out/'train.log').open('x') as stream:
            result = subprocess.run(cmd, cwd=root/'source', stdout=stream, stderr=subprocess.STDOUT)
        (out/'status.json').write_text(json.dumps({'exit_code':result.returncode})+'\n')
        raise SystemExit(result.returncode)
    if not root.exists():
        raise RuntimeError('Prepared campaign directory missing')
    manifest = json.loads((root/'manifest.json').read_text())
    for name, expected in manifest['sha256'].items():
        if sha(root/name) != expected:
            raise RuntimeError('Frozen source changed: '+name)
    print(f"Prepared {len(manifest['tasks'])} jobs: {root}")
    if args.submit:
        (root/'submission.lock').mkdir()
        cmd = ['sbatch', '--parsable', '--partition=research', '--job-name=cubelearn-author',
               '--gres=gpu:h100:1', '--cpus-per-task=4', '--mem=32G', '--time=01:00:00',
               '--array=0-9', '--output='+str(root/'slurm/%A_%a.out'),
               '--error='+str(root/'slurm/%A_%a.err'), str(root/'job.sh'),str(root)]
        job = subprocess.check_output(cmd,text=True).strip()
        (root/'submission.json').write_text(json.dumps(dict(job_id=job,command=cmd),indent=2)+'\n')
        print('Submitted '+job)


if __name__ == '__main__':
    main()
