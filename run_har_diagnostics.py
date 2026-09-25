"""Run six prespecified HAR checks and refresh their Markdown report."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
BATCH = ROOT / 'results' / 'diagnostics_20260909'
CONFIGS = [(1, 32), (2, 32), (0, 8)]


def render():
    rows = []
    for path in sorted((ROOT / 'results/dat_2dcnn_lstm').glob('*/*/results.json')):
        r = json.loads(path.read_text())
        arm = 'DFT' if r['lpp_lr'] == 0 else 'CubeLearn'
        rows.append((r, arm, path))
    lines = ['# HAR reproduction results', '',
             'Accuracies are percentages. All runs use D-A-T 2D CNN-LSTM and the same user/repetition split.', '',
             '| Run | Model | Seed | Batch | Epochs | Best epoch | Validation | Seen test | Held-out test |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r, arm, path in rows:
        lines.append(f"| [{path.parent.name}]({path.relative_to(ROOT / 'results')}) | {arm} | {r['seed']} | {r['batch_size']} | {r['epochs']} | {r['best_epoch']} | {100*r['val_acc']:.2f} | {100*r['test_in']['acc']:.2f} | {100*r['test_out']['acc']:.2f} |")
    lines += ['', '## Fixed diagnostic plan', '',
              '30 epochs for both DFT and CubeLearn at (seed, batch size): (1, 32), (2, 32), (0, 8).',
              'The first two pairs extend the original seed-0 run; batch 8 is a sensitivity experiment, not a claimed author setting.',
              'No user-split search or selection by test accuracy. Failures remain visible below.', '',
              '| Seed | Batch | Model | Status |', '|---:|---:|---|---|']
    for seed, batch in CONFIGS:
        for arm in ('dft', 'cubelearn'):
            out = ROOT / f'results/dat_2dcnn_lstm/{arm}/diag20260909_seed{seed}_bs{batch}'
            status = json.loads((out/'status.json').read_text())['state'] if (out/'status.json').exists() else 'not started'
            lines.append(f'| {seed} | {batch} | {arm} | {status} |')
    baseline = [(r,a) for r,a,_ in rows if r['epochs']==30 and r['batch_size']==32]
    lines += ['', '## Interpretation', '']
    for arm in ('DFT', 'CubeLearn'):
        subset = [r for r,a in baseline if a==arm]
        vals = [100*r['test_in']['acc'] for r in subset]
        outs = [100*r['test_out']['acc'] for r in subset]
        if vals:
            lines.append(f"- {arm}, 30 epochs/batch 32, {len(vals)} seeds: seen test {min(vals):.2f}–{max(vals):.2f}%; held-out test {min(outs):.2f}–{max(outs):.2f}%.")
    lines += ['', 'The historical 200-epoch runs are extended-budget experiments. Numerical agreement at a changed budget does not establish a faithful reproduction.',
              'See [protocol and limitations](../notes.md). Statuses and results are refreshed after every run.', '']
    dest = ROOT / 'results/har_reproduction.md'
    temp = dest.with_suffix('.tmp')
    temp.write_text('\n'.join(lines))
    temp.replace(dest)


def main():
    if '--report-only' in sys.argv:
        render()
        return
    if BATCH.exists():
        raise SystemExit('Historical campaign already exists; use --report-only to inspect it. '
                         'For a new experiment, use train_har.py with a new output directory.')
    BATCH.mkdir(parents=True, exist_ok=False)
    source = BATCH / 'source'
    source.mkdir(exist_ok=True)
    files = ['train_har.py', 'har_dataset.py', 'network_har.py', 'network.py']
    for name in files:
        shutil.copy2(ROOT/name, source/name)
    shutil.copytree(ROOT/'dataset_split', source/'dataset_split', dirs_exist_ok=True)
    manifest = {'job_id': os.environ.get('SLURM_JOB_ID'), 'python': sys.executable,
                'configs': CONFIGS, 'source_sha256': {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*') if p.is_file()}}
    (BATCH/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    failures = 0
    render()
    for seed, batch in CONFIGS:
        for arm, lpp in [('dft', 0), ('cubelearn', 1e-4)]:
            out = ROOT / f'results/dat_2dcnn_lstm/{arm}/diag20260909_seed{seed}_bs{batch}'
            out.mkdir(parents=True, exist_ok=True)
            if (out/'results.json').exists():
                continue
            cmd = [sys.executable, '-u', str(source/'train_har.py'), '--lpp-lr', str(lpp),
                   '--epochs', '30', '--batch-size', str(batch), '--seed', str(seed),
                   '--workers', '8', '--split-dir', str(source/'dataset_split'), '--output-dir', str(out)]
            (out/'command.json').write_text(json.dumps(cmd, indent=2)+'\n')
            (out/'status.json').write_text(json.dumps({'state':'running'})+'\n')
            render()
            with (out/'train.log').open('w') as log:
                code = subprocess.run(cmd, cwd=source, stdout=log, stderr=subprocess.STDOUT).returncode
            (out/'status.json').write_text(json.dumps({'state':'complete' if code==0 else f'failed (exit {code})'})+'\n')
            failures += code != 0
            render()
    raise SystemExit(bool(failures))


if __name__ == '__main__':
    main()
