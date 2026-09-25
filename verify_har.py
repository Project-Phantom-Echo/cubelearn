#!/usr/bin/env python3
"""Verify retained HAR records; optionally run a CPU model/DFT smoke check, without training."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def verify_records():
    campaign = ROOT / 'results/diagnostics_20260909'
    manifest = json.loads((campaign / 'manifest.json').read_text())
    for name, expected in manifest['source_sha256'].items():
        if hashlib.sha256((campaign / 'source' / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Historical source changed: {name}')
    records = sorted((ROOT / 'results/dat_2dcnn_lstm').glob('*/*/results.json'))
    if len(records) != 10:
        raise ValueError(f'Expected ten retained results, found {len(records)}')
    for path in records:
        r = json.loads(path.read_text())
        history = r['history']
        if [h['epoch'] for h in history] != list(range(r['epochs'])):
            raise ValueError(f'Incomplete history: {path}')
        best = max(history, key=lambda h: (h['val_acc'], -h['val_loss'], -h['epoch']))
        if best['epoch'] + 1 != r['best_epoch'] or best['val_acc'] != r['val_acc']:
            raise ValueError(f'Checkpoint selection mismatch: {path}')
        for split in ('test_in', 'test_out'):
            confusion = r[split]['confusion']
            samples = sum(map(sum, confusion))
            accuracy = sum(confusion[i][i] for i in range(6)) / samples
            if samples != 360 or abs(accuracy - r[split]['acc']) > 1e-12:
                raise ValueError(f'Test metric mismatch: {path}/{split}')
    print('Verified all ten results, validation checkpoint selection, confusion matrices and historical source hashes.')


def smoke():
    import torch
    from har_dataset import HARDataset
    from network import Range_Fourier_Net_Small, Doppler_Fourier_Net_Small, AOA_Fourier_Net
    from network_har import DAT_2DCNNLSTM_HAR
    datasets = [HARDataset('', str(ROOT / 'dataset_split'), split, crop=True)
                for split in ('train', 'val', 'test_in', 'test_out')]
    if [len(ds) for ds in datasets] != [540, 180, 360, 360]:
        raise ValueError('Split counts differ from the retained experiment')
    for i, a in enumerate(datasets):
        for b in datasets[i + 1:]:
            if set(a.items) & set(b.items):
                raise ValueError('Overlapping split identities')
    torch.manual_seed(0)
    for constructor, inputs, outputs, shift in (
        (Range_Fourier_Net_Small, 128, 128, False),
        (Doppler_Fourier_Net_Small, 64, 64, True),
        (AOA_Fourier_Net, 8, 64, True),
    ):
        values = torch.randn(2, inputs, dtype=torch.complex64)
        with torch.no_grad():
            transformed = constructor()(values)
        actual = torch.complex(transformed.real, transformed.imag)
        expected = torch.fft.fft(values, n=outputs)
        if shift:
            expected = torch.fft.fftshift(expected, dim=-1)
        torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)
    with torch.no_grad():
        output = DAT_2DCNNLSTM_HAR().eval()(torch.zeros(1, 20, 64, 8, 128, dtype=torch.complex64))
    if output.shape != (1, 6) or not torch.isfinite(output).all():
        raise ValueError('Invalid HAR model output')
    print('CPU smoke passed: disjoint splits, independent FFT equivalence, and 20-frame/6-class model output.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    verify_records()
    if args.smoke:
        smoke()
