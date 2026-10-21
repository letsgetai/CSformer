"""Run a temporary synthetic train/validation/test cycle; this is not a benchmark."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('smoke-output'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    revision = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain', '--untracked-files=normal'], text=True)
    if dirty:
        raise SystemExit('Commit source changes before running the smoke check.')
    with tempfile.TemporaryDirectory(prefix='csformer-smoke-') as tmp:
        tmp = Path(tmp)
        t = np.arange(400)
        frame = pd.DataFrame({'date': pd.date_range('2020-01-01', periods=len(t), freq='h'),
                              'value': np.sin(t / 7) + t / 200,
                              'OT': np.cos(t / 11) + t / 300})
        frame.to_csv(tmp / 'synthetic.csv', index=False)
        command = [sys.executable, str(root / 'run.py'), '--model', 'CSformer', '--data', 'custom',
                   '--root_path', str(tmp), '--data_path', 'synthetic.csv', '--model_id', 'synthetic_smoke',
                   '--seq_len', '32', '--label_len', '16', '--pred_len', '8', '--enc_in', '2', '--c_out', '2',
                   '--d_model', '8', '--d_ff', '16', '--n_heads', '1', '--e_layers', '1', '--n_f_layers', '1',
                   '--top_k', '3', '--batch_size', '16', '--train_epochs', '2', '--patience', '2',
                   '--num_workers', '0', '--use_gpu', '0', '--random_seed', '2024',
                   '--checkpoints', str(tmp / 'checkpoints')]
        environment = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', MPLBACKEND='Agg')
        result = subprocess.run(command, cwd=tmp, env=environment, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
        (args.output / 'smoke.log').write_text(result.stdout)
        if result.returncode:
            print(result.stdout[-10000:])
            raise SystemExit(result.returncode)
        files = list((tmp / 'results').glob('*/pred.npy'))
        assert len(files) == 1, files
        pred = np.load(files[0])
        assert pred.shape == (73, 8, 2), pred.shape
        assert np.isfinite(pred).all()
        checkpoints = list((tmp / 'checkpoints').glob('*/checkpoint.pth'))
        assert len(checkpoints) == 1
        # Exercise separate evaluation and checkpoint loading using the same config.
        evaluation = subprocess.run(command + ['--is_training', '0'], cwd=tmp, env=environment, text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
        (args.output / 'smoke-evaluation.log').write_text(evaluation.stdout)
        assert evaluation.returncode == 0, evaluation.stdout[-5000:]
        np.testing.assert_array_equal(pred, np.load(files[0]))
        summary = {'status': 'passed', 'git_revision': revision, 'python': sys.version.split()[0],
                   'epochs': 2, 'synthetic_rows': 400, 'test_samples': 73,
                   'prediction_shape': list(pred.shape), 'checkpoint_reload': 'exact_predictions_match',
                   'paper_accuracy_reproduced': False}
        (args.output / 'smoke-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
