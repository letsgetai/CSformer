"""Behavior checks for paper structure, evaluation sample retention and public CLI."""
import contextlib
import io
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data_provider.data_factory import data_provider
from exp.exp_main import Exp_Main
from models.CSformer import Model
from models.FTST import Model as ArchivedModel
from utils.tools import EarlyStopping


def config(row):
    paper, archive = row['paper'], row['archive_only']
    return SimpleNamespace(**dict(archive, **{
        'seq_len': paper['seq_len'], 'pred_len': row['pred_len'], 'top_k': paper['scale'],
        'period_list': paper['patch_lengths'], 'n_f_layers': 1, 'e_layers': 1, 'n_heads': 1,
        'individual': 0, 'patch_len': 16, 'stride': 8, 'padding_patch': 'end',
        'revin': 1, 'affine': 0, 'subtract_last': 0, 'decomposition': 0, 'kernel_size': 25}))


class Echo(torch.nn.Module):
    def forward(self, x):
        return x[:, -4:, :]


class ProjectChecks(unittest.TestCase):
    def test_28_paper_configurations(self):
        torch.set_num_threads(2)
        rows = json.loads((ROOT / 'metadata/paper-configs.json').read_text())['configurations']
        self.assertEqual(len(rows), 28)
        self.assertEqual(len({(r['dataset'], r['pred_len']) for r in rows}), 28)
        for row in rows:
            with self.subTest(dataset=row['dataset'], horizon=row['pred_len']):
                cfg = config(row)
                with contextlib.redirect_stdout(io.StringIO()):
                    model = Model(cfg).eval()
                self.assertEqual(model.model.backbone.period_numbers, row['paper']['tokens'])
                with torch.inference_mode():
                    prediction = model(torch.randn(2, 336, cfg.enc_in))
                self.assertEqual(tuple(prediction.shape), (2, row['pred_len'], cfg.enc_in))
                self.assertTrue(torch.isfinite(prediction).all())
        self.assertIs(Model, ArchivedModel)

    def test_evaluation_retains_all_samples(self):
        for samples, batch_size in [(3, 8), (10, 4), (8, 4)]:
            with self.subTest(samples=samples, batch_size=batch_size):
                x = torch.arange(samples * 8 * 2, dtype=torch.float32).reshape(samples, 8, 2) / 50 + 1
                y = x[:, -4:, :] + .5
                loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
                    x, y, torch.zeros(samples, 8, 1), torch.zeros(samples, 4, 1)), batch_size=batch_size)
                exp = object.__new__(Exp_Main)
                exp.args = SimpleNamespace(pred_len=4, label_len=2, use_amp=False, model='CSformer',
                                           features='M', output_attention=False, test_flop=False)
                exp.device = torch.device('cpu')
                exp.model = Echo()
                exp._get_data = lambda flag: (None, loader)
                mse, mae = exp.vali(None, loader, None)
                self.assertAlmostEqual(float(mse), .25)
                self.assertAlmostEqual(float(mae), .5)
                with tempfile.TemporaryDirectory() as tmp:
                    old_cwd = os.getcwd()
                    try:
                        os.chdir(tmp)
                        with contextlib.redirect_stdout(io.StringIO()):
                            exp.test('retention')
                        predictions = np.load(Path(tmp) / 'results/retention/pred.npy')
                        np.testing.assert_allclose(predictions, x[:, -4:, :].numpy())
                    finally:
                        os.chdir(old_cwd)

    def test_validation_protocol_loaders(self):
        with tempfile.TemporaryDirectory() as temporary:
            tmp = Path(temporary)
            rows = 400
            frame = pd.DataFrame({
                'date': pd.date_range('2020-01-01', periods=rows, freq='h'),
                'value': np.sin(np.arange(rows) / 7),
                'OT': np.cos(np.arange(rows) / 11),
            })
            frame.to_csv(tmp / 'protocol.csv', index=False)
            common = dict(data='custom', root_path=str(tmp), data_path='protocol.csv', seq_len=8,
                          label_len=4, pred_len=4, features='M', target='OT', embed='timeF',
                          freq='h', num_workers=0, batch_size=8)

            full_args = SimpleNamespace(val_protocol='full', **common)
            full_set, full_loader = data_provider(full_args, 'val')
            self.assertIsInstance(full_loader.sampler, torch.utils.data.SequentialSampler)
            self.assertFalse(full_loader.drop_last)
            first_pass = [batch[0].shape[0] for batch in full_loader]
            self.assertEqual(sum(first_pass), len(full_set))
            self.assertGreater(len(full_set) % full_loader.batch_size, 0)
            self.assertIsInstance(full_loader.sampler, torch.utils.data.SequentialSampler)

            legacy_args = SimpleNamespace(val_protocol='legacy', **common)
            legacy_set, legacy_loader = data_provider(legacy_args, 'val')
            self.assertIsInstance(legacy_loader.sampler, torch.utils.data.RandomSampler)
            self.assertTrue(legacy_loader.drop_last)
            legacy_batches = [batch[0].shape[0] for batch in legacy_loader]
            self.assertEqual(sum(legacy_batches), len(legacy_set) // legacy_loader.batch_size * legacy_loader.batch_size)
            self.assertLess(sum(legacy_batches), len(legacy_set))

            train_set, train_loader = data_provider(full_args, 'train')
            self.assertIsInstance(train_loader.sampler, torch.utils.data.RandomSampler)
            self.assertTrue(train_loader.drop_last)
            test_set, test_loader = data_provider(full_args, 'test')
            self.assertIsInstance(test_loader.sampler, torch.utils.data.SequentialSampler)
            self.assertFalse(test_loader.drop_last)
            self.assertEqual(sum(batch[0].shape[0] for batch in test_loader), len(test_set))

            with self.assertRaises(ValueError):
                data_provider(SimpleNamespace(val_protocol='unknown', **common), 'val')

    def test_forward_dispatch_uses_model_interface(self):
        captured = []

        class TwoArgModel(nn.Module):
            def forward(self, x):
                captured.append(('single', x))
                return x

        class FourArgModel(nn.Module):
            def forward(self, x, x_mark, dec_inp, y_mark):
                captured.append(('four', x, x_mark, dec_inp, y_mark))
                return (x, 'attention')

        exp = object.__new__(Exp_Main)
        exp.args = SimpleNamespace(model='CSformer', output_attention=False)
        exp.model = TwoArgModel()
        tensors = [torch.zeros(2, 4, 3) for _ in range(4)]
        output = exp._forward(*tensors)
        self.assertEqual(captured[-1][0], 'single')
        self.assertIs(output, tensors[0])

        exp.args = SimpleNamespace(model='Autoformer', output_attention=False)
        exp.model = FourArgModel()
        exp._forward(*tensors)
        self.assertEqual(len(captured[-1]), 5)

        exp.args = SimpleNamespace(model='Autoformer', output_attention=True)
        output = exp._forward(*tensors)
        self.assertEqual(len(captured[-1]), 5)
        self.assertIs(output, tensors[0])

    def test_training_never_reads_test_split(self):
        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.projection = nn.Linear(2, 2)

            def forward(self, x):
                return self.projection(x[:, -4:, :])

        torch.manual_seed(0)
        x = torch.randn(12, 8, 2)
        y = torch.randn(12, 4, 2)
        dataset = torch.utils.data.TensorDataset(
            x, y, torch.zeros(12, 8, 1), torch.zeros(12, 4, 1))
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=4)
        val_loader = torch.utils.data.DataLoader(dataset, batch_size=4)
        requested = []

        with tempfile.TemporaryDirectory() as temporary:
            exp = object.__new__(Exp_Main)
            exp.device = torch.device('cpu')
            exp.model = TinyModel()
            exp.args = SimpleNamespace(
                train_epochs=1, patience=2, use_amp=False, pct_start=0.3, learning_rate=0.01,
                lradj='type3', checkpoints=temporary, pred_len=4, label_len=2, features='M',
                model='CSformer', output_attention=False, test_flop=False, num_workers=0,
                val_protocol='full', random_seed=2024)

            def get_data(flag):
                requested.append(flag)
                if flag == 'test':
                    raise AssertionError('training must not construct the test split')
                return None, (train_loader if flag == 'train' else val_loader)

            exp._get_data = get_data
            with contextlib.redirect_stdout(io.StringIO()):
                exp.train('protocol-regression')
            self.assertEqual(requested, ['train', 'val'])
            self.assertTrue((Path(temporary) / 'protocol-regression' / 'checkpoint.pth').is_file())

    def test_cli_rejects_non_paper_patch_structure(self):
        cases = [['--seq_len', '336', '--top_k', '3', '--period_list', '336', '168', '112'],
                 ['--seq_len', '95', '--top_k', '3'], ['--decomposition', '1'],
                 ['--d_ff', '0']]
        for args in cases:
            with self.subTest(args=args):
                process = subprocess.run([sys.executable, str(ROOT / 'run.py'), *args],
                                         capture_output=True, text=True, timeout=30)
                self.assertEqual(process.returncode, 2, process.stderr)
                self.assertIn('error:', process.stderr)
                self.assertNotIn('Traceback', process.stderr)

    def test_all_experiment_scripts_from_another_directory(self):
        rows = json.loads((ROOT / 'metadata/paper-configs.json').read_text())['configurations']
        with tempfile.TemporaryDirectory() as temporary:
            tmp = Path(temporary)
            shutil.copytree(ROOT / 'scripts', tmp / 'scripts')
            capture = tmp / 'calls.jsonl'
            stub = tmp / 'capture-python'
            stub.write_text('#!/usr/bin/env python3\nimport json, os, sys\nwith open(os.environ["CSFORMER_CALLS"], "a") as f: f.write(json.dumps(sys.argv[1:]) + "\\n")\n')
            stub.chmod(0o755)
            environment = dict(os.environ, PYTHON=str(stub), CSFORMER_CALLS=str(capture))
            for script in sorted((tmp / 'scripts/CSformer').glob('*.sh')):
                subprocess.run(['bash', str(script)], cwd='/', env=environment, check=True, timeout=30)
            calls = [json.loads(line) for line in capture.read_text().splitlines()]
            self.assertEqual(len(calls), 28)
            expected = {(r['current_script'].split('/')[-1], r['pred_len']): r for r in rows}
            matched = set()
            for call in calls:
                self.assertEqual(call[:2], ['-u', 'run.py'])
                fields = {}
                name = None
                for token in call[2:]:
                    if token.startswith('--'):
                        name = token[2:]
                        fields[name] = []
                    else:
                        fields[name].append(token)
                self.assertEqual(fields['model'], ['CSformer'])
                for key in ['e_layers', 'n_heads']:
                    self.assertEqual(fields[key], ['1'])
                self.assertEqual(fields['seq_len'], ['336'])
                self.assertIn('_336_', fields['model_id'][0])
                candidates = [r for r in rows if r['pred_len'] == int(fields['pred_len'][0])
                              and r['archive_only']['enc_in'] == int(fields['enc_in'][0])
                              and (r['dataset'].lower() in fields['data_path'][0].lower())]
                self.assertEqual(len(candidates), 1, fields)
                row = candidates[0]
                self.assertEqual(int(fields['top_k'][0]), row['paper']['scale'])
                periods = list(map(int, fields['period_list']))[:row['paper']['scale']]
                self.assertEqual(periods, row['paper']['patch_lengths'])
                matched.add((row['dataset'], row['pred_len']))
            self.assertEqual(len(matched), 28)

    def test_numpy_early_stopping(self):
        self.assertEqual(EarlyStopping().val_loss_min, np.inf)


if __name__ == '__main__':
    unittest.main()
