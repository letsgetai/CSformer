"""Run committed real-data configurations and persist an auditable result ledger."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def git(root, *arguments):
    return subprocess.check_output(['git', '-C', str(root), *arguments], text=True).strip()


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', default=['ETTh1'])
    parser.add_argument('--horizons', nargs='+', type=int, default=[96])
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if output.is_relative_to(root):
        parser.error('experiment output must be outside the immutable code checkout')
    if git(root, 'status', '--porcelain'):
        parser.error('commit all source changes before running experiments')
    branch = git(root, 'branch', '--show-current') or 'detached'
    import torch
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable; verify driver and PyTorch wheel compatibility')
    rows = json.loads((root / 'metadata/paper-configs.json').read_text())['configurations']
    selected = [r for r in rows if r['dataset'] in args.datasets and r['pred_len'] in args.horizons]
    if len(selected) != len(set(args.datasets)) * len(set(args.horizons)):
        parser.error('selection contains unknown datasets/horizons')
    revision = git(root, 'rev-parse', 'HEAD')
    versions = {d.metadata['Name']: d.version for d in importlib.metadata.distributions() if d.metadata['Name']}
    data_records = {}
    for row in selected:
        name = row['dataset']
        relative = Path('ETT') / (name + '.csv') if name.startswith('ETT') else Path(name.lower()) / (name.lower() + '.csv')
        path = args.data_root.resolve() / relative
        if not path.is_file():
            parser.error(f'missing dataset {relative}')
        if name not in data_records:
            data_records[name] = {'path': str(path), 'sha256': sha256(path), 'bytes': path.stat().st_size}
    output.mkdir(parents=True, exist_ok=False)
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,driver_version', '--format=csv,noheader'], capture_output=True, text=True) if args.device == 'cuda' else None
    metadata = {'code_revision': revision, 'branch': branch, 'python': sys.version,
                'torch': torch.__version__, 'cuda_build': torch.version.cuda, 'device': args.device,
                'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                'driver_gpu': gpu.stdout.strip() if gpu else None, 'packages': versions, 'datasets': data_records,
                'protocol': 'Legacy validation loader (shuffle=True, drop_last=True); no AMP, compile or architecture changes.'}
    save(output / 'environment.json', metadata)
    results = []
    save(output / 'results.json', results)
    for row in selected:
        if git(root, 'rev-parse', 'HEAD') != revision or git(root, 'status', '--porcelain'):
            raise SystemExit('Source changed; experiment stopped before next configuration')
        current = {d.metadata['Name']: d.version for d in importlib.metadata.distributions() if d.metadata['Name']}
        if current != versions:
            raise SystemExit('Environment changed; experiment stopped')
        name, horizon = row['dataset'], row['pred_len']
        data = data_records[name]
        if sha256(Path(data['path'])) != data['sha256']:
            raise SystemExit('Dataset changed; experiment stopped')
        run = output / f'{name}-{horizon}'
        run.mkdir()
        command = [sys.executable, '-u', str(root / 'run.py'), '--model', 'CSformer',
                   '--model_id', f'{name}_336_{horizon}', '--data', name if name.startswith('ETT') else 'custom',
                   '--root_path', str(Path(data['path']).parent), '--data_path', Path(data['path']).name,
                   '--seq_len', '336', '--pred_len', str(horizon), '--label_len', '48',
                   '--top_k', str(row['paper']['scale']), '--period_list', *map(str, row['paper']['patch_lengths']),
                   '--e_layers', '1', '--n_heads', '1', '--n_f_layers', '1', '--num_workers', '0',
                   '--val_protocol', 'legacy',
                   '--use_gpu', '1' if args.device == 'cuda' else '0', '--gpu', '0']
        for key, value in row['archive_only'].items():
            if value is not None:
                command += ['--' + key, str(value)]
        record = {'dataset': name, 'horizon': horizon, 'status': 'running', 'code_revision': revision,
                  'data_sha256': data['sha256'], 'command': command, 'start_unix': time.time()}
        results.append(record)
        save(output / 'results.json', results)
        environment = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
                           PYTHONHASHSEED=str(row['archive_only']['random_seed']), PYTHONDONTWRITEBYTECODE='1')
        with (run / 'train.log').open('w') as log:
            process = subprocess.Popen(command, cwd=run, env=environment, stdout=log, stderr=subprocess.STDOUT)
            try:
                while process.poll() is None:
                    time.sleep(5)
                    if git(root, 'rev-parse', 'HEAD') != revision or git(root, 'status', '--porcelain'):
                        process.terminate()
                        process.wait(timeout=30)
                        raise RuntimeError('Source changed during experiment; terminated')
                record['returncode'] = process.returncode
            except BaseException:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=30)
                record['status'] = 'interrupted'
                save(output / 'results.json', results)
                raise
        text = (run / 'train.log').read_text(errors='replace')
        tail = text[text.rfind('>>>>>>>testing'):]
        matches = re.findall(r'mse:([\d.eE+-]+), mae:([\d.eE+-]+)', tail)
        record['elapsed_seconds'] = time.time() - record['start_unix']
        if process.returncode == 0 and matches:
            record.update(status='completed', mse=float(matches[-1][0]), mae=float(matches[-1][1]))
        else:
            record['status'] = 'failed'
        targets = json.loads((root / 'metadata/paper-targets.json').read_text())['results']
        target = next(t for t in targets if t['dataset'] == name and t['horizon'] == horizon)
        record['paper'] = target
        if record['status'] == 'completed':
            record['mse_difference'] = record['mse'] - target['mse']
            record['mae_difference'] = record['mae'] - target['mae']
        save(output / 'results.json', results)
        print(json.dumps({k: v for k, v in record.items() if k != 'command'}), flush=True)
        if record['status'] != 'completed':
            raise SystemExit('Experiment failed; inspect train.log before continuing')


if __name__ == '__main__':
    main()
