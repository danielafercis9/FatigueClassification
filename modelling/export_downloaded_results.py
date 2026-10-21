"""Validate downloaded real fold results, export public metrics, rebuild summaries.

Import private downloads once:
    python -m modelling.export_downloaded_results --input-dir ../results
Rebuild report summaries from the committed public results.csv:
    python -m modelling.export_downloaded_results

No model training, raw recordings, window predictions or identity mapping used.
"""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MODELS = ('majority', 'lr_rms_mdf', 'lr_selected', 'svm_selected', 'svm_rbf_selected', 'cnn_fixed',
          'svm_rbf_fixed_significant', 'ann_fixed_significant', 'lfc_hfc_fixed_rbf')
METRICS = ('accuracy', 'precision_macro', 'recall_macro', 'f1_macro',
           'f1_class_0', 'f1_class_1', 'cohen_kappa', 'roc_auc', 'pr_auc',
           'average_precision')


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, columns):
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def validate(rows):
    if len(rows) != 15 * len(MODELS) or {r['model'] for r in rows} != set(MODELS):
        raise ValueError(f'Expected {len(MODELS)} completed models and {15 * len(MODELS)} fold rows.')
    for model in MODELS:
        group = [r for r in rows if r['model'] == model]
        if len(group) != 15 or sorted(int(r['fold']) for r in group) != list(range(1, 16)):
            raise ValueError(f'{model}: expected unique folds 1–15.')
        for r in group:
            for metric in METRICS:
                value = float(r[metric])
                minimum = -1 if metric == 'cohen_kappa' else 0
                if not math.isfinite(value) or not minimum <= value <= 1:
                    raise ValueError(f'{model}, fold {r["fold"]}: invalid {metric}.')
            if not math.isclose(float(r['f1_macro']),
                                (float(r['f1_class_0']) + float(r['f1_class_1'])) / 2,
                                abs_tol=1e-10):
                raise ValueError('Macro F1 does not match per-class F1.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path)
    args = parser.parse_args()
    target = ROOT / 'results.csv'
    if args.input_dir:
        files = sorted(args.input_dir.glob('*.csv'))
        rows = []
        fingerprints = []
        archive_checks = []
        reference = None
        for path in files:
            group = read_csv(path)
            if not group or len({r['model'] for r in group}) != 1:
                raise ValueError(f'{path.name}: expected one nonempty model table.')
            archive_path = args.input_dir / 'svm_rbf_selected.zip'
            if group[0]['model'] == 'svm_rbf_selected' and archive_path.is_file():
                with zipfile.ZipFile(archive_path) as archive:
                    manifests = [n for n in archive.namelist() if n.endswith('/run_manifest.json') or n == 'run_manifest.json']
                    result_files = [n for n in archive.namelist() if n.endswith('/results.csv') or n == 'results.csv']
                    if len(manifests) != 1 or len(result_files) != 1:
                        raise ValueError('Expected one selected-RBF run in the private archive.')
                    manifest = json.loads(archive.read(manifests[0]))
                    archived_rows = list(csv.DictReader(io.StringIO(archive.read(result_files[0]).decode('utf-8-sig'))))
                    if manifest['status'] != 'complete' or manifest['model_key'] != 'svm_rbf_selected' or manifest['n_outer_folds'] != 15 or archived_rows != group:
                        raise ValueError('Selected RBF archive does not match the complete downloaded fold table.')
                    archive_checks.append({'file': archive_path.name,
                        'sha256': hashlib.sha256(archive_path.read_bytes()).hexdigest(),
                        'model': manifest['model_key'], 'status': manifest['status'],
                        'n_outer_folds': manifest['n_outer_folds'],
                        'data_sha256': manifest['data_manifest']['data_sha256'],
                        'run_manifest_sha256': hashlib.sha256(archive.read(manifests[0])).hexdigest()})
            signature = {r['test_subject']: (int(r['fold']), int(r['n_test_windows']),
                         int(r['support_0']), int(r['support_1'])) for r in group}
            if len(signature) != 15 or sum(v[1] for v in signature.values()) != 9215:
                raise ValueError(f'{path.name}: unexpected participant/window counts.')
            if any(int(r['n_train_subjects']) != 14 or
                   int(r['n_windows']) != int(r['n_test_windows']) or
                   int(r['support_0']) + int(r['support_1']) != int(r['n_test_windows'])
                   for r in group):
                raise ValueError(f'{path.name}: inconsistent fold supports.')
            if reference is not None and signature != reference:
                raise ValueError('Models have different held-out subjects or supports.')
            reference = signature
            rows.extend(group)
            fingerprints.append({'file': path.name,
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        validate(rows)
        columns = ['model', 'fold', *METRICS]
        rows = [{k: r[k] for k in columns} for r in rows]
        rows.sort(key=lambda r: (MODELS.index(r['model']), int(r['fold'])))
        write_csv(target, rows, columns)
        (ROOT / 'results_provenance.json').write_text(json.dumps({
            'sources': fingerprints, 'n_models': len(MODELS), 'n_folds_per_model': 15,
            'n_rows': 15 * len(MODELS), 'n_windows_per_model': 9215,
            'verified_private_archives': archive_checks,
            'checks': 'Unique folds, matching held-out participant assignments and class supports across models; metric ranges and macro/per-class F1 identity.',
            'limitation': 'Downloaded CSVs do not establish identical original data hashes or source versions; retain original private run manifests for that verification.',
            'public_columns': columns,
        }, indent=2) + '\n', encoding='utf-8')
    rows = read_csv(target)
    validate(rows)
    summaries = []
    for model in MODELS:
        group = [r for r in rows if r['model'] == model]
        for metric in METRICS:
            values = [float(r[metric]) for r in group]
            summaries.append({'model': model, 'metric': metric,
                              'mean': statistics.mean(values),
                              'sd': statistics.stdev(values), 'n_folds': len(values)})
    write_csv(ROOT / 'results_summary.csv', summaries,
              ['model', 'metric', 'mean', 'sd', 'n_folds'])
    print(f'Validated {len(rows)} rows across {len(MODELS)} completed models. Rebuilt results_summary.csv (sample SD).')
    for model in MODELS:
        r = next(r for r in summaries if r['model'] == model and r['metric'] == 'f1_macro')
        print(f'{model}: macro-F1 {r["mean"]:.3f} +/- {r["sd"]:.3f}')


if __name__ == '__main__':
    main()
