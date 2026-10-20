"""Export all six completed LOSO runs as the assignment's model-by-fold results.csv."""
import json
from pathlib import Path

import pandas as pd
from modelling.core import summary_table

ROOT = Path(__file__).resolve().parents[1]
MODELS = ('majority', 'lr_rms_mdf', 'lr_selected', 'svm_selected',
          'svm_rbf_selected', 'cnn_fixed')
METRICS = ('accuracy', 'precision_macro', 'recall_macro', 'f1_macro',
           'f1_class_0', 'f1_class_1', 'cohen_kappa', 'roc_auc', 'pr_auc',
           'average_precision')


def export():
    output = ROOT / 'modelling/outputs'
    tables, summaries = [], []
    reference_hash = reference_subjects = reference_code = None
    for model in MODELS:
        pointer = json.loads((output / f'latest_{model}.json').read_text())
        run = Path(pointer['path'])
        manifest = json.loads((run / 'run_manifest.json').read_text())
        if manifest['status'] != 'complete' or manifest['model_key'] != model:
            raise ValueError(f'{model}: a matching completed run is required.')
        if model == 'cnn_fixed':
            data_hash = manifest['input_manifest']['source_model_data_sha256']
            if manifest['code_sha256']['core.py'] != reference_code['core.py']:
                raise ValueError('CNN and feature runs used different shared evaluation code.')
        else:
            data_hash = manifest['data_manifest']['data_sha256']
            if reference_code is None:
                reference_code = manifest['code_sha256']
            if manifest['code_sha256'] != reference_code:
                raise ValueError('Feature runs used different source code; rerun consistently.')
        table = pd.read_csv(run / 'results.csv')
        subjects = set(table.test_subject)
        if (table.test_subject.duplicated().any() or table.fold.duplicated().any()
                or len(table) != manifest['n_outer_folds']
                or not table.model.eq(model).all()):
            raise ValueError(f'{model}: incomplete or inconsistent LOSO fold rows.')
        if reference_hash is None:
            reference_hash, reference_subjects = data_hash, subjects
        if data_hash != reference_hash or subjects != reference_subjects:
            raise ValueError('Runs use different datasets or held-out participants.')
        # Public export has fold metrics only: no identities, windows, paths or models.
        tables.append(table[['model', 'fold', *METRICS]])
        summaries.append(summary_table(table).assign(model=model))
    combined = pd.concat(tables, ignore_index=True)
    target = ROOT / 'results.csv'
    combined.to_csv(target, index=False)
    summary_path = output / 'report_metrics_mean_sd.csv'
    pd.concat(summaries, ignore_index=True).to_csv(summary_path, index=False)
    print(f'Exported {len(combined)} model-by-fold rows to {target.name}.')
    print(f'Mean and sample SD: {summary_path}')
    return target


if __name__ == '__main__':
    export()
