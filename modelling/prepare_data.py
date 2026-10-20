"""Export an allowlisted, pseudonymous feature bundle locally; never upload data."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assign_labels(w):
    """Only complete eligible windows wholly within reviewed early/late thirds."""
    a, b = w.bounds_start_s, w.bounds_end_s
    if ((b - a) <= 0).any():
        raise ValueError('Nonpositive reviewed interval.')
    early = (w.t_start_s >= a - 1e-9) & (w.t_end_exclusive_s <= a + (b-a)/3 + 1e-9)
    late = (w.t_start_s >= a + 2*(b-a)/3 - 1e-9) & (w.t_end_exclusive_s <= b + 1e-9)
    keep = w.signal_eligible & w.boundary_status.eq('reviewed') & (early | late)
    result = w.loc[keep].copy()
    result['Label'] = np.where(early[keep], 0, 1).astype(int)
    return result


def prepare():
    latest = json.loads((ROOT/'processing/outputs/latest_run.json').read_text())
    # Persisted pointers use Windows separators; support reruns on another OS.
    source = ROOT / latest['path'].replace('\\', '/')
    manifest = json.loads((source/'run_manifest.json').read_text())
    annotations = json.loads((ROOT/'processing/annotations.json').read_text())
    if manifest['annotations'] != annotations:
        raise ValueError('Boundary reviews changed: rerun preprocessing before exporting.')
    status = manifest['status']
    if status['pending_recordings'] or not status['all_original_hashes_unchanged']:
        raise ValueError('Resolve pending reviews/source integrity before exporting.')
    dictionary = pd.read_csv(source/'feature_dictionary.csv')
    features = dictionary.feature.tolist()
    if len(features) != 36 or len(set(features)) != 36:
        raise ValueError('Expected the audited 36-feature dictionary.')
    w = pd.read_csv(source/'window_features.csv')
    if w.signal_eligible.dtype != bool or not w.window_id.is_unique:
        raise ValueError('Unexpected eligibility type or repeated window IDs.')
    selected = assign_labels(w)
    # Explicit allowlist: no reviewer names, source filenames, private paths or notes.
    metadata = ['window_id','subject_id','recording_id','set_id','set_token','leg',
        'sample_start','sample_end_exclusive','t_start_s','t_end_exclusive_s',
        'fs_used_hz','n_samples','segment_id','bounds_start_s','bounds_end_s',
        'signal_eligible','review_possible_motion_or_drift','review_possible_transient',
        'flag_flatline','flag_manual_artifact','Label']
    data = selected[metadata + features].sort_values(
        ['subject_id','recording_id','sample_start']).reset_index(drop=True)
    data[features] = data[features].replace([np.inf, -np.inf], np.nan)
    if data.groupby('subject_id').Label.nunique().ne(2).any():
        raise ValueError('Every participant must have both stages for this evaluation.')
    if not data.subject_id.str.fullmatch(r'S\d{3}').all():
        raise ValueError('Subject IDs must be anonymous stable IDs.')
    out = ROOT/'modelling/data'
    out.mkdir(parents=True, exist_ok=True)
    data.to_csv(out/'model_windows.csv', index=False)
    dictionary.to_csv(out/'feature_dictionary.csv', index=False)
    counts = data.groupby(['subject_id','leg','Label']).size().rename('windows').reset_index()
    counts.to_csv(out/'class_counts.csv', index=False)
    decisions = annotations['recordings']
    excluded = sorted(k for k,v in decisions.items() if v['status']=='exclude')
    safe = dict(schema_version=1, processing_run_id=manifest['run_id'],
        source_window_table_sha256=sha(source/'window_features.csv'),
        annotation_sha256=sha(ROOT/'processing/annotations.json'),
        exporter_sha256=sha(__file__), preprocessing_config=manifest['config'],
        label_definition='0=early, 1=late: elapsed-time thirds of reviewed active set; '
            'middle and boundary-crossing windows omitted; proxy labels, not physiological ground truth',
        excluded_recording_ids=excluded, n_subjects=int(data.subject_id.nunique()),
        n_sets=int(data.recording_id.nunique()), n_windows=len(data),
        class_counts={str(k):int(v) for k,v in data.Label.value_counts().sort_index().items()},
        data_sha256=sha(out/'model_windows.csv'),
        dictionary_sha256=sha(out/'feature_dictionary.csv'),
        limitations=['No MVC reference: no MVC normalization.',
            'Failure endpoint is user testimony, not an independently recorded event.',
            'Elapsed-time thirds adapt the assignment repetition-count example.',
            'Whole-cohort exploratory analysis and manual boundary/QC review preceded modelling.',
            'The saved whole-set exclusion was a manual exploratory QC decision.',
            'Filtering uses full recording segments (offline analysis, not real-time prediction).'])
    (out/'data_manifest.json').write_text(json.dumps(safe, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:safe[k] for k in ['processing_run_id','n_subjects','n_sets','n_windows','class_counts','excluded_recording_ids']}, indent=2))
    return out


if __name__ == '__main__':
    prepare()
