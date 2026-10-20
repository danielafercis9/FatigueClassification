"""Reconstruct the existing labelled filtered windows locally; no training/upload."""
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd
from modelling.core import load_data,sha

ROOT=Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from project_paths import raw_data_dir, resolve_recording_path
sys.path.insert(0,str(ROOT/'feature_explorer'))
from explore_features import Settings,prepare


def export():
    here=ROOT/'modelling'
    w,dictionary,model_manifest=load_data(here/'data')
    run=ROOT/'processing/outputs'/model_manifest['processing_run_id']
    manifest=json.loads((run/'run_manifest.json').read_text())
    if sha(run/'window_features.csv')!=model_manifest['source_window_table_sha256']:
        raise ValueError('Processing feature table differs from modelling provenance.')
    if sha(ROOT/'processing/annotations.json')!=model_manifest['annotation_sha256']:
        raise ValueError('Boundary reviews changed since the modelling bundle was created.')
    keys={f.name for f in fields(Settings)}
    cfg=Settings(**{k:v for k,v in manifest['config'].items() if k in keys})
    records=json.loads((ROOT/'audit/private/recording_paths.json').read_text())
    paths={r['recording_id']:resolve_recording_path(r['source_relative_path']) for r in records}
    reviewed=manifest['annotations']['recordings']
    signals=np.empty((len(w),500),dtype=np.float32)
    source_hashes={};max_rms_error=0.
    for count,(rid,g) in enumerate(w.groupby('recording_id',sort=True),start=1):
        path=paths[rid];expected=reviewed[rid]['source_sha256']
        if sha(path)!=expected:raise ValueError(f'{rid}: source checksum changed.')
        try:
            obj=prepare(rid,path,cfg,compute_features=False,require_valid=False)
        except Exception as exc:
            raise RuntimeError(f'{rid}: reconstruction failed ({type(exc).__name__}); inspect locally.') from None
        generated=obj['frame'].set_index('window_id')
        for index,row in g.iterrows():
            original=generated.loc[row.window_id]
            a,b=int(row.sample_start),int(row.sample_end_exclusive)
            if not original.valid_window or a!=original.sample_start or b!=original.sample_end_exclusive:
                raise ValueError(f'{rid}: original window alignment/eligibility mismatch.')
            if b-a!=500 or not np.isclose(obj['fs'],row.fs_used_hz,rtol=1e-8):
                raise ValueError(f'{rid}: unexpected window size or sampling rate.')
            x=obj['filtered'][a:b]
            if not np.isfinite(x).all():raise ValueError(f'{rid}: nonfinite signal.')
            rms=float(np.sqrt(np.mean(x*x)))
            if not np.isclose(rms,row.RMS,rtol=1e-6,atol=1e-8):
                raise ValueError(f'{rid}: reconstructed waveform and saved RMS disagree.')
            max_rms_error=max(max_rms_error,abs(rms-row.RMS))
            signals[index]=x
        if sha(path)!=expected:raise ValueError(f'{rid}: source changed while reading.')
        source_hashes[rid]=expected
        if count%25==0:print(f'Reconstructed {count}/{w.recording_id.nunique()} anonymous recordings.',flush=True)
    out=here/'cnn_data';out.mkdir(exist_ok=True)
    metadata=w.drop(columns=dictionary.feature.tolist())
    metadata.to_csv(out/'window_metadata.csv',index=False)
    np.savez_compressed(out/'signal_windows.npz',signals=signals,
        window_id=w.window_id.to_numpy(dtype=str))
    safe=dict(schema_version=1,source_model_data_sha256=model_manifest['data_sha256'],
        processing_run_id=model_manifest['processing_run_id'],n_windows=len(w),
        n_subjects=int(w.subject_id.nunique()),n_sets=int(w.recording_id.nunique()),
        signal_shape=list(signals.shape),signal_dtype='float32',
        signal_description='Same offline 20-450 Hz filtered unrectified waveform windows used for feature extraction; no MVC normalization',
        metadata_sha256=sha(out/'window_metadata.csv'),signals_sha256=sha(out/'signal_windows.npz'),
        source_recording_sha256=source_hashes,preprocessing_config=manifest['config'],
        exporter_sha256=sha(__file__),filter_implementation_sha256=sha(ROOT/'feature_explorer/explore_features.py'),
        max_float64_reconstruction_rms_error=max_rms_error,
        source_model_manifest=model_manifest,
        privacy='Restricted pseudonymous waveform data; keep private. No source filenames, identity keys or free-text annotations included.')
    (out/'signal_manifest.json').write_text(json.dumps(safe,indent=2)+'\n',encoding='utf-8')
    archive=here/'cnn_signal_data.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name in ['signal_windows.npz','window_metadata.csv','signal_manifest.json']:
            z.write(out/name,Path('cnn_data')/name)
    print(f'Exported {len(w)} x 500 samples from {w.recording_id.nunique()} sets / {w.subject_id.nunique()} subjects.')
    print(f'Local private signal bundle: {archive.name} ({archive.stat().st_size/1024**2:.2f} MiB). No upload or training performed.')
    return out


if __name__=='__main__':export()
