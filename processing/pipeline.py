"""Section 4 processing; local data only. No target labels or classifiers.

py -3.13 processing/pipeline.py init
py -3.13 processing/pipeline.py run --preview
py -3.13 processing/review_boundaries.py
py -3.13 processing/pipeline.py run
"""
from __future__ import annotations

import argparse
from dataclasses import fields
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy import signal

ROOT=Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from project_paths import raw_data_dir, resolve_recording_path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'feature_explorer'))
from explore_features import FEATURES, Settings, extract_window, prepare


def utc_now(): return datetime.now(timezone.utc).isoformat()
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')
    temporary.replace(path)


def registry():
    records=json.loads((ROOT/'audit/private/recording_paths.json').read_text(encoding='utf-8'))
    return {r['recording_id']:resolve_recording_path(r['source_relative_path']).resolve() for r in records}


def configuration(path=None):
    cfg=json.loads(Path(path or HERE/'config.json').read_text(encoding='utf-8'))
    if cfg['mvc_normalization']!='not_applied_reference_unavailable':
        raise ValueError('MVC reference is unavailable; this pipeline cannot apply MVC normalization.')
    if tuple(cfg['bandpass_hz'])!=(20.,450.):
        raise ValueError('Current feature definitions require the fixed 20-450 Hz analysis band.')
    if cfg['trajectory_bins']<2: raise ValueError('At least two trajectory bins required.')
    if any(k not in FEATURES for k in cfg['figure_features']): raise ValueError('Unknown figure feature.')
    if cfg['flatline_min_s']<=0: raise ValueError('Flatline duration must be positive.')
    keys={f.name for f in fields(Settings)}
    return cfg,Settings(**{k:v for k,v in cfg.items() if k in keys})


def proposed_bounds(t, x, fs):
    """Navigation suggestion only: not exercise events, repetitions or fatigue labels."""
    n=max(1,round(.1*fs))
    # RMS of unrectified signal with only a robust DC offset removed.
    y=np.nan_to_num(x-np.nanmedian(x))
    envelope=np.sqrt(np.convolve(y*y,np.ones(n)/n,mode='same'))
    threshold=.15*np.percentile(envelope,90)
    above=envelope>threshold
    edges=np.r_[0,np.flatnonzero(above[1:]!=above[:-1])+1,len(above)]
    spans=[(int(a),int(b)) for a,b in zip(edges[:-1],edges[1:]) if above[a] and b-a>=round(.2*fs)]
    start=spans[0][0] if spans else 0; end=spans[-1][1] if spans else len(t)
    return float(t[start]),float(t[end-1]+1/fs),envelope,float(threshold)


def minmax_trace(t, x, bins=2000):
    tx=[]; xx=[]
    for b in np.array_split(np.arange(len(t)),min(bins,len(t))):
        ids=sorted([int(b[np.nanargmin(x[b])]),int(b[np.nanargmax(x[b])])])
        tx.extend(float(t[i]) for i in ids); xx.extend(float(x[i]) for i in ids)
    return tx,xx


def initialize():
    """Create pending annotations and local browser data, preserving all human edits."""
    manifest=registry(); path=HERE/'annotations.json'
    annotations=json.loads(path.read_text()) if path.exists() else {'schema_version':1,'recordings':{}}
    output=HERE/'review_data'; output.mkdir(exist_ok=True)
    inventory=[]
    for rid,p in manifest.items():
        raw=pd.read_csv(p); t=raw.Relative_Time_s.to_numpy(); x=raw.Voltage_mV.to_numpy()
        fs=float(1/np.median(np.diff(t))); lo,hi,envelope,threshold=proposed_bounds(t,x,fs)
        source_hash=digest(p)
        current=annotations['recordings'].get(rid)
        if current and current['source_sha256']!=source_hash:
            raise ValueError(f'{rid}: source changed; review prior annotations before reinitializing.')
        if current is None:
            annotations['recordings'][rid]=dict(status='pending',start_s=None,end_s=None,
                candidate_start_s=lo,candidate_end_s=hi,reviewer='',note='',artifact_intervals=[],
                source_sha256=source_hash,updated_utc=None,
                evidence='Complete set to failure per user; active boundaries proposed from EMG, unreviewed.')
        tx,xx=minmax_trace(t,x)
        step=max(1,len(t)//2000)
        tokens=p.stem.split('_'); leg_index=next(i for i,v in enumerate(tokens) if v in ['dr','iz'])
        metadata=dict(recording_id=rid,subject_id=rid.split('_')[0],leg='R' if tokens[leg_index]=='dr' else 'L',set_token='_'.join(tokens[:leg_index]),
                      first_time_s=float(t[0]),end_exclusive_s=float(t[-1]+1/fs),fs=fs,
                      source_sha256=source_hash)
        atomic_json(output/f'{rid}.json',dict(**metadata,t=tx,x=xx,envelope_t=t[::step].tolist(),
                    envelope=envelope[::step].tolist(),suggestion_threshold=threshold))
        inventory.append(metadata)
    atomic_json(path,annotations); atomic_json(output/'inventory.json',inventory)
    print(f'Prepared {len(inventory)} recordings; preserved existing reviews. Boundary proposals remain pending.',flush=True)
    return annotations


def validate_annotation(annotation, info):
    if annotation['status'] not in ['pending','reviewed','exclude']: raise ValueError('Invalid review status.')
    if annotation.get('source_sha256')!=info['source_sha256']: raise ValueError('Source checksum mismatch.')
    start,end=annotation.get('start_s'),annotation.get('end_s')
    if annotation['status']=='reviewed':
        if not isinstance(start,(int,float)) or not isinstance(end,(int,float)) or not np.isfinite([start,end]).all():
            raise ValueError('Reviewed boundaries must be finite numbers.')
        if not info['first_time_s']<=start<end<=info['end_exclusive_s']+1e-8:
            raise ValueError('Boundaries must be ordered and within the recording.')
        if end-start<.5: raise ValueError('Active interval must contain at least one 500 ms window.')
    if annotation['status'] in ['reviewed','exclude'] and not annotation.get('reviewer','').strip():
        raise ValueError('Record a reviewer identifier.')
    if annotation['status']=='exclude' and not annotation.get('note','').strip():
        raise ValueError('Excluding a set requires a documented signal/acquisition reason.')
    for span in annotation.get('artifact_intervals',[]):
        if len(span)!=3: raise ValueError('Artifact rows require start, end and reason.')
        a,b,reason=span
        if not np.isfinite([a,b]).all() or not info['first_time_s']<=a<b<=info['end_exclusive_s']+1e-8 or not str(reason).strip():
            raise ValueError('Invalid artifact interval or missing reason.')


def flatline_mask(adc, minimum):
    mask=np.zeros(len(adc),bool)
    edges=np.r_[0,np.flatnonzero(adc[1:]!=adc[:-1])+1,len(adc)]
    for a,b in zip(edges[:-1],edges[1:]):
        if b-a>=minimum: mask[a:b]=True
    return mask


def aggregate_trajectories(frame, feature_names, bins, eligible_column):
    """Window mean -> equal set mean within subject/leg -> equal subject mean/SD."""
    keep=frame[frame[eligible_column]].copy()
    identifiers=['subject_id','leg','recording_id']
    if keep.empty:
        return pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
    keep['bin']=np.minimum((keep.position_percent*bins/100).astype(int),bins-1)
    per_set=keep.groupby(identifiers+['bin'],observed=True)[feature_names].mean().reset_index()
    # A missing/undefined feature contributes no value for that set, rather than zero.
    per_subject=per_set.groupby(['subject_id','leg','bin'],observed=True)[feature_names].mean().reset_index()
    long=per_subject.melt(id_vars=['subject_id','leg','bin'],value_vars=feature_names,var_name='feature',value_name='value').dropna(subset=['value'])
    summary=long.groupby(['leg','bin','feature'],observed=True).value.agg(mean='mean',sd='std',n_subjects='count').reset_index()
    summary['position_percent']=(summary.bin+.5)*100/bins
    return per_set,per_subject,summary


def plot_trajectories(subjects, summary, names, bins, path, preview=False):
    fig,axes=plt.subplots(len(names),2,figsize=(13,3.1*len(names)),squeeze=False)
    grid=(np.arange(bins)+.5)*100/bins
    for row,name in enumerate(names):
        for col,leg in enumerate(['R','L']):
            ax=axes[row,col]
            for _,group in subjects[subjects.leg.eq(leg)].groupby('subject_id'):
                values=group.set_index('bin')[name].reindex(range(bins)).to_numpy()
                ax.plot(grid,values,color='#abb7bf',lw=.6,alpha=.55)
            stat=summary[(summary.leg==leg)&(summary.feature==name)].set_index('bin').reindex(range(bins))
            ax.plot(grid,stat['mean'],color='#126782',lw=1.8,label='Subject mean')
            ax.fill_between(grid,stat['mean']-stat['sd'],stat['mean']+stat['sd'],color='#126782',alpha=.18,label='Between-subject SD (n>=2)')
            ns=stat.n_subjects.dropna()
            coverage=f'n/bin {int(ns.min())}-{int(ns.max())}' if len(ns) else 'no contributing subjects'
            ax.set(title=f'{name} | {"Right" if leg=="R" else "Left"} | {coverage}',xlabel='Active-interval elapsed-time position (%)',ylabel=FEATURES[name][1])
            ax.set_xlim(0,100); ax.grid(alpha=.2); ax.legend(fontsize=7)
    title='PREVIEW: unreviewed proposed active boundaries' if preview else 'Figure 2: feature trajectories within reviewed active-set intervals'
    fig.suptitle(title,fontsize=14)
    fig.text(.01,.009,'Thin curves: subject means across available sets | No MVC reference | Elapsed position is not repetition count or a fatigue label',fontsize=8)
    fig.tight_layout(rect=(0,.03,1,.96)); fig.savefig(path,dpi=160); plt.close(fig)


def run(config_path=None, preview=False, limit=None):
    config,cfg=configuration(config_path)
    if not (HERE/'annotations.json').exists(): initialize()
    annotations=json.loads((HERE/'annotations.json').read_text())
    entries=list(registry().items())[:limit]
    identity=dict(config=config,annotations=annotations,source_hashes={rid:digest(p) for rid,p in entries},
                  code_hashes={p.name:digest(p) for p in [Path(__file__),ROOT/'feature_explorer/explore_features.py']},
                  preview=preview,recording_ids=[r for r,_ in entries])
    run_id=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:12]
    out=HERE/'outputs'/run_id; out.mkdir(parents=True,exist_ok=True)
    all_frames=[]; summaries=[]; errors=[]
    for index,(rid,path) in enumerate(entries,1):
        ann=annotations['recordings'][rid]
        review_info=json.loads((HERE/'review_data'/f'{rid}.json').read_text())
        validate_annotation(ann,review_info)
        if digest(path)!=ann['source_sha256']: raise ValueError(f'{rid}: original recording checksum changed.')
        try: obj=prepare(rid,path,cfg,compute_features=False,require_valid=False)
        except (ValueError,KeyError) as exc:
            # Do not print a source path or sensitive acquisition text.
            errors.append(dict(recording_id=rid,reason=type(exc).__name__,action='inspect_input_schema_timing_or_length'))
            print(f'{rid}: flagged input error; retained source.',flush=True); continue
        table=obj['frame']; t=obj['t']; raw=pd.read_csv(path); adc=raw.ADC_Value.to_numpy()
        flat=flatline_mask(adc,max(2,round(config['flatline_min_s']*obj['fs'])))
        approved=ann['status']=='reviewed'
        start,end=(ann['start_s'],ann['end_s']) if approved else (ann['candidate_start_s'],ann['candidate_end_s'])
        table['position_percent']=100*(table.center_s-start)/(end-start)
        table['bounds_start_s']=start; table['bounds_end_s']=end
        table['boundary_status']=ann['status']; table['boundary_source']='reviewed_emg_annotation' if approved else 'unreviewed_signal_proposal'
        table['set_id']=rid; table['normalization']='not_applied_MVC_unavailable'
        table['notch_hz']=cfg.notch_hz
        table['in_active_interval']=(table.t_start_s>=start-1e-8)&(table.t_end_exclusive_s<=end+1e-8)
        qc_reasons=[]; manual_flags=[]; flat_flags=[]; low_flags=[]; crest_flags=[]
        for row_index,row in table.iterrows():
            a=int(row.sample_start); b=int(row.sample_end_exclusive)
            reasons=[] if row.valid_window else [row.quality_reason]
            manual=any(row.t_start_s<z and row.t_end_exclusive_s>s for s,z,_ in ann.get('artifact_intervals',[]))
            is_flat=bool(flat[a:b].any()); manual_flags.append(manual); flat_flags.append(is_flat)
            if manual: reasons.append('reviewed_artifact_interval')
            if is_flat: reasons.append('constant_ADC_run_ge_threshold')
            window=obj['x'][a:b]; rms=np.sqrt(np.mean(window**2))
            crest=float(np.max(np.abs(window))/rms) if rms>0 else np.nan
            ff,pp=signal.welch(window,fs=obj['fs'],nperseg=min(256,len(window)),noverlap=min(256,len(window))//2,window='hann',detrend='constant')
            total=pp[ff>0].sum(); fraction=float(pp[(ff>0)&(ff<20)].sum()/total) if total>0 else np.nan
            low_flags.append(bool(fraction>config['review_low_frequency_fraction']))
            crest_flags.append(bool(crest>config['review_crest_factor']))
            table.loc[row_index,'raw_low_frequency_fraction']=fraction
            table.loc[row_index,'raw_crest_factor']=crest
            if row.valid_window and not manual and not is_flat:
                computed=extract_window(obj['filtered'][a:b],obj['fs'],cfg)
                for name,value in computed.items(): table.loc[row_index,name]=value
            qc_reasons.append('|'.join(reasons) if reasons else 'ok')
        table['flag_manual_artifact']=manual_flags; table['flag_flatline']=flat_flags
        table['review_possible_motion_or_drift']=low_flags; table['review_possible_transient']=crest_flags
        table['qc_exclusion_reason']=qc_reasons
        table['signal_eligible']=table.valid_window&~table.flag_manual_artifact&~table.flag_flatline
        table['eligible_figure2']=table.signal_eligible&table.in_active_interval&approved&(ann['status']!='exclude')
        table['eligible_preview']=table.signal_eligible&table.in_active_interval&(ann['status']!='exclude')
        table['exclusion_reason']=table.qc_exclusion_reason
        for i,row in table.iterrows():
            more=[]
            if not row.in_active_interval: more.append('outside_active_bounds')
            if not approved: more.append('set_excluded_by_review' if ann['status']=='exclude' else 'active_bounds_pending_review')
            reasons=([row.qc_exclusion_reason] if row.qc_exclusion_reason!='ok' else [])+more
            table.loc[i,'exclusion_reason']='|'.join(reasons) or 'included'
        table['undefined_features']=table[list(FEATURES)].isna().sum(axis=1)
        all_frames.append(table)
        summaries.append(dict(**obj['info'],boundary_status=ann['status'],active_start_s=start,active_end_s=end,
             eligible_figure2_windows=int(table.eligible_figure2.sum()),eligible_preview_windows=int(table.eligible_preview.sum()),
             manual_artifact_windows=int(table.flag_manual_artifact.sum()),flatline_windows=int(table.flag_flatline.sum()),
             possible_transient_windows=int(table.review_possible_transient.sum()),
             possible_low_frequency_windows=int(table.review_possible_motion_or_drift.sum()),
             mvc_normalization='not_applied_reference_unavailable',resting_SNR='unavailable_no_baseline',
             saturation_assessment='unknown_hardware_rails',notch_applied=cfg.notch_hz is not None,
             notch_rationale='Disabled: audit sideband screening did not establish dominant narrowband interference; no claim of absence.' if cfg.notch_hz is None else 'Explicit configuration choice; document rationale before final reporting.'))
        print(f'[{index}/{len(entries)}] {rid}: {int(table.signal_eligible.sum())} signal-eligible; bounds {ann["status"]}',flush=True)
    if not all_frames: raise ValueError('No recordings processed; inspect input error log.')
    windows=pd.concat(all_frames,ignore_index=True)
    windows.to_csv(out/'window_features.csv',index=False)
    windows.loc[~windows.eligible_figure2,['window_id','subject_id','recording_id','sample_start','sample_end_exclusive','exclusion_reason']].to_csv(out/'window_exclusion_log.csv',index=False)
    pd.DataFrame(summaries).to_csv(out/'recording_processing_log.csv',index=False)
    pd.DataFrame(errors,columns=['recording_id','reason','action']).to_csv(out/'input_errors.csv',index=False)
    pd.DataFrame([dict(feature=k,family=v[0],units=v[1],definition=v[2]) for k,v in FEATURES.items()]).to_csv(out/'feature_dictionary.csv',index=False)
    for provisional,eligibility in [(False,'eligible_figure2')]+([(True,'eligible_preview')] if preview else []):
        prefix='preview_' if provisional else ''
        sets,subjects,summary=aggregate_trajectories(windows,list(FEATURES),config['trajectory_bins'],eligibility)
        if summary.empty: continue
        sets.to_csv(out/f'{prefix}trajectories_by_set.csv',index=False)
        subjects.to_csv(out/f'{prefix}trajectories_by_subject_leg.csv',index=False)
        summary.to_csv(out/f'{prefix}trajectory_summary.csv',index=False)
        plot_trajectories(subjects,summary,config['figure_features'],config['trajectory_bins'],out/('figure2_PREVIEW.png' if provisional else 'figure2.png'),provisional)
        for name in FEATURES:
            folder=out/('preview_feature_trajectories' if provisional else 'feature_trajectories'); folder.mkdir(exist_ok=True)
            plot_trajectories(subjects,summary,[name],config['trajectory_bins'],folder/f'{name}.png',provisional)
    status=dict(recordings_processed=len(summaries),input_errors=len(errors),subjects=windows.subject_id.nunique(),
                window_rows=len(windows),signal_eligible_windows=int(windows.signal_eligible.sum()),
                reviewed_recordings=sum(s['boundary_status']=='reviewed' for s in summaries),
                pending_recordings=sum(s['boundary_status']=='pending' for s in summaries),
                review_excluded_recordings=sum(s['boundary_status']=='exclude' for s in summaries),
                figure2_windows=int(windows.eligible_figure2.sum()),preview_windows=int(windows.eligible_preview.sum()),
                final_figure_created=(out/'figure2.png').exists(),preview_created=(out/'figure2_PREVIEW.png').exists(),
                all_original_hashes_unchanged=all(digest(path)==identity['source_hashes'][rid] for rid,path in entries))
    metadata=dict(run_id=run_id,created_utc=utc_now(),**identity,status=status,
        versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,matplotlib=matplotlib.__version__),
        notes=['No models, labels, imputation, scaling, feature selection or PCA.',
               'Boundary proposals use the signal only for navigation; no physiological ground truth is inferred.',
               'Review flags for low-frequency power/transients are not automatic exclusions.',
               'Current amplitude threshold and edge guard are provisional; no baseline/MVC exists.',
               'Population aggregation is descriptive and is not a parameter fitted for modelling.'])
    atomic_json(out/'run_manifest.json',metadata)
    atomic_json(HERE/'outputs/latest_run.json',dict(run_id=run_id,path=str(out.relative_to(ROOT)),status=status))
    (out/'figure2_status.txt').write_text('Figure 2 uses reviewed bounds only.\n'+json.dumps(status,indent=2)+'\nPreview, if present, uses unreviewed proposals and must not be reported as final.\n',encoding='utf-8')
    print(json.dumps(status,indent=2),flush=True)
    print(f'Output: processing/outputs/{run_id}',flush=True)
    return windows,out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['init','run'])
    parser.add_argument('--config'); parser.add_argument('--preview',action='store_true')
    parser.add_argument('--limit',type=int,help='Smoke-test only: first N recordings')
    args=parser.parse_args()
    if args.action=='init': initialize()
    else: run(args.config,args.preview,args.limit)


if __name__=='__main__': main()
