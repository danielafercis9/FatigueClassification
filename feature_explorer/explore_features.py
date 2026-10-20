"""Offline, exploratory sEMG features for one or two complete recording files.

Run from project root: py -3.13 feature_explorer/explore_features.py --no-show
Accepts anonymous audit recording IDs, exact CSV basenames, or CSV paths.
Does not assign fatigue classes, select predictors, reduce dimensions or train.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import platform

import matplotlib
import numpy as np
import pandas as pd
import scipy
from scipy import signal, stats
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from project_paths import raw_data_dir, resolve_recording_path


@dataclass(frozen=True)
class Settings:
    window_s: float = .5
    hop_s: float = .25
    bandpass_hz: tuple = (20., 450.)
    butter_order: int = 4  # bandpass design order; forward/backward doubles response order
    notch_hz: float | None = None  # optional; evidence needed; never enabled automatically
    notch_q: float = 30.
    timing_tolerance_s: float = .00005
    edge_guard_s: float = .1
    count_threshold_mv: float = 2.  # exploratory device-unit threshold, NOT baseline-derived
    welch_nperseg: int = 256
    sample_entropy: bool = True
    entropy_r_sd: float = .2
    x_axis: str = 'seconds'  # 'percent': percent of selected interval, not rep count


# name: (family, units, implementation definition)
FEATURES = {
    'RMS': ('Amplitude', 'header mV', 'sqrt(mean(x**2))'),
    'MAV': ('Amplitude', 'header mV', 'mean(abs(x))'),
    'IEMG': ('Amplitude', 'header mV s', 'sum(abs(x))/fs; MAV * fixed window duration'),
    'Variance': ('Amplitude', '(header mV)^2', 'population variance, ddof=0'),
    'Waveform_length': ('Amplitude', 'header mV', 'sum(abs(diff(x)))'),
    'DASDV': ('Amplitude', 'header mV', 'sqrt(mean(diff(x)**2))'),
    'Peak_to_peak': ('Amplitude', 'header mV', 'max(x)-min(x); sensitive to transients'),
    'Crest_factor': ('Shape', 'ratio', 'max(abs(x))/RMS; sensitive to transients'),
    'Skewness': ('Shape', 'unitless', 'bias-corrected scipy.stats.skew'),
    'Excess_kurtosis': ('Shape', 'unitless', 'bias-corrected Fisher kurtosis; Gaussian reference=0'),
    'ZC_rate': ('Threshold counts', 'events/s', 'strict adjacent sign crossing AND abs(diff)>=threshold; zeros excluded'),
    'SSC_rate': ('Threshold counts', 'events/s', 'local extremum AND both adjacent differences >=threshold'),
    'WAMP_rate': ('Threshold counts', 'events/s', 'abs(diff)>=threshold; opportunity duration=(N-1)/fs'),
    'MDF': ('Spectrum', 'Hz', 'first Welch bin containing >=50% of 20-450 Hz band power'),
    'MNF': ('Spectrum', 'Hz', 'sum(f*P)/sum(P), analysis band only'),
    'Peak_frequency': ('Spectrum', 'Hz', 'frequency bin of largest PSD, analysis band only'),
    'Spectral_entropy': ('Spectrum', '0 to 1', '-sum(q*log(q))/log(K), normalized band PSD bins'),
    'Spectral_bandwidth': ('Spectrum', 'Hz', 'sqrt(sum(q*(f-MNF)**2))'),
    'Relative_power_20_60': ('Spectrum', 'fraction', '20<=f<60 Hz power / analysis band power'),
    'Relative_power_60_150': ('Spectrum', 'fraction', '60<=f<150 Hz power / analysis band power'),
    'Relative_power_150_450': ('Spectrum', 'fraction', '150<=f<=450 Hz power / analysis band power'),
    'Low_high_power_ratio': ('Spectrum', 'ratio', '20-60 Hz power / 60-450 Hz power; exploratory bands'),
    'Hjorth_activity': ('Complexity', '(header mV)^2', 'var(x); exactly redundant with Variance'),
    'Hjorth_mobility': ('Complexity', '1/s', 'sqrt(var(diff(x)*fs)/var(x))'),
    'Hjorth_complexity': ('Complexity', 'ratio', 'mobility(first derivative)/mobility(x)'),
    'Permutation_entropy': ('Complexity', '0 to 1', 'order 3, delay 1; stable time-order tie handling; normalize by log(3!)'),
    'Sample_entropy': ('Complexity', 'unitless', 'm=2, r=.2*window SD default; Chebyshev distance; self matches excluded'),
    **{f'AR{i}': ('Autoregressive', 'unitless', 'OLS x[t]=sum(a_k*x[t-k]), order 4, demeaned, no intercept') for i in range(1,5)},
    **{f'Haar_D{i}_energy': ('Wavelet', 'fraction', 'orthonormal Haar detail energy / total coefficient energy; 4 levels') for i in range(1,5)},
    'Haar_A4_energy': ('Wavelet', 'fraction', 'level-4 approximation energy / total coefficient energy'),
}


def ratio(a, b):
    return float(a/b) if np.isfinite(b) and b > 0 else np.nan


def permutation_entropy(x):
    patterns = np.argsort(np.lib.stride_tricks.sliding_window_view(x, 3), axis=1, kind='stable')
    _, counts = np.unique(patterns, axis=0, return_counts=True)
    q = counts/counts.sum()
    return float(-np.sum(q*np.log(q))/math.log(6))


def sample_entropy(x, r_sd):
    sd = np.std(x)
    if sd <= 0: return np.nan
    # Same N-m starting positions for length m and m+1 templates.
    templates = np.lib.stride_tricks.sliding_window_view(x, 3)
    def matches(z):
        tree = cKDTree(z)
        # count_neighbors counts ordered pairs including self; subtract self, divide by 2.
        return (tree.count_neighbors(tree, r_sd*sd, p=np.inf)-len(z))/2
    b, a = matches(templates[:, :2]), matches(templates)
    return float(-np.log(a/b)) if a > 0 and b > 0 else np.nan


def extract_window(x, fs, cfg):
    n = len(x); dx = np.diff(x); var = np.var(x); rms = np.sqrt(np.mean(x*x))
    q = cfg.count_threshold_mv
    out = dict(RMS=rms, MAV=np.mean(np.abs(x)), IEMG=np.abs(x).sum()/fs,
               Variance=var, Waveform_length=np.abs(dx).sum(), DASDV=np.sqrt(np.mean(dx*dx)),
               Peak_to_peak=np.ptp(x), Crest_factor=ratio(np.max(np.abs(x)), rms),
               Skewness=stats.skew(x, bias=False) if var>0 else np.nan,
               Excess_kurtosis=stats.kurtosis(x, fisher=True, bias=False) if var>0 else np.nan,
               ZC_rate=np.sum((x[:-1]*x[1:]<0)&(np.abs(dx)>=q))/((n-1)/fs),
               SSC_rate=np.sum((dx[:-1]*dx[1:]<0)&(np.abs(dx[:-1])>=q)&(np.abs(dx[1:])>=q))/((n-2)/fs),
               WAMP_rate=np.sum(np.abs(dx)>=q)/((n-1)/fs))
    f, p = signal.welch(x, fs=fs, window='hann', nperseg=min(cfg.welch_nperseg,n),
                        noverlap=min(cfg.welch_nperseg,n)//2, detrend='constant', scaling='density')
    mask=(f>=cfg.bandpass_hz[0])&(f<=cfg.bandpass_hz[1]); f,p=f[mask],p[mask]
    total=p.sum()
    if total>0:
        prob=p/total; mnf=np.dot(f,prob); low=p[f<60].sum(); high=p[f>=60].sum()
        out.update(MDF=f[np.searchsorted(np.cumsum(p),total/2)], MNF=mnf,
                   Peak_frequency=f[np.argmax(p)],
                   Spectral_entropy=-np.sum(prob[prob>0]*np.log(prob[prob>0]))/np.log(len(prob)),
                   Spectral_bandwidth=np.sqrt(np.dot(prob,(f-mnf)**2)),
                   Relative_power_20_60=low/total,
                   Relative_power_60_150=p[(f>=60)&(f<150)].sum()/total,
                   Relative_power_150_450=p[f>=150].sum()/total,
                   Low_high_power_ratio=ratio(low,high))
    else:
        out.update({k:np.nan for k,(family,_,_) in FEATURES.items() if family=='Spectrum'})
    dvar=np.var(dx*fs); mobility=np.sqrt(ratio(dvar,var))
    d_mobility=np.sqrt(ratio(np.var(np.diff(x,n=2)*fs**2),dvar))
    out.update(Hjorth_activity=var, Hjorth_mobility=mobility,
               Hjorth_complexity=ratio(d_mobility,mobility),
               Permutation_entropy=permutation_entropy(x),
               Sample_entropy=sample_entropy(x,cfg.entropy_r_sd) if cfg.sample_entropy else np.nan)
    centered=x-x.mean(); y=centered[4:]
    design=np.column_stack([centered[4-i:n-i] for i in range(1,5)])
    coefficients=np.linalg.lstsq(design,y,rcond=None)[0] if var>0 else np.full(4,np.nan)
    out.update({f'AR{i+1}':v for i,v in enumerate(coefficients)})
    # Trim <=15 trailing samples so four orthonormal Haar levels require no padding.
    a=centered[:n//16*16].copy(); energies=[]
    for level in range(4):
        even,odd=a[::2],a[1::2]
        detail=(even-odd)/np.sqrt(2); a=(even+odd)/np.sqrt(2)
        energies.append(float(np.dot(detail,detail)))
    energies.append(float(np.dot(a,a))); energy=sum(energies)
    out.update({f'Haar_D{i+1}_energy':ratio(e,energy) for i,e in enumerate(energies[:4])})
    out['Haar_A4_energy']=ratio(energies[-1],energy)
    return out


def resolve_files(files):
    manifest_path=ROOT/'audit/private/recording_paths.json'
    if not manifest_path.exists(): raise ValueError('Run audit/audit_data.py first to create the private ID mapping.')
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    registry={r['recording_id']:resolve_recording_path(r['source_relative_path']).resolve() for r in manifest}
    resolved=[]
    for item in files:
        if item in registry: path=registry[item]
        else:
            candidate=Path(item).expanduser()
            if candidate.is_file(): path=candidate.resolve()
            elif (ROOT/candidate).is_file(): path=(ROOT/candidate).resolve()
            else:
                matches=[p for p in registry.values() if p.name==item]
                if len(matches)!=1: raise ValueError('CSV basename not found uniquely; use an exact path or audit ID.')
                path=matches[0]
        matches=[rid for rid,p in registry.items() if p==path]
        if len(matches)!=1: raise ValueError('Selected file is outside the audited dataset; audit it before exploring.')
        resolved.append((matches[0],path))
    if not 1<=len(resolved)<=2: raise ValueError('Choose one or two files.')
    if len({rid for rid,_ in resolved})!=len(resolved): raise ValueError('The two selected files must differ.')
    return resolved


def prepare(rid, path, cfg, bounds=None, *, compute_features=True, require_valid=True):
    raw=pd.read_csv(path)
    for c in ['Relative_Time_s','Voltage_mV','ADC_Value']:
        if c not in raw: raise ValueError(f'{rid}: required column missing: {c}')
        raw[c]=pd.to_numeric(raw[c],errors='coerce')
    t=raw.Relative_Time_s.to_numpy(); x=raw.Voltage_mV.to_numpy()
    if not np.isfinite(t).all() or np.any(np.diff(t)<=0): raise ValueError(f'{rid}: timestamps must be finite and strictly increasing.')
    dt=np.diff(t); fs=float(1/np.median(dt)); lo,hi=cfg.bandpass_hz
    if not 0<lo<hi<fs/2: raise ValueError('Bandpass must lie strictly inside Nyquist.')
    if cfg.window_s<=0 or not 0<cfg.hop_s<=cfg.window_s: raise ValueError('Require 0 < hop <= window.')
    if cfg.count_threshold_mv<0 or cfg.entropy_r_sd<=0: raise ValueError('Thresholds must be nonnegative; entropy tolerance positive.')
    if cfg.edge_guard_s<0 or cfg.timing_tolerance_s<0 or cfg.notch_q<=0: raise ValueError('Invalid edge guard, timing tolerance or notch Q.')
    irregular=np.abs(dt-1/fs)>cfg.timing_tolerance_s
    cuts=np.r_[0,np.flatnonzero(irregular)+1,len(x)]
    filtered=np.full(len(x),np.nan); segment=np.full(len(x),-1,int)
    sos=signal.butter(cfg.butter_order,[lo,hi],btype='bandpass',fs=fs,output='sos')
    if cfg.notch_hz is not None:
        if not 0<cfg.notch_hz<fs/2: raise ValueError('Notch frequency outside valid band.')
        b,a=signal.iirnotch(cfg.notch_hz,cfg.notch_q,fs=fs)
        sos=np.vstack([sos,signal.tf2sos(b,a)])
    finite=np.isfinite(x); finite_cuts=np.flatnonzero(finite[1:]!=finite[:-1])+1
    cuts=np.unique(np.r_[cuts,finite_cuts]); guard=round(cfg.edge_guard_s*fs)
    for sid,(a,b) in enumerate(zip(cuts[:-1],cuts[1:])):
        if not finite[a:b].all() or b-a<max(64,2*guard+16): continue
        try: y=signal.sosfiltfilt(sos,x[a:b])
        except ValueError: continue
        filtered[a:b]=y
        segment[a+guard:b-guard if guard else b]=sid
    start_t,end_t=(float(t[0]),float(t[-1]+1/fs)) if bounds is None else bounds
    if not t[0]<=start_t<end_t<=t[-1]+1/fs+1e-8: raise ValueError('Bounds must lie within the original recording.')
    n=round(cfg.window_s*fs); hop=round(cfg.hop_s*fs)
    if n<32 or hop<1: raise ValueError('Window must contain >=32 samples and hop >=1.')
    begin=int(np.searchsorted(t,start_t)); stop=int(np.searchsorted(t,end_t))
    rows=[]
    for a in range(begin,stop-n+1,hop):
        b=a+n; seg=segment[a:b]; valid=seg[0]>=0 and np.all(seg==seg[0]) and np.isfinite(filtered[a:b]).all()
        reason='ok' if valid else 'timing_gap_nonfinite_or_segment_edge'
        row=dict(window_id=f'{rid}_W{a:07d}',subject_id=rid.split('_')[0],recording_id=rid,
                 set_token='_'.join(path.stem.split('_')[:path.stem.split('_').index('dr' if '_dr_' in path.stem else 'iz')]),
                 leg='R' if '_dr_' in path.stem else 'L', sample_start=a,sample_end_exclusive=b,
                 t_start_s=t[a],t_last_observed_s=t[b-1],t_end_exclusive_s=t[b-1]+1/fs,
                 center_s=(t[a]+t[b-1])/2,position_percent=100*((t[a]+t[b-1])/2-start_t)/(end_t-start_t),
                 fs_used_hz=fs,n_samples=n,segment_id=int(seg[0]) if valid else -1,
                 valid_window=bool(valid),quality_reason=reason)
        row.update(extract_window(filtered[a:b],fs,cfg) if valid and compute_features else {k:np.nan for k in FEATURES})
        rows.append(row)
    frame=pd.DataFrame(rows)
    if frame.empty or (require_valid and not frame.valid_window.any()): raise ValueError(f'{rid}: no valid windows under these timing/edge/bounds settings.')
    info=dict(recording_id=rid,subject_id=rid.split('_')[0],leg=frame.leg.iloc[0],set_token=frame.set_token.iloc[0],
              samples=len(t),duration_s=float(t[-1]-t[0]),fs_median_hz=fs,
              fs_effective_hz=(len(t)-1)/(t[-1]-t[0]),irregular_intervals=int(irregular.sum()),
              longest_interval_s=float(dt.max()),valid_windows=int(frame.valid_window.sum()),
              flagged_windows=int((~frame.valid_window).sum()),
              selected_bounds_s=[start_t,end_t],active_bounds_verified=False,
              set_completeness_evidence='User states each file is a complete set to failure; no independent markers.',
              sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return dict(rid=rid,t=t,x=x,filtered=filtered,segment=segment,frame=frame,fs=fs,info=info)


def make_figures(data,cfg):
    import matplotlib.pyplot as plt
    colors=['#126782','#b34c16']; figs=[]
    def finish(fig,title):
        fig.suptitle(title,fontsize=15)
        fig.text(.01,.008,'Exploratory only | CSV header units; no MVC calibration | No fatigue labels or feature selection',fontsize=8)
        fig.tight_layout(rect=(0,.025,1,.96)); figs.append(fig)
    # Diagnostic plots keep originals and filtered segments visibly distinct.
    for obj,color in zip(data,colors):
        rid=obj['rid']; t=obj['t']; fs=obj['fs']; fig,axes=plt.subplots(3,1,figsize=(13,10))
        # Keep all raw samples to preserve transients; matplotlib simplifies only display.
        axes[0].plot(t,obj['x'],lw=.35,color='0.6',label='Original')
        first=True
        for sid in np.unique(obj['segment'][obj['segment']>=0]):
            ids=np.flatnonzero(obj['segment']==sid)
            axes[0].plot(t[ids],obj['filtered'][ids],lw=.35,color=color,
                         label='Bandpassed, per contiguous segment' if first else None)
            first=False
        axes[0].set(ylabel='Voltage (header mV)',xlabel='Recording time (s)'); axes[0].legend(fontsize=8)
        axes[1].plot(t[1:],np.diff(t)*1000,lw=.5,color=color)
        axes[1].set(xlabel='Recording time (s)',ylabel='Timestamp interval (ms)')
        axes[2].set(xlabel='Recording time (s)',ylabel='Frequency (Hz)')
        meshes=[]
        # One common scale across segments, no interpolation over gaps.
        specs=[]
        for sid in np.unique(obj['segment'][obj['segment']>=0]):
            ids=np.flatnonzero(obj['segment']==sid)
            if len(ids)<256: continue
            f,tt,p=signal.spectrogram(obj['filtered'][ids],fs=fs,window='hann',nperseg=256,
                                     noverlap=128,detrend='constant',scaling='density',mode='psd')
            specs.append((f,tt+t[ids[0]],10*np.log10(np.maximum(p,1e-20))))
        if specs:
            v=np.concatenate([s[2].ravel() for s in specs]); vmin,vmax=np.percentile(v,[5,99])
            for f,tt,p in specs:
                if len(tt)>1: meshes.append(axes[2].pcolormesh(tt,f,p,shading='auto',cmap='magma',vmin=vmin,vmax=vmax))
            if meshes: fig.colorbar(meshes[0],ax=axes[2],label='PSD dB [header mV²/Hz]')
        axes[2].set_ylim(0,fs/2)
        for ax in axes: ax.grid(alpha=.2)
        info=obj['info']
        finish(fig,f'{rid}: original signal, timing and spectrogram | {info["valid_windows"]} valid / {info["flagged_windows"]} flagged windows')
    groups=[('Amplitude',['RMS','MAV','IEMG','Variance','Waveform_length','DASDV','Peak_to_peak']),
            ('Waveform shape and threshold counts',['Crest_factor','Skewness','Excess_kurtosis','ZC_rate','SSC_rate','WAMP_rate']),
            ('Spectral features',[k for k,v in FEATURES.items() if v[0]=='Spectrum']),
            ('Complexity and autoregressive features',[k for k,v in FEATURES.items() if v[0] in ['Complexity','Autoregressive']]),
            ('Haar wavelet energy fractions',[k for k,v in FEATURES.items() if v[0]=='Wavelet'])]
    for title,features in groups:
        fig,axes=plt.subplots(math.ceil(len(features)/3),3,figsize=(14,3.2*math.ceil(len(features)/3)),squeeze=False)
        for ax,name in zip(axes.ravel(),features):
            for obj,color in zip(data,colors):
                df=obj['frame']; xx=df.center_s if cfg.x_axis=='seconds' else df.position_percent
                ax.plot(xx,df[name],color=color,lw=.85,marker='.',ms=2,label=obj['rid'])
                # Rejected windows retain NaN values, visibly breaking trajectories.
            ax.set_title(name,fontsize=10); ax.set_ylabel(FEATURES[name][1],fontsize=8)
            ax.set_xlabel('Recording time (s)' if cfg.x_axis=='seconds' else 'Selected-interval position (%)',fontsize=8)
            ax.grid(alpha=.25); ax.legend(fontsize=7)
        for ax in axes.ravel()[len(features):]: ax.set_visible(False)
        finish(fig,title+' | each file shown independently')
    # Per-recording correlation, never pooled across participants/sets.
    for obj in data:
        df=obj['frame'].loc[lambda d:d.valid_window,list(FEATURES)]
        usable=[c for c in df if df[c].count()>=3 and df[c].std()>1e-12]
        corr=df[usable].corr(method='spearman',min_periods=3)
        fig,ax=plt.subplots(figsize=(15,13))
        im=ax.imshow(corr,vmin=-1,vmax=1,cmap='RdBu_r')
        ax.set_xticks(range(len(usable)),usable,rotation=90,fontsize=7)
        ax.set_yticks(range(len(usable)),usable,fontsize=7)
        fig.colorbar(im,ax=ax,label='Spearman correlation (descriptive; overlapping windows)')
        finish(fig,obj['rid']+': feature-to-feature redundancy, not evidence of fatigue relevance')
    return figs


def explore(files=('S001_REC01','S014_REC08'),cfg=None,output_dir=None,show=True,bounds=None):
    """bounds optionally maps recording IDs to (start_s,end_s); no automated trimming."""
    cfg=cfg or Settings()
    if cfg.x_axis not in ['seconds','percent']: raise ValueError('x_axis must be seconds or percent.')
    resolved=resolve_files(files)
    data=[prepare(rid,path,cfg,(bounds or {}).get(rid)) for rid,path in resolved]
    out=Path(output_dir) if output_dir else ROOT/'feature_explorer/outputs'/('__'.join(d['rid'] for d in data))
    out.mkdir(parents=True,exist_ok=True)
    frames=pd.concat([d['frame'] for d in data],ignore_index=True)
    frames.to_csv(out/'exploratory_features.csv',index=False)
    dictionary=pd.DataFrame([dict(feature=k,family=v[0],units=v[1],definition=v[2]) for k,v in FEATURES.items()])
    dictionary.to_csv(out/'feature_dictionary.csv',index=False)
    relations=[]
    for d in data:
        df=d['frame']; valid=df.loc[df.valid_window]
        corr=valid[list(FEATURES)].corr(method='spearman',min_periods=3)
        corr.to_csv(out/f'{d["rid"]}_spearman.csv')
        for name in FEATURES:
            pair=valid[['position_percent',name]].dropna()
            rho=pair.corr(method='spearman').iloc[0,1] if len(pair)>=3 and pair[name].nunique()>1 else np.nan
            relations.append(dict(recording_id=d['rid'],feature=name,n_defined=len(pair),rho_recording_position=rho))
        print(f'{d["rid"]}: {d["info"]["valid_windows"]} valid, {d["info"]["flagged_windows"]} flagged windows; {d["info"]["irregular_intervals"]} irregular intervals.')
    pd.DataFrame(relations).to_csv(out/'descriptive_position_correlations.csv',index=False)
    manifest=dict(settings=asdict(cfg),recordings=[d['info'] for d in data],
                  versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,matplotlib=matplotlib.__version__),
                  notes=['No fatigue labels, feature selection, PCA, hypothesis tests or models.',
                         'Recordings are complete sets to failure according to user testimony; active bounds unverified.',
                         'Position correlation is not physiological validation; overlapping windows are dependent.',
                         'Later sets may start with residual fatigue; between-set recovery is undocumented.',
                         'Features are exploratory. Whole selected interval includes lead-in/out unless explicit bounds supplied.',
                         'All outputs contain restricted course data; do not publish or upload.'])
    (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    figs=make_figures(data,cfg)
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(out/'feature_explorer.pdf') as pdf:
        for i,fig in enumerate(figs,1):
            fig.savefig(out/f'figure_{i:02d}.png',dpi=130)
            pdf.savefig(fig)
    import html
    options=''.join(f'<option value="figure_{i:02d}.png">{html.escape(fig._suptitle.get_text())}</option>' for i,fig in enumerate(figs,1))
    viewer='''<!doctype html><meta charset="utf-8"><title>Local sEMG feature explorer</title>
<style>body{font:16px system-ui;margin:24px;background:#f5f7fa;color:#172b3a}select{width:100%;padding:12px;font:inherit}img{width:100%;background:white}p{max-width:1000px}</style>
<h1>sEMG feature explorer</h1><p>Local restricted data. Choose a figure below. Correlation shows redundancy or recording progression, not independently verified fatigue. No labels, feature selection or models.</p>
<select onchange="document.getElementById('figure').src=this.value">'''+options+'''</select><img id="figure" src="figure_01.png" alt="Selected EMG feature figure">'''
    (out/'index.html').write_text(viewer,encoding='utf-8')
    print('Saved local figures, PDF, window features, dictionary and configuration.')
    if show:
        import matplotlib.pyplot as plt
        plt.show()
    else:
        import matplotlib.pyplot as plt
        for fig in figs: plt.close(fig)
    return frames,figs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('files',nargs='*',help='One/two audit IDs, exact CSV basenames or paths')
    parser.add_argument('--no-show',action='store_true')
    parser.add_argument('--percent',action='store_true')
    args=parser.parse_args()
    if args.no_show: matplotlib.use('Agg')
    explore(args.files or ['S001_REC01','S014_REC08'],Settings(x_axis='percent' if args.percent else 'seconds'),show=not args.no_show)


if __name__=='__main__': main()
