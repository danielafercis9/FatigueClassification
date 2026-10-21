"""Read-only local sEMG audit. No filtering, labels, or model training.

Run from the project root with Python, numpy, pandas and Pillow.
All generated outputs are restricted. Never publish the private identity key.
"""
from pathlib import Path
import hashlib
import json
import platform
import re
from collections import Counter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from project_paths import raw_data_dir, resolve_recording_path
OUT = ROOT / 'audit'
DATA = raw_data_dir()
PRIVATE = OUT / 'private'
PRIVATE.mkdir(exist_ok=True)
key_path = PRIVATE / 'subject_key.json'
folders = sorted(p for p in DATA.iterdir() if p.is_dir())
key = json.loads(key_path.read_text()) if key_path.exists() else {}
for folder in folders:
    if folder.name not in key:
        key[folder.name] = f'S{len(key)+1:03d}'
key_path.write_text(json.dumps(key, indent=2), encoding='utf-8')

def psd_welch(x, fs, nperseg=1024):
    """Hann, 50% overlap, constant detrending, one-sided density; audit only."""
    n = min(nperseg, len(x))
    w = 0.5 - 0.5*np.cos(2*np.pi*np.arange(n)/n)
    windows = np.lib.stride_tricks.sliding_window_view(x, n)[::max(1,n//2)]
    ft = np.fft.rfft((windows-windows.mean(axis=1, keepdims=True))*w, axis=1)
    p = (np.abs(ft)**2).mean(axis=0)/(fs*(w*w).sum())
    p[1:-1 if n%2==0 else None] *= 2
    return np.fft.rfftfreq(n, 1/fs), p

def longest_run(x):
    if not len(x): return 0
    edges = np.r_[0, np.flatnonzero(x[1:] != x[:-1])+1, len(x)]
    return int(np.diff(edges).max())

records, private_manifest, traces, hashes, signal_hashes, anomalies = [], [], {}, {}, {}, []
for folder in folders:
    sid = key[folder.name]
    for j, path in enumerate(sorted(folder.glob('*.csv')), 1):
        rid = f'{sid}_REC{j:02d}'
        # Match known leg tokens, retaining all preceding tokens as uninterpreted identifiers.
        match = re.match(r'^(.*?)_(dr|iz)_', path.stem)
        token, leg = (match[1], {'dr':'R','iz':'L'}[match[2]]) if match else ('unknown','unknown')
        frame = pd.read_csv(path)
        rec = dict(subject_id=sid, recording_id=rid, source_token=token, leg=leg,
                   bytes=path.stat().st_size, columns='|'.join(frame.columns),
                   dtypes='|'.join(str(v) for v in frame.dtypes), samples=len(frame),
                   missing_cells=int(frame.isna().sum().sum()),
                   duplicate_rows=int(frame.duplicated().sum()))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rec['sha256'] = digest
        hashes.setdefault(digest, []).append(rid)
        private_manifest.append(dict(recording_id=rid, source_relative_path=str(Path('Dataset') / 'EMG - RAW data' / path.relative_to(DATA))))
        num = frame.apply(pd.to_numeric, errors='coerce')
        rec['nonfinite_cells'] = int((~np.isfinite(num.to_numpy())).sum())
        t = num['Relative_Time_s'].to_numpy(); x = num['Voltage_mV'].to_numpy(); a = num['ADC_Value'].to_numpy()
        dt = np.diff(t); positive = dt[dt>0]; med = float(np.median(positive))
        signal_digest=hashlib.sha256(a.tobytes()).hexdigest()
        signal_hashes.setdefault(signal_digest,[]).append(rid)
        for k in np.flatnonzero(np.abs(dt-.001)>1e-7):
            anomalies.append(dict(recording_id=rid, sample_before=int(k), sample_after=int(k+1),
                                  time_before_s=t[k], time_after_s=t[k+1], interval_s=dt[k]))
        fs = 1/med
        rec.update(t_start_s=t[0], t_end_s=t[-1], duration_s=t[-1]-t[0],
                   dt_min_s=dt.min(), dt_median_s=med, dt_max_s=dt.max(),
                   dt_p01_s=np.quantile(dt,.01), dt_p99_s=np.quantile(dt,.99),
                   nonincreasing_dt=int((dt<=0).sum()), time_resets=int((dt<0).sum()),
                   duplicate_timestamps=int(frame['Relative_Time_s'].duplicated().sum()),
                   fs_median_hz=fs, fs_effective_hz=(len(t)-1)/(t[-1]-t[0]),
                   gaps_gt_1p5_median=int((dt>1.5*med+1e-9).sum()),
                   gaps_gt_2_median=int((dt>2*med+1e-9).sum()),
                   interval_not_1ms=int((np.abs(dt-.001)>1e-7).sum()),
                   voltage_min=x.min(), voltage_max=x.max(), voltage_mean=x.mean(),
                   voltage_std=x.std(), voltage_rms=np.sqrt(np.mean(x*x)),
                   adc_min=a.min(), adc_max=a.max(), adc_unique=len(np.unique(a)),
                   adc_noninteger=int((np.abs(a-np.round(a))>1e-9).sum()),
                   negative_fraction=np.mean(x<0), zero_fraction=np.mean(x==0),
                   equal_adjacent_fraction=np.mean(np.diff(a)==0),
                   longest_constant_samples=longest_run(a),
                   extrema_fraction=np.mean((a==a.min())|(a==a.max())),
                   max_step_mV=np.max(np.abs(np.diff(x))))
        mask=a!=0
        scale=np.dot(a[mask],x[mask])/np.dot(a[mask],a[mask])
        rec['voltage_per_adc_slope']=scale
        rec['voltage_adc_max_abs_residual']=np.max(np.abs(x-scale*a))
        # Quantify timing change within each file rather than relying on median alone.
        chunks=np.array_split(dt,10)
        rec['decile_dt_medians_s']='|'.join(f'{np.median(c):.8g}' for c in chunks)
        rec['decile_fs_effective_hz']='|'.join(f'{len(c)/c.sum():.5f}' for c in chunks)
        f,p=psd_welch(x,fs)
        total=p[1:].sum()
        rec['psd_peak_hz']=f[1+np.argmax(p[1:])]
        for label,lo,hi in [('below20',0,20),('20to450',20,450),('above450',450,fs/2+1),('50pm2',48,52),('60pm2',58,62)]:
            rec['power_fraction_'+label]=p[(f>=lo)&(f<hi)&(f>0)].sum()/total if total>0 else np.nan
        for hz in (50,60):
            peak=p[np.abs(f-hz)<=2].mean()
            side=p[((f>=hz-10)&(f<=hz-4))|((f>=hz+4)&(f<=hz+10))].mean()
            rec[f'line_{hz}_sideband_ratio']=peak/side if side>0 else np.nan
        # 100 ms local metrics identify candidate dropouts without declaring rejection.
        block=max(1,round(.1*fs)); cropped=a[:len(a)//block*block].reshape(-1,block)
        sd=cropped.std(axis=1)
        rec['blocks100ms_std_lt1adc']=int((sd<1).sum())
        rec['blocks100ms_count']=len(sd)
        traces[rid]=(t,x,a,f,p)
        records.append(rec)

df=pd.DataFrame(records)
df.to_csv(OUT/'recording_inventory.csv',index=False)
pd.DataFrame(anomalies).to_csv(OUT/'timing_anomalies.csv',index=False)
(PRIVATE/'recording_paths.json').write_text(json.dumps(private_manifest,indent=2),encoding='utf-8')
counts=df.groupby(['subject_id','leg']).agg(recordings=('recording_id','size'), samples=('samples','sum'), duration_s=('duration_s','sum')).reset_index()
counts.to_csv(OUT/'subject_inventory.csv',index=False)
summary=dict(subject_folders=len(folders), recordings=len(df), samples=int(df.samples.sum()),
             bytes=int(df.bytes.sum()), column_schemas=df['columns'].value_counts().to_dict(),
             source_token_counts=df.source_token.value_counts().to_dict(),
             leg_counts=df.leg.value_counts().to_dict(),
             exact_duplicate_file_groups=[v for v in hashes.values() if len(v)>1],
             exact_duplicate_adc_array_groups=[v for v in signal_hashes.values() if len(v)>1],
             files_with_irregular_intervals=int((df.interval_not_1ms>0).sum()),
             numeric_ranges={c:{'min':float(df[c].min()),'median':float(df[c].median()),'max':float(df[c].max())} for c in ['samples','duration_s','fs_effective_hz','fs_median_hz','dt_max_s','voltage_min','voltage_max','voltage_mean','voltage_rms','longest_constant_samples','power_fraction_below20','power_fraction_50pm2','power_fraction_60pm2']},
             totals={c:int(df[c].sum()) for c in ['missing_cells','nonfinite_cells','duplicate_rows','nonincreasing_dt','time_resets','duplicate_timestamps','gaps_gt_1p5_median','gaps_gt_2_median','interval_not_1ms']},
             versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__),
             notes=['IDs are local pseudonyms backed by a restricted persisted key, not verified coordinator IDs.',
                    'Recording tokens have no assumed repetition/set meaning.',
                    'Spectra assume uniform sampling at median timestamp interval and are descriptive only.',
                    'No files excluded, no signals filtered, no labels assigned, no models trained.'])
(OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
print('\nSUBJECT INVENTORY\n'+counts.to_string(index=False))
print('\nMOST NONUNIFORM\n'+df.nlargest(10,'gaps_gt_1p5_median')[['recording_id','source_token','leg','samples','duration_s','fs_median_hz','fs_effective_hz','dt_max_s','gaps_gt_1p5_median','decile_dt_medians_s']].to_string(index=False))
print('\nPOTENTIAL QUALITY FLAGS\n'+df.nlargest(12,'power_fraction_below20')[['recording_id','source_token','leg','longest_constant_samples','zero_fraction','voltage_min','voltage_max','power_fraction_below20','line_50_sideband_ratio','line_60_sideband_ratio']].to_string(index=False))

# Render static audit plots using Pillow, avoiding dependency installation.
from PIL import Image, ImageDraw, ImageFont
def portable_font(size):
    for name in ['DejaVuSans.ttf','Arial.ttf']:
        try:return ImageFont.truetype(name,size)
        except OSError:pass
    return ImageFont.load_default(size=size)

font=portable_font(18)
small=portable_font(14)
titlefont=portable_font(22)

def panel(draw, rect, xx, yy, title, xlabel, ylabel, logy=False, color='#176b9a'):
    left,top,right,bottom=rect
    draw.text((left,top-31),title,fill='black',font=font)
    xx=np.asarray(xx); yy=np.asarray(yy)
    if logy: yy=10*np.log10(np.maximum(yy,1e-16))
    xmin,xmax=float(xx.min()),float(xx.max()); ymin,ymax=float(yy.min()),float(yy.max())
    if ymax==ymin:ymax+=1
    if 'Interval' in ylabel and ymax-ymin<.01: ymin,ymax=.9,1.1
    for i in range(5):
        xpix=left+(right-left)*i/4; ypix=bottom-(bottom-top)*i/4
        draw.line((left,ypix,right,ypix),fill='#dddddd')
        draw.text((left-60,ypix-8),f'{ymin+(ymax-ymin)*i/4:.3g}',fill='black',font=small)
        draw.text((xpix-12,bottom+6),f'{xmin+(xmax-xmin)*i/4:.3g}',fill='black',font=small)
    draw.rectangle(rect,outline='black')
    pts=list(zip(left+(xx-xmin)/(xmax-xmin)*(right-left),bottom-(yy-ymin)/(ymax-ymin)*(bottom-top)))
    draw.line(pts,fill=color,width=1)
    draw.text(((left+right)//2-50,bottom+28),xlabel,fill='black',font=small)
    draw.text((left,top-49),ylabel,fill='black',font=small)

selected=['S001_REC01', df.loc[df.power_fraction_below20.idxmax(),'recording_id'], df.loc[df.voltage_max.idxmax(),'recording_id'], df.loc[df.fs_effective_hz.idxmin(),'recording_id']]
selected=list(dict.fromkeys(selected))
for rid in selected:
    t,x,a,f,p=traces[rid]; row=df.set_index('recording_id').loc[rid]
    im=Image.new('RGB',(1500,980),'white'); d=ImageDraw.Draw(im)
    d.text((65,15),f'{rid} | leg {row.leg} | token {row.source_token} | {len(t):,} samples | {t[-1]-t[0]:.3f} s',fill='black',font=titlefont)
    d.text((65,48),'Restricted local audit. Voltage units follow CSV header; calibration unverified. No fatigue labels.',fill='black',font=font)
    # Min/max envelope per plotting bin retains spikes over the entire file.
    bins=np.array_split(np.arange(len(x)),1800)
    ti=[]; xi=[]
    for b in bins:
        if len(b):
            inds=sorted([b[np.argmin(x[b])],b[np.argmax(x[b])]])
            ti.extend(t[inds]);xi.extend(x[inds])
    panel(d,(90,145,720,410),ti,xi,'Full recording (min/max bins)','Relative time (s)','Voltage_mV')
    center=int(np.argmax(np.abs(x))) if row.power_fraction_below20>.1 or row.voltage_max>1000 else len(t)//2
    n=min(round(row.fs_median_hz*.25),len(t)); mid=min(max(0,center-n//2),len(t)-n)
    excerpt_title='250 ms around largest absolute peak' if center!=len(t)//2 else 'Representative 250 ms excerpt'
    panel(d,(850,145,1460,410),t[mid:mid+n]-t[mid],x[mid:mid+n],excerpt_title,'Time from excerpt start (s)','Voltage_mV')
    # Min/max binning includes every anomalous interval and suppresses floating-point jitter.
    dt=np.round(np.diff(t)*1000,7); tt=t[1:]; tdi=[]; ddi=[]
    for b in np.array_split(np.arange(len(dt)),1800):
        inds=sorted([b[np.argmin(dt[b])],b[np.argmax(dt[b])]])
        tdi.extend(tt[inds]); ddi.extend(dt[inds])
    panel(d,(90,570,720,850),tdi,ddi,'Timestamp intervals (min/max bins)','Relative time (s)','Interval (ms)')
    panel(d,(850,570,1460,850),f,p,'Welch spectrum (timing assumption)','Frequency (Hz)','PSD (dB relative to header units squared/Hz)',logy=True)
    d.text((80,935),f'Median fs {row.fs_median_hz:.3f} Hz; elapsed-time fs {row.fs_effective_hz:.3f} Hz; longest constant run {row.longest_constant_samples} samples.',fill='black',font=font)
    im.save(OUT/f'{rid}_audit.png')
print('PLOTS',selected)
