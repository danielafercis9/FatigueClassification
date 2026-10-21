"""Exploratory participant-level stage comparisons, not predictive feature selection.

Run directly or through Feature_Statistics.ipynb. Stages are elapsed-time thirds,
not repetition-count thirds or independently measured physiological fatigue.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import warnings

import numpy as np
import pandas as pd
import scipy
from scipy import stats
import matplotlib
matplotlib.use('Agg')  # Figures are exported and displayed as saved images in the notebook.
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent
SCENARIOS={
    'reviewed':'Reviewed active interval; main exploratory comparison',
    'full_recording':'Same reviewed sets, entire CSV interval including quiet lead-in/out',
    'reviewed_shrink_250ms':'Reviewed start +250 ms and end -250 ms; boundary sensitivity',
    'reviewed_flagged_omitted':'Reviewed interval, omit provisional QC-flagged windows; stress test only',
}


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_run(run_dir=None):
    if run_dir is None:
        latest=json.loads((ROOT/'processing/outputs/latest_run.json').read_text())
        run_dir=ROOT/latest['path'].replace('\\','/')
    run_dir=Path(run_dir).resolve()
    if not run_dir.is_relative_to((ROOT/'processing/outputs').resolve()):
        raise ValueError('Input must be a local processing run directory.')
    manifest=json.loads((run_dir/'run_manifest.json').read_text())
    annotations=json.loads((ROOT/'processing/annotations.json').read_text())
    if manifest['annotations']!=annotations:
        raise ValueError('Saved boundary reviews differ from this processing run. Rerun processing/pipeline.py run first.')
    if manifest['status']['reviewed_recordings']==0:
        raise ValueError('No reviewed intervals in this processing run. Review boundaries and rerun preprocessing.')
    if not manifest['status']['all_original_hashes_unchanged']:
        raise ValueError('Processing did not confirm unchanged source recordings.')
    window_path=run_dir/'window_features.csv'
    w=pd.read_csv(window_path)
    # Use the persisted reviewed set decisions, even in full-recording sensitivity.
    decisions=annotations['recordings']
    reviewed={k for k,v in decisions.items() if v['status']=='reviewed'}
    w=w[w.recording_id.isin(reviewed)].copy()
    for rid,group in w.groupby('recording_id'):
        a=decisions[rid]
        if not np.allclose(group.bounds_start_s,a['start_s']) or not np.allclose(group.bounds_end_s,a['end_s']):
            raise ValueError(f'{rid}: feature-table and reviewed bounds differ.')
    dictionary=pd.read_csv(run_dir/'feature_dictionary.csv')
    features=dictionary.feature.tolist()
    if len(features)!=36 or not w.window_id.is_unique: raise ValueError('Feature dictionary/window IDs failed validation.')
    for flag in ['signal_eligible','review_possible_motion_or_drift','review_possible_transient']:
        if w[flag].dtype!=bool: raise ValueError(f'{flag} must parse as Boolean, not string truthiness.')
    inventory=pd.read_csv(ROOT/'audit/recording_inventory.csv').set_index('recording_id')
    return run_dir,manifest,w,dictionary,inventory


def stage_windows(w, inventory, scenario):
    if scenario not in SCENARIOS: raise ValueError('Unknown scenario.')
    a=w.bounds_start_s.copy(); b=w.bounds_end_s.copy()
    if scenario=='full_recording':
        a=w.recording_id.map(inventory.t_start_s)
        b=w.recording_id.map(inventory.t_end_s)+1/w.fs_used_hz
    elif scenario=='reviewed_shrink_250ms':
        a=a+.25; b=b-.25
    duration=b-a
    if (duration<=0).any(): raise ValueError('Nonpositive active interval.')
    early_end=a+duration/3; late_start=a+2*duration/3
    early=(w.t_start_s>=a-1e-9)&(w.t_end_exclusive_s<=early_end+1e-9)
    late=(w.t_start_s>=late_start-1e-9)&(w.t_end_exclusive_s<=b+1e-9)
    # A window straddling a third boundary belongs to neither stage.
    stage=np.where(early,'early',np.where(late,'late','not_compared'))
    eligible=w.signal_eligible.copy()
    if scenario=='reviewed_flagged_omitted':
        eligible &= ~w.review_possible_motion_or_drift & ~w.review_possible_transient
    result=w[['window_id','subject_id','recording_id','leg','set_token','t_start_s','t_end_exclusive_s']].copy()
    result['scenario']=scenario; result['stage']=stage
    result['stage_eligible']=eligible & (stage!='not_compared')
    result['interval_start_s']=a;result['interval_end_s']=b
    result['early_end_s']=early_end;result['late_start_s']=late_start
    return result


def paired_summaries(w, assignments, features, config):
    joined=w[['window_id']+features].merge(assignments,on='window_id',validate='one_to_one')
    used=joined[joined.stage_eligible]
    ids=['subject_id','leg','recording_id','set_token','scenario']
    med=used.groupby(ids+['stage'],observed=True)[features].median()
    counts=used.groupby(ids+['stage'],observed=True)[features].count()
    rows=[]
    # Retain every reviewed set-feature combination, including failed pairs.
    for values,group in joined.groupby(ids,observed=True):
        meta=dict(zip(ids,values))
        for feature in features:
            pair={};count={}
            for stage in ['early','late']:
                key=values+(stage,)
                pair[stage]=float(med.loc[key,feature]) if key in med.index else np.nan
                count[stage]=int(counts.loc[key,feature]) if key in counts.index else 0
            complete=all(count[s]>=config['min_windows_per_stage'] and np.isfinite(pair[s]) for s in ['early','late'])
            rows.append(dict(**meta,feature=feature,early=pair['early'],late=pair['late'],
                n_early_windows=count['early'],n_late_windows=count['late'],paired_set=complete,
                delta=pair['late']-pair['early'] if complete else np.nan,
                pairing_reason='included' if complete else 'insufficient_defined_early_or_late_windows'))
    sets=pd.DataFrame(rows)
    good=sets[sets.paired_set]
    legs=good.groupby(['scenario','subject_id','leg','feature'],observed=True).agg(
        early=('early','mean'),late=('late','mean'),delta=('delta','mean'),n_sets=('recording_id','nunique')).reset_index()
    subjects=legs.groupby(['scenario','subject_id','feature'],observed=True).agg(
        early=('early','mean'),late=('late','mean'),delta=('delta','mean'),n_legs=('leg','nunique'),n_sets=('n_sets','sum')).reset_index()
    subjects['included']=subjects.n_legs.eq(2) if config['require_both_legs'] else True
    subjects['reason']=np.where(subjects.included,'included','both_legs_required')
    return sets,legs,subjects


def adjustment(pvalues, method):
    p=np.asarray(pvalues,float); answer=np.full(len(p),np.nan);ok=np.isfinite(p)
    if ok.any(): answer[ok]=stats.false_discovery_control(p[ok],method=method)
    return answer


def comparison_table(subjects,dictionary,config):
    rows=[]
    for scenario in config['scenarios']:
        for _,feature_row in dictionary.iterrows():
            name=feature_row.feature
            g=subjects[(subjects.scenario==scenario)&(subjects.feature==name)&subjects.included].sort_values('subject_id')
            d=g.delta.to_numpy();d=d[np.isfinite(d)]; n=len(d)
            tol=config['zero_tolerance']; positive=int((d>tol).sum());negative=int((d<-tol).sum());zeros=n-positive-negative
            nonzero=positive+negative
            test=stats.binomtest(positive,nonzero,.5,alternative='two-sided') if nonzero else None
            probability_ci=test.proportion_ci(config['confidence_level'],method='exact') if test else None
            ci_low=ci_high=np.nan;ci_status='insufficient_subjects';mean_change=float(d.mean()) if n else np.nan
            # Same seed for an identical scenario/cohort across features preserves paired resample indices.
            key=scenario+'|'+ '|'.join(g.subject_id)
            seed=(config['seed']+int(hashlib.sha256(key.encode()).hexdigest()[:8],16))%(2**32)
            if n>=3 and np.ptp(d)>tol:
                with warnings.catch_warnings(record=True) as captured:
                    bootstrap=stats.bootstrap((d,),np.mean,vectorized=True,method='BCa',
                        n_resamples=config['bootstrap_resamples'],confidence_level=config['confidence_level'],rng=np.random.default_rng(seed))
                ci_low=float(bootstrap.confidence_interval.low);ci_high=float(bootstrap.confidence_interval.high)
                ci_status='BCa_participant_bootstrap' if np.isfinite([ci_low,ci_high]).all() else 'undefined_BCa'
            elif n>=3:
                ci_low=ci_high=mean_change;ci_status='degenerate_observed_differences_no_uncertainty_estimate'
            sd=np.std(d,ddof=1) if n>=2 else np.nan
            rows.append(dict(scenario=scenario,feature=name,family=feature_row.family,units=feature_row.units,
                n_subjects=n,n_positive=positive,n_negative=negative,n_zero=zeros,n_sign_test=nonzero,
                mean_early=float(g.early.mean()) if n else np.nan,mean_late=float(g.late.mean()) if n else np.nan,
                mean_delta=mean_change,median_delta=float(np.median(d)) if n else np.nan,
                mean_delta_ci_low=ci_low,mean_delta_ci_high=ci_high,ci_method=ci_status,
                standardized_mean_delta=mean_change/sd if sd>0 else np.nan,
                proportion_positive=positive/nonzero if nonzero else np.nan,
                positive_fraction_ci_low=probability_ci.low if test else np.nan,
                positive_fraction_ci_high=probability_ci.high if test else np.nan,
                p_sign_two_sided=float(test.pvalue) if test else (1. if n else np.nan),
                bootstrap_seed=seed,test='exact_paired_sign_test_nonzero_participant_differences'))
    table=pd.DataFrame(rows)
    for _,idx in table.groupby('scenario',observed=True).groups.items():
        table.loc[idx,'q_by_36']=adjustment(table.loc[idx,'p_sign_two_sided'],'by')
        table.loc[idx,'q_bh_36']=adjustment(table.loc[idx,'p_sign_two_sided'],'bh')
    table['q_by_all_scenarios']=adjustment(table.p_sign_two_sided,'by')
    table['q_bh_all_scenarios']=adjustment(table.p_sign_two_sided,'bh')
    table['direction_evidence_by_36']=table.q_by_36.le(config['alpha'])
    return table


def redundancy_table(subjects,config):
    main=subjects[(subjects.scenario=='reviewed')&subjects.included]
    wide=main.pivot(index='subject_id',columns='feature',values='delta')
    rho=wide.corr(method='spearman',min_periods=5)
    pairs=[]
    for i,a in enumerate(rho.columns):
        for b in rho.columns[i+1:]:
            value=rho.loc[a,b];n=int(wide[[a,b]].dropna().shape[0])
            if np.isfinite(value) and abs(value)>=config['redundancy_abs_rho']:
                pairs.append(dict(feature_a=a,feature_b=b,rho_subject_delta=float(value),n_paired_subjects=n))
    return rho,pd.DataFrame(pairs,columns=['feature_a','feature_b','rho_subject_delta','n_paired_subjects'])


def evidence_worksheet(table,pairs):
    main=table[table.scenario=='reviewed'].copy()
    main['required_baseline_feature']=main.feature.isin(['RMS','MDF'])
    partners={name:[] for name in main.feature}
    for _,row in pairs.iterrows():
        partners[row.feature_a].append(row.feature_b);partners[row.feature_b].append(row.feature_a)
    main['correlated_change_partners']=main.feature.map(lambda f:'|'.join(partners[f]))
    main['known_redundancy']=main.feature.map({'IEMG':'MAV * 0.5 s (fixed window duration)',
        'MAV':'IEMG / 0.5 s (fixed window duration)','Hjorth_activity':'Exactly equals Variance','Variance':'Exactly equals Hjorth_activity'}).fillna('')
    direction=table.pivot(index='feature',columns='scenario',values='mean_delta').apply(np.sign)
    main['mean_direction_same_all_scenarios']=main.feature.map(lambda f:bool(direction.loc[f].notna().all() and direction.loc[f].nunique()==1))
    main['model_selection_status']='not_selected_by_this_analysis'
    main['physiological_rationale_to_write']=''
    main['planned_training_only_selection_or_ablation']=''
    return main


def matched_sensitivity(subjects,config):
    rows=[]
    for feature,group in subjects[subjects.included].groupby('feature',observed=True):
        wide=group.pivot(index='subject_id',columns='scenario',values='delta').reindex(columns=config['scenarios']).dropna()
        for scenario in config['scenarios']:
            rows.append(dict(feature=feature,scenario=scenario,n_common_subjects=len(wide),
                mean_delta_same_subjects=float(wide[scenario].mean()) if len(wide) else np.nan,
                mean_effect_change_vs_reviewed=float((wide[scenario]-wide['reviewed']).mean()) if len(wide) else np.nan,
                sign_agreement_with_reviewed=float((np.sign(wide[scenario])==np.sign(wide['reviewed'])).mean()) if len(wide) else np.nan))
    return pd.DataFrame(rows)


def render_plots(subjects,table,rho,config,out):
    folder=out/'figures';folder.mkdir(exist_ok=True)
    main=subjects[(subjects.scenario=='reviewed')&subjects.included]
    names=config['plot_features']
    fig,axes=plt.subplots(1,len(names),figsize=(5*len(names),4),squeeze=False)
    for ax,name in zip(axes[0],names):
        group=main[main.feature==name]
        for _,r in group.iterrows(): ax.plot([0,1],[r.early,r.late],color='#7e949e',alpha=.6,marker='o',ms=3,lw=.8)
        ax.plot([0,1],[group.early.mean(),group.late.mean()],color='#126782',marker='o',lw=2.5,label='Participant mean')
        ax.set_xticks([0,1],['Early third','Late third']);ax.set_title(f'{name} | n={len(group)} participants')
        ax.set_ylabel(table.loc[table.feature.eq(name),'units'].iloc[0]);ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Reviewed intervals: paired participant summaries; elapsed-time stage proxies')
    fig.tight_layout();fig.savefig(folder/'paired_early_late.png',dpi=160);plt.close(fig)
    # Every candidate has an individual participant-difference plot, with sign-test q and mean CI.
    for _,r in table[table.scenario=='reviewed'].iterrows():
        group=main[main.feature==r.feature].sort_values('subject_id'); fig,ax=plt.subplots(figsize=(8,4))
        ax.scatter(group.subject_id,group.delta,color='#126782');ax.axhline(0,color='black',lw=.8)
        ax.axhline(r.mean_delta,color='#b34c16',label='Mean change')
        if np.isfinite([r.mean_delta_ci_low,r.mean_delta_ci_high]).all():
            ax.axhspan(r.mean_delta_ci_low,r.mean_delta_ci_high,color='#b34c16',alpha=.15,label='Pointwise 95% BCa CI of mean')
        ax.set(title=f'{r.feature}: late minus early | sign-test BY q={r.q_by_36:.4g}',ylabel=r.units,xlabel='Independent participant ID')
        ax.tick_params(axis='x',rotation=45);ax.grid(alpha=.2);ax.legend(fontsize=8);fig.tight_layout()
        fig.savefig(folder/f'delta_{r.feature}.png',dpi=130);plt.close(fig)
    fig,ax=plt.subplots(figsize=(15,13)); im=ax.imshow(rho,vmin=-1,vmax=1,cmap='RdBu_r')
    ax.set_xticks(range(len(rho)),rho.columns,rotation=90,fontsize=7);ax.set_yticks(range(len(rho)),rho.index,fontsize=7)
    ax.set_title('Redundancy of participant-level changes (Spearman; descriptive)')
    fig.colorbar(im,ax=ax,label='Correlation of participant changes');fig.tight_layout();fig.savefig(folder/'redundancy_subject_changes.png',dpi=140);plt.close(fig)
    # Assumption/outlier inspection without selecting a test based on a normality p-value.
    fig,axes=plt.subplots(2,len(names),figsize=(5*len(names),7),squeeze=False)
    for col,name in enumerate(names):
        d=main.loc[main.feature.eq(name),'delta'].to_numpy()
        axes[0,col].hist(d,bins='auto',color='#126782',alpha=.7);axes[0,col].axvline(0,color='black');axes[0,col].set_title(name+' difference distribution')
        if len(d)>=3 and np.ptp(d)>0: stats.probplot(d,dist='norm',plot=axes[1,col])
        axes[1,col].set_title(name+' Q-Q inspection (not a test-selection gate)')
    fig.tight_layout();fig.savefig(folder/'difference_diagnostics.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(1,len(names),figsize=(5*len(names),4),squeeze=False)
    for ax,name in zip(axes[0],names):
        g=table[table.feature.eq(name)]
        ax.hlines(range(len(g)),g.mean_delta_ci_low,g.mean_delta_ci_high,color='#126782')
        ax.scatter(g.mean_delta,range(len(g)),color='#126782')
        ax.set_yticks(range(len(g)),g.scenario,fontsize=8);ax.axvline(0,color='black',lw=.8);ax.set(title=name,xlabel='Mean participant change (late - early)');ax.grid(alpha=.2)
    fig.suptitle('Processing sensitivity; pointwise 95% BCa intervals, not model performance')
    fig.tight_layout();fig.savefig(folder/'boundary_qc_sensitivity.png',dpi=140);plt.close(fig)


def write_notes(table,manifest,config,out):
    main=table[table.scenario=='reviewed']; selected=main[main.feature.isin(config['plot_features'])]
    status=manifest['status']
    lines=[
        'Exploratory feature-statistics methods and result notes',
        'This is a supporting analysis record, not a completed final report or a model feature-selection result.',
        f'Input processing run: {manifest["run_id"]}. Reviewed sets: {status["reviewed_recordings"]}; excluded sets: {status["review_excluded_recordings"]}; pending: {status["pending_recordings"]}.',
        'Operational stages: first and last thirds of elapsed active-set time; middle third and boundary-straddling windows omitted. These are not repetition-count thirds or independent fatigue ground truth.',
        f'At least {config["min_windows_per_stage"]} defined, signal-eligible windows per set/feature/stage are required. Median within stage/set, paired-set differences, equal means across sets within leg and across legs within participant. Both legs required for the main comparison.',
        'Independent units are participant differences. Sets, legs and overlapping windows are not counted as independent participants.',
        'Primary two-sided exact paired sign test: among nonzero participant differences, positive and negative changes are equally probable under the null. Zeros within the configured tolerance are omitted from its denominator. Magnitudes do not enter this test.',
        'Effect summaries include mean and median participant change, standardized mean change, sign counts and exact binomial intervals for the positive fraction.',
        f'Mean-change intervals: pointwise {100*config["confidence_level"]:.0f}% BCa bootstrap, {config["bootstrap_resamples"]} resamples of whole participant summaries with paired early/late correspondence, fixed seed {config["seed"]}. These intervals have not been adjusted for multiple comparisons and target mean change, whereas the sign test targets directional balance.',
        'Main multiplicity control: Benjamini-Yekutieli across all 36 candidate sign tests; valid under arbitrary p-value dependence when individual test assumptions hold. BH values are additionally shown, with its dependence assumptions. Sensitivity-scenario p-values also receive all-scenario corrections. No significance-based list of predictors is generated.',
        'All comparisons are exploratory: the data/trajectories have already been inspected. Small convenience-sample confidence intervals do not establish external validity.',
        'Sensitivity scenarios retain the same reviewed set decisions. Full-recording bounds deliberately include lead-in/out; 250 ms inward shifts test small annotation changes; omission of provisional QC flags is a stress test, not a validated artifact-exclusion rule. Cohort/paired-set counts are exported and can change by scenario/feature.',
        'Redundancy is descriptive Spearman correlation across independent participant changes, not pooled overlapping windows. No significance tests of correlation are performed.',
        'Do not interpret stage differences as proof of peripheral fatigue. Contraction phase, quiet signal, artifacts, subject gain and prior-set fatigue remain alternative explanations. MVC normalization, force, Borg, video and rest metadata are unavailable.',
        'The required RMS+MDF baseline is retained regardless of these p-values. To evaluate added features, freeze a selection algorithm and apply it inside training-only grouped inner validation; do not threshold full-cohort p-values and then claim unbiased LOSO evaluation.',
        '', 'Illustrative feature results (all 36 candidates remain in the exported tables):']
    for _,r in selected.iterrows():
        lines.append(f'{r.feature}: n={r.n_subjects}, mean late-early={r.mean_delta:.6g} {r.units}, pointwise 95% BCa CI [{r.mean_delta_ci_low:.6g}, {r.mean_delta_ci_high:.6g}], positive/negative/zero={r.n_positive}/{r.n_negative}/{r.n_zero}, sign p={r.p_sign_two_sided:.6g}, BY q across 36={r.q_by_36:.6g}.')
    lines += ['', 'Methods references:',
        'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html',
        'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html',
        'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.false_discovery_control.html',
        'https://scikit-learn.org/stable/common_pitfalls.html']
    (out/'report_notes.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def run_analysis(run_dir=None,config=None,make_plots=True):
    config=config or json.loads((HERE/'config.json').read_text())
    if 'reviewed' not in config['scenarios'] or any(s not in SCENARIOS for s in config['scenarios']): raise ValueError('Invalid scenario configuration.')
    if config['bootstrap_resamples']<1000: raise ValueError('Use at least 1000 resamples for the analysis.')
    source,manifest,w,dictionary,inventory=source_run(run_dir)
    fingerprint=dict(config=config,processing_run_id=manifest['run_id'],feature_file_sha256=sha(source/'window_features.csv'),
                     processing_manifest_sha256=sha(source/'run_manifest.json'),statistics_code_sha256=sha(__file__))
    run_id=hashlib.sha256(json.dumps(fingerprint,sort_keys=True).encode()).hexdigest()[:12]
    out=HERE/'outputs'/run_id;out.mkdir(parents=True,exist_ok=True)
    all_sets=[];all_legs=[];all_subjects=[];all_stages=[]
    for scenario in config['scenarios']:
        stages=stage_windows(w,inventory,scenario)
        sets,legs,subjects=paired_summaries(w,stages,dictionary.feature.tolist(),config)
        all_sets.append(sets);all_legs.append(legs);all_subjects.append(subjects);all_stages.append(stages)
        print(f'{scenario}: {int(sets.paired_set.sum())} paired set-feature rows; {subjects.loc[subjects.included,"subject_id"].nunique()} independent participant IDs.',flush=True)
    sets=pd.concat(all_sets,ignore_index=True);legs=pd.concat(all_legs,ignore_index=True)
    subjects=pd.concat(all_subjects,ignore_index=True);stages=pd.concat(all_stages,ignore_index=True)
    table=comparison_table(subjects,dictionary,config)
    rho,pairs=redundancy_table(subjects,config);worksheet=evidence_worksheet(table,pairs)
    sensitivity=matched_sensitivity(subjects,config)
    outputs={'stage_window_assignments':stages,'paired_sets':sets,'paired_subject_legs':legs,'paired_subjects':subjects,
             'all_scenario_comparisons':table,'main_feature_comparisons':table[table.scenario.eq('reviewed')],
             'redundant_change_pairs':pairs,'feature_evidence_worksheet':worksheet,'matched_cohort_sensitivity':sensitivity}
    for name,frame in outputs.items(): frame.to_csv(out/f'{name}.csv',index=False)
    rho.to_csv(out/'subject_change_correlations.csv')
    if make_plots: render_plots(subjects,table,rho,config,out)
    write_notes(table,manifest,config,out)
    metadata=dict(run_id=run_id,created_utc=datetime.now(timezone.utc).isoformat(),**fingerprint,
                  source_reviewed_sets=manifest['status']['reviewed_recordings'],source_excluded_sets=manifest['status']['review_excluded_recordings'],
                  source_pending_sets=manifest['status']['pending_recordings'],feature_tests_main=len(dictionary),
                  versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,matplotlib=matplotlib.__version__),
                  scope='Exploratory elapsed-time stage association; no feature selection, labels for modelling, PCA or classifiers.')
    (out/'analysis_manifest.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    (HERE/'outputs/latest_analysis.json').write_text(json.dumps(dict(run_id=run_id,path=str(out.relative_to(ROOT))),indent=2),encoding='utf-8')
    print(f'Saved statistics/outputs/{run_id}',flush=True)
    return dict(output_dir=out,main=outputs['main_feature_comparisons'],comparisons=table,sets=sets,legs=legs,
                subjects=subjects,stages=stages,correlations=rho,redundant_pairs=pairs,worksheet=worksheet,sensitivity=sensitivity,metadata=metadata)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir');args=parser.parse_args()
    run_analysis(args.run_dir)
