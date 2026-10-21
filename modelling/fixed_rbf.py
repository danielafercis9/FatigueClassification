"""Fixed RBF SVM with outer-training significance filtering; no inner search."""
from datetime import datetime,timezone
import json
from pathlib import Path
import platform
import time

import joblib
import numpy as np
import pandas as pd
import scipy
from scipy import stats
import sklearn
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
from modelling.core import (load_data,sha,metrics,summary_table,_figures,
    _selection_artifacts,_fit_with_warnings)

FIXED_RBF_CONFIG=dict(seed=20261018,C=1.,gamma='scale',alpha=.05,
    correction='BY',redundancy_abs_rho=.90,min_windows_per_stage=3,
    min_subjects_per_feature=3,require_both_legs=True,zero_tolerance=1e-12,
    cache_mb=128,tol=.001,max_iter=-1,final_refit=True,n_jobs=1,
    outer_validation='LOSO',inner_validation='none_fixed_configuration')
MODEL_KEY='svm_rbf_fixed_significant'


class SignificantFeatureSelector(TransformerMixin,BaseEstimator):
    def __init__(self,features,alpha=.05,redundancy_abs_rho=.9,
                 min_windows_per_stage=3,min_subjects_per_feature=3,
                 require_both_legs=True,zero_tolerance=1e-12):
        self.features=features;self.alpha=alpha;self.redundancy_abs_rho=redundancy_abs_rho
        self.min_windows_per_stage=min_windows_per_stage
        self.min_subjects_per_feature=min_subjects_per_feature
        self.require_both_legs=require_both_legs;self.zero_tolerance=zero_tolerance

    def fit(self,X,y):
        features=list(self.features)
        if len(features)!=36:raise ValueError('BY family is the audited 36 candidates.')
        self.training_subjects_=sorted(X.subject_id.unique().tolist())
        z=X[['subject_id','leg','recording_id']+features].reset_index(drop=True).copy()
        z[features]=z[features].replace([np.inf,-np.inf],np.nan);z['_stage']=np.asarray(y,dtype=int)
        group=z.groupby(['subject_id','leg','recording_id','_stage'],observed=True)[features]
        med=group.median();counts=group.count()
        early=med.xs(0,level='_stage');late=med.xs(1,level='_stage')
        enough=(counts.xs(0,level='_stage')>=self.min_windows_per_stage)&(
            counts.xs(1,level='_stage')>=self.min_windows_per_stage)
        changes=(late-early).where(enough)
        legs=changes.groupby(level=['subject_id','leg']).mean()
        subjects=legs.groupby(level='subject_id').mean()
        if self.require_both_legs:
            subjects=subjects.where(legs.notna().groupby(level='subject_id').sum()==2)
        self.subject_changes_=subjects
        rows=[]
        for order,f in enumerate(features):
            d=subjects[f].dropna();positive=int((d>self.zero_tolerance).sum())
            negative=int((d<-self.zero_tolerance).sum());nonzero=positive+negative
            sufficient=len(d)>=self.min_subjects_per_feature
            p=float(stats.binomtest(positive,nonzero,.5,alternative='two-sided').pvalue) if sufficient and nonzero else 1.
            sd=float(d.std(ddof=1));mean=float(d.mean())
            effect=abs(mean/sd) if np.isfinite(sd) and sd>self.zero_tolerance else np.nan
            rows.append(dict(feature=f,dictionary_order=order,n_subjects=len(d),
                n_positive=positive,n_negative=negative,n_zero=len(d)-nonzero,
                n_sign_test=nonzero,mean_change=mean,sd_change=sd,abs_paired_dz=effect,
                p_sign_two_sided=p,selected=False,passes_significance=False,
                reason='not_significant',redundant_with='',max_abs_rho=np.nan,undefined_correlations=0))
        rank=pd.DataFrame(rows)
        # Keep all 36 hypotheses in the correction; unavailable comparisons get p=1.
        rank['q_by_36']=stats.false_discovery_control(rank.p_sign_two_sided.to_numpy(),method='by')
        rank['passes_significance']=rank.q_by_36.lt(self.alpha)
        rank=rank.sort_values(['q_by_36','abs_paired_dz','dictionary_order'],
            ascending=[True,False,True],na_position='last').reset_index(drop=True)
        selected=[]
        for i,row in rank.iterrows():
            if not row.passes_significance:continue
            redundant='';maximum=np.nan;undefined=0
            for previous in selected:
                pair=subjects[[row.feature,previous]].dropna()
                if len(pair)<self.min_subjects_per_feature or (pair.nunique()<2).any():
                    undefined+=1;continue
                rho=abs(float(pair.corr(method='spearman').iloc[0,1]))
                maximum=rho if np.isnan(maximum) else max(maximum,rho)
                if rho>=self.redundancy_abs_rho:redundant=previous;break
            rank.loc[i,['redundant_with','max_abs_rho','undefined_correlations']]=[redundant,maximum,undefined]
            if redundant:rank.loc[i,'reason']='redundant_training_subject_changes'
            elif undefined:rank.loc[i,'reason']='redundancy_unverifiable'
            else:
                selected.append(row.feature);rank.loc[i,['selected','reason']]=[True,'significant_nonredundant']
        self.selected_=selected;self.ranking_=rank
        if not selected:
            raise ValueError('No features survived this training-only significance/redundancy rule. '
                'Run is incomplete; inspect feature_selection.csv. No nonsignificant fallback is used.')
        return self

    def transform(self,X):return X[self.selected_].replace([np.inf,-np.inf],np.nan).copy()
    def get_feature_names_out(self,input_features=None):return np.asarray(self.selected_,dtype=object)


def make_pipeline(features,cfg):
    return Pipeline([
        ('select',SignificantFeatureSelector(tuple(features),alpha=cfg['alpha'],
            redundancy_abs_rho=cfg['redundancy_abs_rho'],
            min_windows_per_stage=cfg['min_windows_per_stage'],
            min_subjects_per_feature=cfg['min_subjects_per_feature'],
            require_both_legs=cfg['require_both_legs'],zero_tolerance=cfg['zero_tolerance'])),
        ('impute',SimpleImputer(strategy='median',keep_empty_features=True)),
        ('scale',StandardScaler()),
        ('model',SVC(kernel='rbf',C=cfg['C'],gamma=cfg['gamma'],probability=False,
            cache_size=cfg['cache_mb'],tol=cfg['tol'],max_iter=cfg['max_iter'],
            random_state=cfg['seed']))])


def _json(path,value):Path(path).write_text(json.dumps(value,indent=2,default=str)+'\n',encoding='utf-8')


def _fit(model,X,y,directory):
    try:return _fit_with_warnings(model,X,y)
    except Exception:
        selector=model.named_steps['select']
        if hasattr(selector,'ranking_'):selector.ranking_.to_csv(directory/'feature_selection.csv',index=False)
        raise


def _fixed_diagram(out,n):
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(12,3.5),layout='constrained');ax.axis('off')
    boxes=[(.22,.57,f'{n-1} training participants\nSignificance + redundancy filter\nFit imputation and scaling\nFit one RBF SVM: C=1, gamma=scale'),
        (.67,.57,'One held-out participant\nFrozen pipeline\nEvaluate once'),
        (.92,.57,f'Repeat for\n{n} subjects')]
    for x,y,label in boxes:ax.text(x,y,label,ha='center',va='center',transform=ax.transAxes,
        bbox=dict(boxstyle='round',facecolor='#e8f1fa'),fontsize=10)
    for a,b in [(.43,.51),(.8,.85)]:ax.annotate('',xy=(b,.57),xytext=(a,.57),
        xycoords='axes fraction',arrowprops=dict(arrowstyle='->'))
    ax.set_title('LOSO — one fixed configuration; no inner splits, grid search or threshold tuning')
    fig.savefig(out/'figures/validation_diagram.png',dpi=170);plt.close(fig)


def run_fixed_rbf(data_dir,output_root,config=None):
    cfg={**FIXED_RBF_CONFIG,**(config or {})}
    if cfg['correction']!='BY' or cfg['n_jobs']!=1 or cfg['outer_validation']!='LOSO':
        raise ValueError('This experiment fixes BY, serial execution and LOSO.')
    w,dictionary,source=load_data(data_dir);features=dictionary.feature.tolist()
    X=w[['subject_id','recording_id','leg']+features].copy()
    y=w.Label.to_numpy(dtype=int);groups=w.subject_id.to_numpy()
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out=Path(output_root)/MODEL_KEY/stamp;out.mkdir(parents=True,exist_ok=False)
    manifest=dict(status='running',model_key=MODEL_KEY,created_utc=stamp,config=cfg,
        data_manifest=source,code_sha256={p.name:sha(p) for p in [Path(__file__),Path(__file__).parent/'core.py']},
        versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,
            scipy=scipy.__version__,sklearn=sklearn.__version__,joblib=joblib.__version__),
        validation='LOSO; no inner folds or hyperparameter search; feature selection fitted on outer-training subjects only',
        score='Uncalibrated SVC decision_function; class labels use default SVC.predict; no threshold tuning',
        limitations=['Fixed pragmatic C/gamma, not optimized settings.',
            'Exploratory extension proposed after earlier model results were inspected.',
            'Both search strategy and feature-selection rule differ from notebook 05.',
            'Significance indicates consistent stage changes, not physiological specificity or guaranteed prediction.',
            'Participant-change redundancy filtering does not guarantee low window-level predictor dependence.'])
    _json(out/'run_manifest.json',manifest)
    folds=[];predictions=[];classes=[];legs=[];selections=[];splits=[];started=time.perf_counter()
    try:
        with threadpool_limits(limits=1):
            print('Fixed RBF SVM: one CPU fit per held-out participant, no inner search.',flush=True)
            for fold,(train,test) in enumerate(LeaveOneGroupOut().split(X,y,groups),start=1):
                start=time.perf_counter();subject=groups[test][0];training_subjects=sorted(set(groups[train]))
                assert subject not in training_subjects
                directory=out/f'fold_{fold:02d}_{subject}';directory.mkdir()
                model=make_pipeline(features,cfg)
                print(f'Fold {fold}/{len(np.unique(groups))}: one fit; test {subject}.',flush=True)
                warns=_fit(model,X.iloc[train],y[train],directory)
                pred=model.predict(X.iloc[test]);scores=model.decision_function(X.iloc[test])
                result,pc=metrics(y[test],pred,scores)
                names=_selection_artifacts(model,X.iloc[train],y[train],features,directory)
                joblib.dump(model,directory/'pipeline.joblib');_json(directory/'fit_warnings.json',warns)
                split=dict(fold=fold,train_subjects=training_subjects,test_subjects=[subject],inner=[])
                _json(directory/'split.json',split);splits.append(split)
                significant=int(model.named_steps['select'].ranking_.passes_significance.sum())
                folds.append(dict(model=MODEL_KEY,fold=fold,test_subject=subject,n_train_subjects=len(training_subjects),
                    n_train_windows=len(train),n_test_windows=len(test),n_features=len(names),n_significant=significant,
                    selected_features='|'.join(names),C=cfg['C'],gamma=str(cfg['gamma']),
                    score_kind='decision_function',seconds=time.perf_counter()-start,**result))
                pc['fold']=fold;pc['subject_id']=subject;classes.append(pc)
                oo=w.iloc[test][['window_id','subject_id','recording_id','set_token','leg',
                    'sample_start','sample_end_exclusive','t_start_s','t_end_exclusive_s']].copy()
                oo['fold']=fold;oo['y_true']=y[test];oo['y_pred']=pred;oo['score']=scores;oo['score_kind']='decision_function'
                predictions.append(oo)
                for leg,g in oo.groupby('leg'):
                    values,_=metrics(g.y_true,g.y_pred,g.score)
                    legs.append(dict(fold=fold,subject_id=subject,leg=leg,**values))
                for name in names:selections.append(dict(fold=fold,subject_id=subject,feature=name))
                pd.DataFrame(folds).to_csv(out/'partial_results.csv',index=False)
                print(f'{subject}: macro-F1={result["f1_macro"]:.3f}; '
                    f'{len(names)} retained of {significant} significant; {time.perf_counter()-start:.1f} s',flush=True)
            fold_table=pd.DataFrame(folds);oof=pd.concat(predictions,ignore_index=True)
            if len(oof)!=len(w) or not oof.window_id.is_unique or set(oof.window_id)!=set(w.window_id):
                raise AssertionError('Out-of-fold prediction coverage mismatch.')
            fold_table.to_csv(out/'results.csv',index=False)
            fold_table.rename(columns={'test_subject':'subject_id'}).to_csv(out/'per_subject_metrics.csv',index=False)
            oof.to_csv(out/'out_of_fold_predictions.csv',index=False)
            pc=pd.concat(classes,ignore_index=True);pc.to_csv(out/'per_class_metrics.csv',index=False)
            pc.groupby('class_label')[['precision','recall','f1']].agg(['mean','std']).to_csv(out/'per_class_mean_sd.csv')
            leg_table=pd.DataFrame(legs);leg_table.to_csv(out/'per_subject_leg_metrics.csv',index=False)
            leg_table.pivot(index='subject_id',columns='leg',values='f1_macro').to_csv(out/'paired_leg_f1.csv')
            chosen=pd.DataFrame(selections).merge(dictionary[['feature','family']],on='feature',validate='many_to_one')
            chosen.to_csv(out/'selected_features_by_fold.csv',index=False)
            chosen.groupby(['feature','family']).size().rename('outer_folds_selected').sort_values(ascending=False).to_csv(out/'selection_frequency.csv')
            summary=summary_table(fold_table);summary.to_csv(out/'metrics_mean_sd.csv',index=False)
            pooled,pooled_pc=metrics(oof.y_true,oof.y_pred,np.zeros(len(oof)))
            for key in ['roc_auc','pr_auc','average_precision']:pooled.pop(key)
            _json(out/'pooled_window_metrics_secondary.json',pooled)
            pooled_pc.to_csv(out/'pooled_per_class_metrics.csv',index=False)
            _json(out/'subject_splits.json',splits)
            _figures(oof,fold_table,summary,out,'RBF SVM — fixed settings, training-only significant features')
            _fixed_diagram(out,len(np.unique(groups)))
            if cfg['final_refit']:
                final=out/'final_refit';final.mkdir();model=make_pipeline(features,cfg)
                print('One final all-subject fit; no additional test score.',flush=True)
                warns=_fit(model,X,y,final)
                names=_selection_artifacts(model,X,y,features,final)
                joblib.dump(model,final/'pipeline.joblib');_json(final/'fit_warnings.json',warns)
                _json(final/'model_card.json',dict(purpose='Final full-data fit; no new independent test score',
                    config=cfg,selected_features=names,source_model_data_sha256=source['data_sha256']))
            manifest.update(status='complete',n_outer_folds=len(folds),total_seconds=time.perf_counter()-started)
            _json(out/'run_manifest.json',manifest)
            _json(Path(output_root)/f'latest_{MODEL_KEY}.json',dict(path=str(out.resolve()),status='complete'))
    except Exception as exc:
        manifest.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        _json(out/'run_manifest.json',manifest);raise
    return out


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=Path(__file__).parent/'data')
    parser.add_argument('--outputs',type=Path,default=Path(__file__).parent/'outputs')
    args=parser.parse_args();print(run_fixed_rbf(args.data,args.outputs))
