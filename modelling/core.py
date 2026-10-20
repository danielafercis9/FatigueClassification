"""Offline nested subject validation. No training occurs on import.

X carries metadata only as far as the selector. Its transform returns feature
columns exclusively, before imputation, scaling and classification.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time
import warnings

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, auc,
    cohen_kappa_score, confusion_matrix, precision_recall_curve,
    precision_recall_fscore_support, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from threadpoolctl import threadpool_limits

MODEL_NAMES = {
    'majority': 'Majority-class baseline',
    'lr_rms_mdf': 'Logistic regression: RMS + MDF',
    'lr_selected': 'Logistic regression: training-selected features',
    'svm_selected': 'Linear SVM: training-selected features',
    'svm_rbf_selected': 'RBF SVM: expanded training-selected features',
}
ORIGINAL_MODELS = ('majority','lr_rms_mdf','lr_selected','svm_selected')
CLASSES = [0, 1]
CLASS_NAMES = ['Early / reference (0)', 'Late / proxy fatigue (1)']
METRICS = ['accuracy','precision_macro','recall_macro','f1_macro',
    'f1_class_0','f1_class_1','cohen_kappa','roc_auc','pr_auc','average_precision']
DEFAULTS = dict(seed=20261018, inner_folds=3, C=[.01,.1,1.,10.,100.],
    max_features=[2,3,5], redundancy_abs_rho=.90, min_windows_per_stage=3,
    min_subjects_per_feature=3, require_both_legs=True, max_iter=20000,
    final_refit=True, n_jobs=1, outer_validation='LOSO',
    inner_objective='mean_subject_macro_f1', selection_rule='absolute_paired_dz_then_change_spearman')
RBF_DEFAULTS = dict(C=[.1,1.,10.,100.], gamma=['scale',.01,.1,1.],
    max_features=[2,3,5,8,12,20,36], rbf_cache_mb=128,max_iter=-1,rbf_tol=1e-3)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_data(data_dir):
    data_dir = Path(data_dir)
    manifest = json.loads((data_dir/'data_manifest.json').read_text())
    for name, key in [('model_windows.csv','data_sha256'),('feature_dictionary.csv','dictionary_sha256')]:
        if sha(data_dir/name) != manifest[key]:
            raise ValueError(f'{name} differs from the exported manifest.')
    w = pd.read_csv(data_dir/'model_windows.csv')
    dictionary = pd.read_csv(data_dir/'feature_dictionary.csv')
    features = dictionary.feature.tolist()
    if not w.window_id.is_unique or not w.Label.isin(CLASSES).all():
        raise ValueError('Repeated windows or unexpected labels.')
    if w.groupby('subject_id').Label.nunique().ne(2).any():
        raise ValueError('Every outer test participant must have both classes.')
    if len(features) != 36 or len(set(features)) != 36:
        raise ValueError('Expected the audited 36 features.')
    if not w.subject_id.str.fullmatch(r'S\d{3}').all():
        raise ValueError('Unexpected participant identifiers.')
    if (w[features].dtypes == object).any():
        raise ValueError('Features must be numeric.')
    w[features] = w[features].replace([np.inf,-np.inf],np.nan)
    if w.subject_id.nunique() != manifest['n_subjects'] or len(w) != manifest['n_windows']:
        raise ValueError('Bundle dimensions do not match manifest.')
    return w, dictionary, manifest


class FeatureSelector(TransformerMixin, BaseEstimator):
    """Cloneable selector; all group summaries are computed in fit, never transform."""
    def __init__(self, features, mode='selected', max_features=3,
                 redundancy_abs_rho=.9, min_windows_per_stage=3,
                 min_subjects_per_feature=3, require_both_legs=True):
        self.features = features
        self.mode = mode
        self.max_features = max_features
        self.redundancy_abs_rho = redundancy_abs_rho
        self.min_windows_per_stage = min_windows_per_stage
        self.min_subjects_per_feature = min_subjects_per_feature
        self.require_both_legs = require_both_legs

    def fit(self, X, y):
        self.training_subjects_ = sorted(X.subject_id.unique().tolist())
        if self.mode == 'baseline':
            self.selected_ = ['RMS','MDF']
            self.ranking_ = pd.DataFrame({'feature':self.selected_, 'selected':[True,True],
                'reason':['assignment_fixed','assignment_fixed']})
            return self
        features = list(self.features)
        z = X[['subject_id','recording_id','leg']+features].reset_index(drop=True).copy()
        z[features] = z[features].replace([np.inf,-np.inf],np.nan)
        z['_stage'] = np.asarray(y, dtype=int)
        ids = ['subject_id','leg','recording_id']
        grouped = z.groupby(ids+['_stage'], observed=True)[features]
        med, counts = grouped.median(), grouped.count()
        early = med.xs(0, level='_stage')
        late = med.xs(1, level='_stage')
        enough = (counts.xs(0,level='_stage') >= self.min_windows_per_stage) & (
            counts.xs(1,level='_stage') >= self.min_windows_per_stage)
        changes = (late - early).where(enough)
        legs = changes.groupby(level=['subject_id','leg']).mean()
        subjects = legs.groupby(level='subject_id').mean()
        if self.require_both_legs:
            subjects = subjects.where(legs.notna().groupby(level='subject_id').sum() == 2)
        self.subject_changes_ = subjects
        rows = []
        for order, f in enumerate(features):
            d = subjects[f].dropna()
            mean, sd = d.mean(), d.std(ddof=1)
            # A zero-variance effect is reported explicitly, without a silent infinity.
            usable = len(d) >= self.min_subjects_per_feature and np.isfinite(sd) and sd > 1e-12
            score = abs(float(mean / sd)) if usable else np.nan
            rows.append(dict(feature=f, dictionary_order=order, n_subjects=len(d),
                mean_change=mean, sd_change=sd, abs_paired_dz=score,
                selected=False, reason='candidate' if usable else 'insufficient_or_constant_changes',
                redundant_with='', max_abs_rho=np.nan, undefined_correlations=0))
        rank = pd.DataFrame(rows).sort_values(['abs_paired_dz','dictionary_order'],
            ascending=[False,True], na_position='last').reset_index(drop=True)
        selected = []
        for i, row in rank.iterrows():
            if not np.isfinite(row.abs_paired_dz):
                continue
            correlated = ''; maximum = np.nan; undefined = 0
            for prior in selected:
                pair = subjects[[row.feature, prior]].dropna()
                if len(pair) < self.min_subjects_per_feature or (pair.nunique() < 2).any():
                    undefined += 1
                    continue
                rho = abs(float(pair.corr(method='spearman').iloc[0,1]))
                maximum = rho if np.isnan(maximum) else max(maximum, rho)
                if rho >= self.redundancy_abs_rho:
                    correlated = prior
                    break
            rank.loc[i,['redundant_with','max_abs_rho','undefined_correlations']] = [correlated,maximum,undefined]
            if correlated:
                rank.loc[i,'reason'] = 'redundant_training_subject_changes'
            elif undefined:
                # Do not assert nonredundancy when a comparison cannot be estimated.
                rank.loc[i,'reason'] = 'redundancy_unverifiable'
            elif len(selected) < self.max_features:
                selected.append(row.feature)
                rank.loc[i,['selected','reason']] = [True,'selected']
            else:
                rank.loc[i,'reason'] = 'feature_limit'
        if not selected:
            raise ValueError('Training-only selection found no estimable features; inspect training coverage.')
        self.selected_, self.ranking_ = selected, rank
        return self

    def transform(self, X):
        return X[self.selected_].replace([np.inf,-np.inf],np.nan).copy()

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.selected_, dtype=object)


def metrics(y, pred, score):
    y, pred, score = np.asarray(y), np.asarray(pred), np.asarray(score, dtype=float)
    p,r,f,s = precision_recall_fscore_support(y,pred,labels=CLASSES,zero_division=0)
    result = dict(accuracy=accuracy_score(y,pred), precision_macro=p.mean(),
        recall_macro=r.mean(), f1_macro=f.mean(), f1_class_0=f[0], f1_class_1=f[1],
        cohen_kappa=cohen_kappa_score(y,pred,labels=CLASSES),
        n_windows=len(y), support_0=int(s[0]), support_1=int(s[1]),
        roc_auc=np.nan, pr_auc=np.nan, average_precision=np.nan)
    if len(np.unique(y)) == 2:
        result['roc_auc'] = roc_auc_score(y,score)
        precision, recall, _ = precision_recall_curve(y,score,pos_label=1)
        result['pr_auc'] = auc(recall,precision)
        result['average_precision'] = average_precision_score(y,score)
    per_class = pd.DataFrame(dict(class_label=CLASSES,class_name=CLASS_NAMES,
        precision=p, recall=r, f1=f, support=s))
    return result, per_class


def positive_scores(model, X):
    if hasattr(model, 'decision_function'):
        return model.decision_function(X), 'decision_function'
    return model.predict_proba(X)[:, list(model.classes_).index(1)], 'probability_class_1'


def subject_macro_f1(estimator, X, y):
    pred = estimator.predict(X)
    subject = X.subject_id.to_numpy()
    # Each participant contributes once regardless of their number of sets/windows.
    return float(np.mean([precision_recall_fscore_support(
        np.asarray(y)[subject==s], pred[subject==s], labels=CLASSES,
        average='macro', zero_division=0)[2] for s in np.unique(subject)]))


def make_search(model_key, features, cfg, X, y, groups):
    selector = FeatureSelector(features=tuple(features),
        mode='baseline' if model_key=='lr_rms_mdf' else 'selected',
        redundancy_abs_rho=cfg['redundancy_abs_rho'],
        min_windows_per_stage=cfg['min_windows_per_stage'],
        min_subjects_per_feature=cfg['min_subjects_per_feature'],
        require_both_legs=cfg['require_both_legs'])
    if model_key == 'svm_rbf_selected':
        clf = SVC(kernel='rbf', probability=False, cache_size=cfg['rbf_cache_mb'],
            random_state=cfg['seed'], max_iter=cfg['max_iter'], tol=cfg['rbf_tol'])
    elif model_key == 'svm_selected':
        clf = LinearSVC(C=1., penalty='l2', loss='squared_hinge', dual=False,
            max_iter=cfg['max_iter'], random_state=cfg['seed'])
    else:
        clf = LogisticRegression(C=1., solver='lbfgs', max_iter=cfg['max_iter'],
            random_state=cfg['seed'])
    pipe = Pipeline([('select',selector), ('impute',SimpleImputer(strategy='median',keep_empty_features=True)),
        ('scale',StandardScaler()), ('model',clf)])
    grid = {'model__C':cfg['C']}
    if model_key != 'lr_rms_mdf':
        grid['select__max_features'] = cfg['max_features']
    if model_key == 'svm_rbf_selected':
        grid['model__gamma'] = cfg['gamma']
    inner = list(GroupKFold(n_splits=cfg['inner_folds'],shuffle=True,
        random_state=cfg['seed']).split(X,y,groups))
    for tr, va in inner:
        if set(groups[tr]) & set(groups[va]):
            raise AssertionError('Inner subject leakage.')
    search = GridSearchCV(pipe,grid,scoring=subject_macro_f1,cv=inner,
        refit=True,n_jobs=cfg['n_jobs'],return_train_score=False,error_score='raise',
        verbose=1 if model_key=='svm_rbf_selected' else 0)
    return search, inner


def _json(path, value):
    Path(path).write_text(json.dumps(value,indent=2,default=str)+'\n',encoding='utf-8')


def _selection_artifacts(model, Xtrain, ytrain, features, directory):
    sel = model.named_steps['select']
    sel.ranking_.to_csv(directory/'feature_selection.csv',index=False)
    if hasattr(sel,'subject_changes_'):
        sel.subject_changes_.to_csv(directory/'training_subject_changes.csv')
    names = sel.selected_
    clf = model.named_steps['model']
    if isinstance(clf,SVC) and clf.kernel != 'linear':
        _json(directory/'nonlinear_model_details.json',dict(kernel=clf.kernel,C=clf.C,
            gamma_setting=clf.gamma,effective_gamma=float(clf._gamma),
            support_vectors_per_class=clf.n_support_.tolist(),n_support_vectors=len(clf.support_),
            explanation='RBF SVM has no single linear coefficient per input feature. '
                'Selection frequency is descriptive stability, not feature importance.'))
    else:
        pd.DataFrame({'feature':names,'standardized_coefficient':clf.coef_[0]}).to_csv(
            directory/'coefficients.csv',index=False)
    # Selected-feature window correlation is a separate diagnostic, not an extra selection pass.
    Xtrain[names].corr(method='spearman').to_csv(directory/'training_window_correlations.csv')
    return names


def _fit_with_warnings(estimator, X, y):
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        estimator.fit(X,y)
    messages = [dict(category=w.category.__name__, message=str(w.message)) for w in captured]
    if any(issubclass(w.category,ConvergenceWarning) for w in captured):
        raise RuntimeError('Convergence warning: run is incomplete; inspect warnings and revise solver settings explicitly.')
    return messages


def summary_table(folds):
    return pd.DataFrame([dict(metric=m,mean=folds[m].mean(),sd=folds[m].std(ddof=1),
        n_defined=int(folds[m].notna().sum()), n_folds=len(folds)) for m in METRICS])


def _figures(oof, folds, summary, out, title):
    figs = out/'figures'; figs.mkdir(exist_ok=True)
    cm = confusion_matrix(oof.y_true,oof.y_pred,labels=CLASSES)
    norm = np.divide(cm,cm.sum(axis=1,keepdims=True),out=np.zeros_like(cm,dtype=float),
        where=cm.sum(axis=1,keepdims=True)!=0)
    pd.DataFrame(cm,index=CLASSES,columns=CLASSES).to_csv(out/'confusion_counts.csv')
    pd.DataFrame(norm,index=CLASSES,columns=CLASSES).to_csv(out/'confusion_row_normalized.csv')
    fig, axes = plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for ax,values,name,fmt in [(axes[0],cm,'Counts','d'),(axes[1],norm,'Row-normalized','.2f')]:
        ax.imshow(values,cmap='Blues',vmin=0)
        for i in CLASSES:
            for j in CLASSES: ax.text(j,i,format(values[i,j],fmt),ha='center',va='center')
        ax.set(xticks=CLASSES,yticks=CLASSES,xlabel='Predicted stage',ylabel='True stage',title=name)
    fig.suptitle(title); fig.savefig(figs/'confusion_matrices.png',dpi=170); plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for _,g in oof.groupby('fold'):
        if g.y_true.nunique() < 2: continue
        fpr,tpr,_ = roc_curve(g.y_true,g.score)
        precision,recall,_ = precision_recall_curve(g.y_true,g.score)
        axes[0].plot(fpr,tpr,alpha=.45,lw=1,label=g.subject_id.iloc[0])
        axes[1].step(recall,precision,where='post',alpha=.45,lw=1)
    axes[0].plot([0,1],[0,1],'k--',lw=1)
    axes[0].set(xlabel='False positive rate',ylabel='True positive rate',title='Held-out subject ROC curves')
    axes[1].set(xlabel='Recall (late class)',ylabel='Precision (late class)',title='Held-out subject PR curves')
    for ax in axes: ax.set(xlim=(0,1),ylim=(0,1.02))
    axes[0].legend(fontsize=6,ncol=3)
    fig.savefig(figs/'roc_pr_by_subject.png',dpi=170);plt.close(fig)
    fig,ax = plt.subplots(figsize=(10,4),layout='constrained')
    ax.bar(folds.test_subject,folds.f1_macro)
    ax.axhline(folds.f1_macro.mean(),color='black',linestyle='--',label='Subject mean')
    ax.set(ylim=(0,1),xlabel='Held-out subject',ylabel='Macro-F1',title=title)
    ax.tick_params(axis='x',rotation=45); ax.legend()
    fig.savefig(figs/'per_subject_f1.png',dpi=170);plt.close(fig)
    # Metrics shown on comparable scale; kappa can be negative, so retain negative values.
    show = summary[summary.metric.isin(['accuracy','precision_macro','recall_macro','f1_macro','cohen_kappa','roc_auc','pr_auc'])]
    fig,ax = plt.subplots(figsize=(10,4),layout='constrained')
    ax.errorbar(np.arange(len(show)),show['mean'],yerr=show.sd,fmt='o',capsize=4)
    ax.set(xticks=np.arange(len(show)),xticklabels=show.metric,ylabel='Mean ± sample SD across subjects',title=title)
    ax.tick_params(axis='x',rotation=25); ax.axhline(0,color='grey',lw=.5)
    fig.savefig(figs/'metrics_mean_sd.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,3),layout='constrained')
    ax.axis('off')
    boxes=[(.10,.65,'15 participants\nLOSO outer split'),
        (.37,.72,'14 training participants\n3-fold grouped inner search'),
        (.67,.72,'Refit selected pipeline\non 14 training participants'),
        (.37,.15,'1 untouched test participant\nboth legs and all sets'),
        (.90,.43,'Test once\nrepeat for each subject')]
    for x,z,label in boxes:
        ax.text(x,z,label,ha='center',va='center',transform=ax.transAxes,fontsize=9,
            bbox=dict(boxstyle='round',facecolor='#e8f1fa',edgecolor='#294b6d'))
    for start,end in [((.19,.65),(.25,.72)),((.19,.60),(.25,.15)),
            ((.49,.72),(.55,.72)),((.78,.72),(.85,.48)),((.49,.15),(.85,.38))]:
        ax.annotate('',xy=end,xytext=start,xycoords='axes fraction',
            arrowprops=dict(arrowstyle='->',color='#294b6d'))
    ax.set_title('Selection → imputation → scaling → classifier fitted within training splits\n'
        'Majority baseline only estimates the training class prior; no inner tuning')
    fig.savefig(figs/'validation_diagram.png',dpi=170);plt.close(fig)


def run_model(model_key, data_dir, output_root, config=None):
    """Run only when called explicitly by a notebook or CLI. Serial, bounded threads."""
    if model_key not in MODEL_NAMES:
        raise ValueError('Unknown model key.')
    cfg = {**DEFAULTS, **(RBF_DEFAULTS if model_key=='svm_rbf_selected' else {}), **(config or {})}
    if cfg['outer_validation'] != 'LOSO' or cfg['n_jobs'] != 1:
        raise ValueError('This approved implementation uses LOSO and serial fitting.')
    w,dictionary,source = load_data(data_dir)
    features = dictionary.feature.tolist()
    X = w[['subject_id','recording_id','leg']+features].copy()
    y = w.Label.to_numpy(dtype=int); groups = w.subject_id.to_numpy()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out = Path(output_root)/model_key/stamp
    out.mkdir(parents=True,exist_ok=False)
    code_hashes = {p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}
    manifest = dict(status='running',model_key=model_key,model_name=MODEL_NAMES[model_key],
        created_utc=stamp,config=cfg,data_manifest=source,code_sha256=code_hashes,
        versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,
            scipy=scipy.__version__,sklearn=sklearn.__version__,joblib=joblib.__version__),
        class_names=CLASS_NAMES,positive_class=1,
        thresholds=('Default libsvm SVC.predict binary vote; no threshold tuning; '
            'positive decision scores indicate class 1 (ties use libsvm behavior).'
            if model_key=='svm_rbf_selected' else
            'Default sklearn predict: LR cutoff 0.5, SVM cutoff 0; exact ties choose class 0; no threshold tuning'),
        aggregation='Primary mean and sample SD across held-out participants; pooled window metrics secondary',
        caution='Exploratory development after whole-cohort inspection; no independent external test cohort.')
    _json(out/'run_manifest.json',manifest)
    folds=[]; predictions=[]; perclasses=[]; splits=[]; selections=[]; legrows=[]
    started=time.perf_counter()
    try:
        with threadpool_limits(limits=1):
            for fold,(train,test) in enumerate(LeaveOneGroupOut().split(X,y,groups),start=1):
                tick=time.perf_counter()
                train_subjects=sorted(set(groups[train])); test_subjects=sorted(set(groups[test]))
                assert not set(train_subjects)&set(test_subjects) and len(test_subjects)==1
                subject=test_subjects[0]; directory=out/f'fold_{fold:02d}_{subject}'
                directory.mkdir()
                Xtr,Xte=X.iloc[train],X.iloc[test]; ytr,yte=y[train],y[test]
                split=dict(fold=fold,train_subjects=train_subjects,test_subjects=test_subjects,inner=[])
                if model_key=='majority':
                    # A prior score supports discrimination metrics; prediction is training majority.
                    fitted=DummyClassifier(strategy='prior',random_state=cfg['seed'])
                    warn=_fit_with_warnings(fitted,Xtr[['RMS']],ytr)
                    pred=fitted.predict(Xte[['RMS']]); score,kind=positive_scores(fitted,Xte[['RMS']])
                    names=[]; best={}; inner_score=np.nan
                else:
                    search,inner=make_search(model_key,features,cfg,Xtr,ytr,groups[train])
                    for j,(itr,iva) in enumerate(inner,start=1):
                        split['inner'].append(dict(inner_fold=j,
                            train_subjects=sorted(set(groups[train][itr])),
                            validation_subjects=sorted(set(groups[train][iva]))))
                    warn=_fit_with_warnings(search,Xtr,ytr)
                    fitted=search.best_estimator_; best=search.best_params_; inner_score=search.best_score_
                    cv_results=pd.DataFrame(search.cv_results_)
                    cv_results.to_csv(directory/'inner_search.csv',index=False)
                    names=_selection_artifacts(fitted,Xtr,ytr,features,directory)
                    # Reconstruct the winning selector in each inner training split for audit.
                    # No inner validation labels enter these fits; no classifier is refitted here.
                    for j,(itr,iva) in enumerate(inner,start=1):
                        inner_sel=clone(fitted.named_steps['select']).fit(Xtr.iloc[itr],ytr[itr])
                        inner_sel.ranking_.to_csv(directory/f'inner_{j}_winning_selection.csv',index=False)
                    pred=fitted.predict(Xte); score,kind=positive_scores(fitted,Xte)
                joblib.dump(fitted,directory/'pipeline.joblib')
                _json(directory/'fit_warnings.json',warn)
                _json(directory/'split.json',split); splits.append(split)
                result,pc=metrics(yte,pred,score)
                row=dict(model=model_key,fold=fold,test_subject=subject,n_train_subjects=len(train_subjects),
                    n_train_windows=len(train),n_test_windows=len(test),
                    selected_features='|'.join(names),n_features=len(names),
                    best_params=json.dumps(best,sort_keys=True),inner_mean_subject_macro_f1=inner_score,
                    score_kind=kind,seconds=time.perf_counter()-tick,**result)
                folds.append(row)
                pc['fold']=fold; pc['subject_id']=subject; perclasses.append(pc)
                oo=w.iloc[test][['window_id','subject_id','recording_id','set_token','leg',
                    'sample_start','sample_end_exclusive','t_start_s','t_end_exclusive_s']].copy()
                oo['fold']=fold;oo['y_true']=yte;oo['y_pred']=pred;oo['score']=score;oo['score_kind']=kind
                predictions.append(oo)
                for leg,g in oo.groupby('leg'):
                    lm,_=metrics(g.y_true,g.y_pred,g.score)
                    legrows.append(dict(fold=fold,subject_id=subject,leg=leg,**lm))
                for name in names: selections.append(dict(fold=fold,subject_id=subject,feature=name))
                # Recoverable partial results, but completion requires all outer folds.
                pd.DataFrame(folds).to_csv(out/'partial_results.csv',index=False)
                print(f'{model_key} | {fold}/{len(np.unique(groups))} | {subject} | '
                    f'macro-F1={result["f1_macro"]:.3f} | features={names or "none"}',flush=True)
            fold_table=pd.DataFrame(folds); oof=pd.concat(predictions,ignore_index=True)
            if len(oof)!=len(w) or not oof.window_id.is_unique or set(oof.window_id)!=set(w.window_id):
                raise AssertionError('Out-of-fold coverage is incomplete or repeated.')
            fold_table.to_csv(out/'results.csv',index=False)
            oof.to_csv(out/'out_of_fold_predictions.csv',index=False)
            pd.concat(perclasses,ignore_index=True).to_csv(out/'per_class_metrics.csv',index=False)
            fold_table.rename(columns={'test_subject':'subject_id'}).to_csv(out/'per_subject_metrics.csv',index=False)
            legs=pd.DataFrame(legrows);legs.to_csv(out/'per_subject_leg_metrics.csv',index=False)
            # Paired leg table; no simultaneous bilateral channel correlation is implied.
            legs.pivot(index='subject_id',columns='leg',values='f1_macro').to_csv(out/'paired_leg_f1.csv')
            if selections:
                chosen=pd.DataFrame(selections)
                chosen=chosen.merge(dictionary[['feature','family']],on='feature',validate='many_to_one')
                chosen.to_csv(out/'selected_features_by_fold.csv',index=False)
                chosen.groupby(['feature','family']).size().rename('outer_folds_selected').sort_values(
                    ascending=False).to_csv(out/'selection_frequency.csv')
            summary=summary_table(fold_table);summary.to_csv(out/'metrics_mean_sd.csv',index=False)
            pc=pd.concat(perclasses,ignore_index=True)
            pc.groupby('class_label')[['precision','recall','f1']].agg(['mean','std']).to_csv(out/'per_class_mean_sd.csv')
            pooled,pooled_classes=metrics(oof.y_true,oof.y_pred,np.zeros(len(oof)))
            # Raw SVM margins and training priors differ across models/folds: do not pool AUCs.
            for key in ['roc_auc','pr_auc','average_precision']: pooled.pop(key)
            _json(out/'pooled_window_metrics_secondary.json',pooled)
            pooled_classes.to_csv(out/'pooled_per_class_metrics.csv',index=False)
            _json(out/'subject_splits.json',splits)
            _figures(oof,fold_table,summary,out,MODEL_NAMES[model_key])
            if cfg['final_refit']:
                final_dir=out/'final_refit';final_dir.mkdir()
                if model_key=='majority':
                    final=DummyClassifier(strategy='prior',random_state=cfg['seed'])
                    warn=_fit_with_warnings(final,X[['RMS']],y)
                    final_names=[]; final_params={}; final_inner=[]
                else:
                    search,inner=make_search(model_key,features,cfg,X,y,groups)
                    warn=_fit_with_warnings(search,X,y);final=search.best_estimator_
                    pd.DataFrame(search.cv_results_).to_csv(final_dir/'inner_search.csv',index=False)
                    final_names=_selection_artifacts(final,X,y,features,final_dir)
                    final_params=search.best_params_
                    final_inner=[dict(train_subjects=sorted(set(groups[a])),validation_subjects=sorted(set(groups[b])))
                        for a,b in inner]
                joblib.dump(final,final_dir/'pipeline.joblib')
                _json(final_dir/'fit_warnings.json',warn)
                _json(final_dir/'model_card.json',dict(model_key=model_key,selected_features=final_names,
                    best_params=final_params,inner_splits=final_inner,
                    purpose='Refit on all participants after evaluation; no new unbiased test score',
                    data_sha256=source['data_sha256'],feature_order=features))
            manifest.update(status='complete',n_outer_folds=len(folds),total_seconds=time.perf_counter()-started)
            _json(out/'run_manifest.json',manifest)
            _json(Path(output_root)/f'latest_{model_key}.json',dict(path=str(out.resolve()),status='complete'))
            refresh_combined_results(output_root)
    except Exception as exc:
        manifest.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        _json(out/'run_manifest.json',manifest)
        raise
    return out


def refresh_combined_results(output_root):
    """Combine only each model's latest completed run for the identical data/config/code."""
    root=Path(output_root);tables=[];reference=None;included=[]
    # Keep the original four-model comparison intact. RBF is a separately specified
    # exploratory extension with a different search space and its own results.csv.
    for key in ORIGINAL_MODELS:
        pointer=root/f'latest_{key}.json'
        if not pointer.exists(): continue
        out=Path(json.loads(pointer.read_text())['path'])
        manifest=json.loads((out/'run_manifest.json').read_text())
        signature=(manifest['data_manifest']['data_sha256'],json.dumps(manifest['config'],sort_keys=True),
            json.dumps(manifest['code_sha256'],sort_keys=True))
        if manifest['status']!='complete': continue
        if reference is None: reference=signature
        if signature != reference:
            warnings.warn('Latest model runs differ in data/config/code. Combined results were not refreshed.')
            return
        tables.append(pd.read_csv(out/'results.csv'));included.append(dict(model=key,path=str(out)))
    if tables:
        pd.concat(tables,ignore_index=True).to_csv(root/'results.csv',index=False)
        _json(root/'combined_results_manifest.json',dict(models=included,
            complete_four_model_comparison=len(tables)==4))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model',choices=list(MODEL_NAMES))
    parser.add_argument('--data',type=Path,default=Path(__file__).parent/'data')
    parser.add_argument('--outputs',type=Path,default=Path(__file__).parent/'outputs')
    args=parser.parse_args()
    print(run_model(args.model,args.data,args.outputs))
