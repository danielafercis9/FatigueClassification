"""Fixed feature ANN using notebook 07 selection and LOSO; no inner search."""
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import time
import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits
from modelling.core import load_data, sha, metrics, summary_table, _figures, _fit_with_warnings
from modelling.fixed_rbf import SignificantFeatureSelector, FIXED_RBF_CONFIG

ANN_CONFIG = {k:v for k,v in FIXED_RBF_CONFIG.items() if k not in
    ['C','gamma','cache_mb','tol','max_iter']}
ANN_CONFIG.update(hidden_layers=(32,16), epochs=100, batch_size=128,
    learning_rate_init=.001, alpha_l2=.0001)
MODEL_KEY = 'ann_fixed_significant'


class FixedEpochANN(ClassifierMixin, BaseEstimator):
    """Train exactly the specified epochs, retaining Adam state between epochs."""
    def __init__(self, hidden_layers=(32,16), epochs=100, batch_size=128,
                 learning_rate_init=.001, alpha_l2=.0001, seed=20261018):
        self.hidden_layers=hidden_layers; self.epochs=epochs; self.batch_size=batch_size
        self.learning_rate_init=learning_rate_init; self.alpha_l2=alpha_l2; self.seed=seed

    def fit(self, X, y):
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError('Epochs and batch size must be positive.')
        if set(np.unique(y)) != {0,1}:
            raise ValueError('Binary early/late training labels are required.')
        self.network_ = MLPClassifier(hidden_layer_sizes=self.hidden_layers,
            activation='relu', solver='adam', alpha=self.alpha_l2,
            batch_size=min(self.batch_size,len(y)), learning_rate_init=self.learning_rate_init,
            shuffle=True, random_state=self.seed, early_stopping=False)
        self.training_loss_=[]
        for epoch in range(self.epochs):
            self.network_.partial_fit(X,y,classes=np.array([0,1]))
            self.training_loss_.append(float(self.network_.loss_))
        if not np.isfinite(self.training_loss_).all():
            raise ValueError('Nonfinite ANN training loss.')
        self.classes_=self.network_.classes_; self.n_features_in_=self.network_.n_features_in_
        return self

    def predict_proba(self, X):
        return self.network_.predict_proba(X)

    def predict(self, X):
        return (self.predict_proba(X)[:,1] > .5).astype(int)


def make_pipeline(features,cfg):
    return Pipeline([
        ('select',SignificantFeatureSelector(tuple(features),alpha=cfg['alpha'],
            redundancy_abs_rho=cfg['redundancy_abs_rho'],
            min_windows_per_stage=cfg['min_windows_per_stage'],
            min_subjects_per_feature=cfg['min_subjects_per_feature'],
            require_both_legs=cfg['require_both_legs'],zero_tolerance=cfg['zero_tolerance'])),
        ('impute',SimpleImputer(strategy='median',keep_empty_features=True)),
        ('scale',StandardScaler()),
        ('model',FixedEpochANN(**{k:cfg[k] for k in ['hidden_layers','epochs',
            'batch_size','learning_rate_init','alpha_l2','seed']}))])


def _selection_artifacts(model,Xtrain,ytrain,features,directory):
    selector=model.named_steps['select']; clf=model.named_steps['model']
    selector.ranking_.to_csv(directory/'feature_selection.csv',index=False)
    selector.subject_changes_.to_csv(directory/'training_subject_changes.csv')
    names=selector.selected_
    Xtrain[names].corr(method='spearman').to_csv(directory/'training_window_correlations.csv')
    pd.DataFrame({'epoch':np.arange(1,len(clf.training_loss_)+1),
        'training_loss':clf.training_loss_}).to_csv(directory/'training_history.csv',index=False)
    _json(directory/'ann_model_details.json',dict(hidden_layers=clf.hidden_layers,
        epochs=len(clf.training_loss_),weight_shapes=[list(a.shape) for a in clf.network_.coefs_],
        trainable_parameters=sum(a.size for a in clf.network_.coefs_+clf.network_.intercepts_),
        explanation='Neural weights and selection frequency are not per-feature importance.'))
    return names


def _json(path,value):Path(path).write_text(json.dumps(value,indent=2,default=str)+'\n',encoding='utf-8')


def _fit(model,X,y,directory):
    try:return _fit_with_warnings(model,X,y)
    except Exception:
        selector=model.named_steps['select']
        if hasattr(selector,'ranking_'):selector.ranking_.to_csv(directory/'feature_selection.csv',index=False)
        raise


def _fixed_diagram(out,n,cfg):
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(12,3.5),layout='constrained');ax.axis('off')
    boxes=[(.22,.57,f'{n-1} training participants\nSignificance + redundancy filter\nFit imputation and scaling\nFit ANN: {cfg["hidden_layers"]}, {cfg["epochs"]} epochs'),
        (.67,.57,'One held-out participant\nFrozen pipeline\nEvaluate once'),
        (.92,.57,f'Repeat for\n{n} subjects')]
    for x,y,label in boxes:ax.text(x,y,label,ha='center',va='center',transform=ax.transAxes,
        bbox=dict(boxstyle='round',facecolor='#e8f1fa'),fontsize=10)
    for a,b in [(.43,.51),(.8,.85)]:ax.annotate('',xy=(b,.57),xytext=(a,.57),
        xycoords='axes fraction',arrowprops=dict(arrowstyle='->'))
    ax.set_title('LOSO — one fixed configuration; no inner splits, grid search or threshold tuning')
    fig.savefig(out/'figures/validation_diagram.png',dpi=170);plt.close(fig)


def run_ann(data_dir,output_root,config=None):
    cfg={**ANN_CONFIG,**(config or {})}
    if cfg['correction']!='BY' or cfg['n_jobs']!=1 or cfg['outer_validation']!='LOSO':
        raise ValueError('This experiment fixes BY, serial execution and LOSO.')
    w,dictionary,source=load_data(data_dir);features=dictionary.feature.tolist()
    X=w[['subject_id','recording_id','leg']+features].copy()
    y=w.Label.to_numpy(dtype=int);groups=w.subject_id.to_numpy()
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out=Path(output_root)/MODEL_KEY/stamp;out.mkdir(parents=True,exist_ok=False)
    manifest=dict(status='running',model_key=MODEL_KEY,created_utc=stamp,config=cfg,
        data_manifest=source,code_sha256={p.name:sha(p) for p in [Path(__file__),Path(__file__).parent/'core.py',Path(__file__).parent/'fixed_rbf.py']},
        versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,
            scipy=scipy.__version__,sklearn=sklearn.__version__,joblib=joblib.__version__),
        hardware=dict(device='CPU',platform=platform.platform(),machine=platform.machine(),
            processor=platform.processor(),numerical_threads=1),
        validation='LOSO; no inner folds or hyperparameter search; feature selection fitted on outer-training subjects only',
        score='Uncalibrated ANN sigmoid probabilities; class 1 if p>0.5, ties class 0; no threshold tuning',
        limitations=['Fixed pragmatic architecture, Adam and epoch budget, not optimized settings.',
            'Exploratory extension proposed after earlier model results were inspected.',
            'Same selection/scaling and splits as notebook 07; classifier and optimization differ.',
            'Significance indicates consistent stage changes, not physiological specificity or guaranteed prediction.',
            'Participant-change redundancy filtering does not guarantee low window-level predictor dependence.'])
    _json(out/'run_manifest.json',manifest)
    folds=[];predictions=[];classes=[];legs=[];selections=[];splits=[];started=time.perf_counter()
    try:
        with threadpool_limits(limits=1):
            print('Fixed ANN: one CPU fit per held-out participant, no inner search.',flush=True)
            for fold,(train,test) in enumerate(LeaveOneGroupOut().split(X,y,groups),start=1):
                start=time.perf_counter();subject=groups[test][0];training_subjects=sorted(set(groups[train]))
                assert subject not in training_subjects
                directory=out/f'fold_{fold:02d}_{subject}';directory.mkdir()
                model=make_pipeline(features,cfg)
                print(f'Fold {fold}/{len(np.unique(groups))}: one fit; test {subject}.',flush=True)
                warns=_fit(model,X.iloc[train],y[train],directory)
                pred=model.predict(X.iloc[test]);scores=model.predict_proba(X.iloc[test])[:,1]
                result,pc=metrics(y[test],pred,scores)
                names=_selection_artifacts(model,X.iloc[train],y[train],features,directory)
                joblib.dump(model,directory/'pipeline.joblib');_json(directory/'fit_warnings.json',warns)
                split=dict(fold=fold,train_subjects=training_subjects,test_subjects=[subject],inner=[])
                _json(directory/'split.json',split);splits.append(split)
                significant=int(model.named_steps['select'].ranking_.passes_significance.sum())
                folds.append(dict(model=MODEL_KEY,fold=fold,test_subject=subject,n_train_subjects=len(training_subjects),
                    n_train_windows=len(train),n_test_windows=len(test),n_features=len(names),n_significant=significant,
                    selected_features='|'.join(names),epochs=cfg['epochs'],hidden_layers=str(cfg['hidden_layers']),
                    score_kind='predict_proba',seconds=time.perf_counter()-start,**result))
                pc['fold']=fold;pc['subject_id']=subject;classes.append(pc)
                oo=w.iloc[test][['window_id','subject_id','recording_id','set_token','leg',
                    'sample_start','sample_end_exclusive','t_start_s','t_end_exclusive_s']].copy()
                oo['fold']=fold;oo['y_true']=y[test];oo['y_pred']=pred;oo['score']=scores;oo['score_kind']='predict_proba'
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
            _figures(oof,fold_table,summary,out,'ANN — fixed settings, training-only significant features')
            _fixed_diagram(out,len(np.unique(groups)),cfg)
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
    args=parser.parse_args();print(run_ann(args.data,args.outputs))
