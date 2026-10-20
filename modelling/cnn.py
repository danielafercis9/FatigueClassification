"""Fixed small waveform CNN, LOSO, no inner search or early stopping."""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from datetime import datetime,timezone
import gc
import json
from pathlib import Path
import platform
import random
import time
from importlib.metadata import version

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from sklearn.model_selection import LeaveOneGroupOut
from modelling.core import metrics,summary_table,_figures,sha

CNN_CONFIG=dict(seed=20261018,epochs=20,batch_size=128,learning_rate=.001,
    weight_decay=.0001,dropout=.2,final_refit=True,device='auto',cpu_threads=1,
    selection='none',inner_validation='none_fixed_configuration',
    architecture='Conv1d(1,16,k9,p4)-ReLU-MaxPool4-Conv1d(16,32,k5,p2)-ReLU-MaxPool4-GlobalAveragePool-Dropout-Linear(32,1)')


class SmallCNN(nn.Module):
    def __init__(self,dropout=.2):
        super().__init__()
        self.layers=nn.Sequential(nn.Conv1d(1,16,9,padding=4),nn.ReLU(),nn.MaxPool1d(4),
            nn.Conv1d(16,32,5,padding=2),nn.ReLU(),nn.MaxPool1d(4),
            nn.AdaptiveAvgPool1d(1),nn.Flatten(),nn.Dropout(dropout),nn.Linear(32,1))

    def forward(self,x):return self.layers(x).squeeze(-1)


def load_signals(data_dir,model_data_dir=None):
    data_dir=Path(data_dir)
    manifest=json.loads((data_dir/'signal_manifest.json').read_text())
    for name,key in [('window_metadata.csv','metadata_sha256'),('signal_windows.npz','signals_sha256')]:
        if sha(data_dir/name)!=manifest[key]:raise ValueError(f'{name} checksum mismatch.')
    w=pd.read_csv(data_dir/'window_metadata.csv')
    with np.load(data_dir/'signal_windows.npz',allow_pickle=False) as z:
        x=z['signals'];ids=z['window_id']
    if x.shape!=(len(w),500) or not np.isfinite(x).all():raise ValueError('Expected finite 500-sample windows.')
    if not np.array_equal(ids,w.window_id.to_numpy()) or not w.window_id.is_unique:
        raise ValueError('Signals and metadata have different/repeated window IDs.')
    if not w.Label.isin([0,1]).all() or w.groupby('subject_id').Label.nunique().ne(2).any():
        raise ValueError('Each participant must have both proxy stages.')
    if len(w)!=manifest['n_windows'] or w.subject_id.nunique()!=manifest['n_subjects']:
        raise ValueError('Signal manifest counts differ.')
    if model_data_dir is not None:
        model_path=Path(model_data_dir)/'model_windows.csv'
        if sha(model_path)!=manifest['source_model_data_sha256']:
            raise ValueError('CNN and feature models use different input versions.')
        reference=pd.read_csv(model_path,usecols=['window_id','subject_id','recording_id','leg','Label'])
        pd.testing.assert_frame_equal(w[reference.columns],reference,check_dtype=False)
    return np.asarray(x,dtype=np.float32),w,manifest


def fit_scaler(x):
    mean=float(np.mean(x,dtype=np.float64));std=float(np.std(x,dtype=np.float64))
    if not np.isfinite(std) or std<=1e-12:raise ValueError('Training signal scale is invalid.')
    return dict(mean=mean,std=std)


def standardize(x,scaler):
    return ((np.asarray(x,dtype=np.float32)-scaler['mean'])/scaler['std']).astype(np.float32)


def _seed(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True


def _device(value):
    name='cuda' if value=='auto' and torch.cuda.is_available() else ('cpu' if value=='auto' else value)
    return torch.device(name)


def fit_network(x,y,cfg,seed,device):
    """Receives training data only; no validation/test labels or epoch selection."""
    _seed(seed)
    model=SmallCNN(cfg['dropout']).to(device)
    scaler=fit_scaler(x)
    xx=torch.from_numpy(standardize(x,scaler)[:,None,:])
    yy=torch.from_numpy(np.asarray(y,dtype=np.float32))
    generator=torch.Generator().manual_seed(seed)
    loader=DataLoader(TensorDataset(xx,yy),batch_size=cfg['batch_size'],shuffle=True,
        generator=generator,num_workers=0)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    criterion=nn.BCEWithLogitsLoss();history=[]
    for epoch in range(1,cfg['epochs']+1):
        model.train();total=0.
        for inputs,labels in loader:
            inputs,labels=inputs.to(device),labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss=criterion(model(inputs),labels)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss; run is incomplete.')
            loss.backward();optimizer.step();total+=float(loss.detach().cpu())*len(labels)
        history.append(dict(epoch=epoch,train_loss=total/len(y)))
    return model,scaler,pd.DataFrame(history)


def score_network(model,x,scaler,device,batch_size=128):
    model.eval();scores=[]
    with torch.inference_mode():
        for start in range(0,len(x),batch_size):
            batch=standardize(x[start:start+batch_size],scaler)
            tensor=torch.from_numpy(batch[:,None,:]).to(device)
            scores.append(model(tensor).cpu().numpy())
    return np.concatenate(scores)


def save_model(path,model,scaler,cfg,seed,training_subjects):
    torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
        scaler=scaler,config=cfg,seed=seed,training_subjects=[str(s) for s in training_subjects],
        torch_version=str(torch.__version__),input_samples=500,
        normalization='Single scalar mean/std fitted over outer-training waveforms only',
        threshold=0.),path)


def predict_saved(path,x,device='auto',batch_size=128):
    target=_device(device)
    saved=torch.load(path,map_location='cpu',weights_only=True)
    model=SmallCNN(saved['config']['dropout'])
    model.load_state_dict(saved['state_dict']);model.to(target)
    scores=score_network(model,x,saved['scaler'],target,batch_size)
    return (scores>0).astype(int),scores


def _json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,default=str)+'\n',encoding='utf-8')


def _cnn_figures(oof,folds,summary,history,out,epochs):
    import matplotlib.pyplot as plt
    _figures(oof,folds,summary,out,'Small 1D CNN — fixed configuration')
    # Replace the shared nested-validation diagram with the actual fixed-CNN design.
    fig,ax=plt.subplots(figsize=(11,3),layout='constrained');ax.axis('off')
    n=oof.subject_id.nunique()
    text=[(.15,.55,f'{n-1} training participants\nFit scalar normalization\nTrain fixed CNN for {epochs} epochs'),
        (.60,.55,'One held-out participant\nFrozen normalization + CNN\nEvaluate once'),
        (.9,.55,f'Repeat for\n{n} subjects')]
    for a,b,s in text:ax.text(a,b,s,ha='center',va='center',transform=ax.transAxes,
        bbox=dict(boxstyle='round',facecolor='#e8f1fa'),fontsize=10)
    for a,b in [(.32,.45),(.75,.83)]:
        ax.annotate('',xy=(b,.55),xytext=(a,.55),xycoords='axes fraction',arrowprops=dict(arrowstyle='->'))
    ax.set_title('LOSO with fixed architecture/settings — no inner search or early stopping')
    fig.savefig(out/'figures/validation_diagram.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
    for _,g in history.groupby('fold'):ax.plot(g.epoch,g.train_loss,alpha=.6)
    ax.set(xlabel='Fixed training epoch',ylabel='Training BCE loss',title='Training diagnostics only — no epoch selection using test results')
    fig.savefig(out/'figures/training_loss.png',dpi=170);plt.close(fig)


def run_cnn(data_dir,output_root,config=None,model_data_dir=None):
    cfg={**CNN_CONFIG,**(config or {})}
    if cfg['epochs']<1 or cfg['batch_size']<1:raise ValueError('Epochs/batch size must be positive.')
    torch.set_num_threads(cfg['cpu_threads'])
    x,w,source=load_signals(data_dir,model_data_dir)
    groups=w.subject_id.to_numpy();y=w.Label.to_numpy(dtype=int)
    device=_device(cfg['device']);stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out=Path(output_root)/'cnn_fixed'/stamp;out.mkdir(parents=True,exist_ok=False)
    manifest=dict(status='running',model_key='cnn_fixed',created_utc=stamp,config=cfg,
        input_manifest=source,code_sha256={p.name:sha(p) for p in [Path(__file__),Path(__file__).parent/'core.py']},
        versions=dict(python=platform.python_version(),torch=str(torch.__version__),numpy=np.__version__,pandas=pd.__version__,
            scipy=version('scipy'),sklearn=version('scikit-learn'),matplotlib=version('matplotlib')),
        cuda_version=torch.version.cuda,device=str(device),
        device_name=torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU',
        validation='LOSO; fixed settings, no inner folds, no early stopping, no hyperparameter/threshold tuning',
        limitations=['Exploratory extension after earlier model results were inspected.',
            'Input changes from handcrafted features to filtered waveforms.',
            'Early/late stage proxies do not independently establish peripheral fatigue.',
            'Fixed epochs are a pragmatic budget, not an optimized training duration.',
            'Exact reproducibility across PyTorch versions/devices is not guaranteed.'])
    _json(out/'run_manifest.json',manifest)
    folds=[];predictions=[];classes=[];legs=[];splits=[];histories=[];started=time.perf_counter()
    print(f'CNN device: {device}; {len(np.unique(groups))} LOSO evaluation fits; {cfg["epochs"]} fixed epochs each.',flush=True)
    try:
        for fold,(train,test) in enumerate(LeaveOneGroupOut().split(x,y,groups),start=1):
            start=time.perf_counter();subject=groups[test][0];seed=cfg['seed']+fold
            training_subjects=sorted(set(groups[train]));assert subject not in training_subjects
            directory=out/f'fold_{fold:02d}_{subject}';directory.mkdir()
            print(f'Fold {fold}/{len(np.unique(groups))}: train on {len(training_subjects)} subjects; test {subject}.',flush=True)
            model,scaler,history=fit_network(x[train],y[train],cfg,seed,device)
            scores=score_network(model,x[test],scaler,device,cfg['batch_size']);pred=(scores>0).astype(int)
            result,pc=metrics(y[test],pred,scores)
            save_model(directory/'model.pt',model,scaler,cfg,seed,training_subjects)
            _json(directory/'scaler.json',scaler)
            split=dict(fold=fold,train_subjects=training_subjects,test_subjects=[subject],inner=[],seed=seed)
            _json(directory/'split.json',split);splits.append(split)
            history.to_csv(directory/'training_history.csv',index=False)
            history['fold']=fold;history['subject_id']=subject;histories.append(history)
            row=dict(model='cnn_fixed',fold=fold,test_subject=subject,n_train_subjects=len(training_subjects),
                n_train_windows=len(train),n_test_windows=len(test),input_samples=500,
                trainable_parameters=sum(p.numel() for p in model.parameters()),
                inner_mean_subject_macro_f1=np.nan,score_kind='uncalibrated_logit',
                seconds=time.perf_counter()-start,**result)
            folds.append(row);pc['fold']=fold;pc['subject_id']=subject;classes.append(pc)
            meta=['window_id','subject_id','recording_id','set_token','leg','sample_start',
                'sample_end_exclusive','t_start_s','t_end_exclusive_s']
            oo=w.iloc[test][meta].copy();oo['fold']=fold;oo['y_true']=y[test]
            oo['y_pred']=pred;oo['score']=scores;oo['score_kind']='uncalibrated_logit';predictions.append(oo)
            for leg,g in oo.groupby('leg'):
                values,_=metrics(g.y_true,g.y_pred,g.score)
                legs.append(dict(fold=fold,subject_id=subject,leg=leg,**values))
            pd.DataFrame(folds).to_csv(out/'partial_results.csv',index=False)
            print(f'{subject}: macro-F1={result["f1_macro"]:.3f}; {time.perf_counter()-start:.1f} s',flush=True)
            del model;gc.collect()
            if device.type=='cuda':torch.cuda.empty_cache()
        fold_table=pd.DataFrame(folds);oof=pd.concat(predictions,ignore_index=True)
        if len(oof)!=len(w) or not oof.window_id.is_unique or set(oof.window_id)!=set(w.window_id):
            raise AssertionError('Out-of-fold coverage mismatch.')
        fold_table.to_csv(out/'results.csv',index=False)
        fold_table.rename(columns={'test_subject':'subject_id'}).to_csv(out/'per_subject_metrics.csv',index=False)
        oof.to_csv(out/'out_of_fold_predictions.csv',index=False)
        pc=pd.concat(classes,ignore_index=True);pc.to_csv(out/'per_class_metrics.csv',index=False)
        pc.groupby('class_label')[['precision','recall','f1']].agg(['mean','std']).to_csv(out/'per_class_mean_sd.csv')
        leg_table=pd.DataFrame(legs);leg_table.to_csv(out/'per_subject_leg_metrics.csv',index=False)
        leg_table.pivot(index='subject_id',columns='leg',values='f1_macro').to_csv(out/'paired_leg_f1.csv')
        summary=summary_table(fold_table);summary.to_csv(out/'metrics_mean_sd.csv',index=False)
        pooled,pooled_pc=metrics(oof.y_true,oof.y_pred,np.zeros(len(oof)))
        for k in ['roc_auc','pr_auc','average_precision']:pooled.pop(k)
        _json(out/'pooled_window_metrics_secondary.json',pooled)
        pooled_pc.to_csv(out/'pooled_per_class_metrics.csv',index=False)
        _json(out/'subject_splits.json',splits)
        all_history=pd.concat(histories,ignore_index=True);all_history.to_csv(out/'training_history.csv',index=False)
        _cnn_figures(oof,fold_table,summary,all_history,out,cfg['epochs'])
        if cfg['final_refit']:
            final=out/'final_refit';final.mkdir();seed=cfg['seed']+1000
            print('Final all-subject refit: fixed settings; no additional test score.',flush=True)
            model,scaler,history=fit_network(x,y,cfg,seed,device)
            save_model(final/'model.pt',model,scaler,cfg,seed,sorted(set(groups)))
            history.to_csv(final/'training_history.csv',index=False)
            _json(final/'model_card.json',dict(purpose='Final full-data refit; no independent test score',
                input_shape=[1,500],config=cfg,scaler=scaler,
                source_model_data_sha256=source['source_model_data_sha256']))
            del model
        manifest.update(status='complete',n_outer_folds=len(folds),total_seconds=time.perf_counter()-started)
        _json(out/'run_manifest.json',manifest)
        _json(Path(output_root)/'latest_cnn_fixed.json',dict(path=str(out.resolve()),status='complete'))
    except Exception as exc:
        manifest.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        _json(out/'run_manifest.json',manifest);raise
    return out


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=Path(__file__).parent/'cnn_data')
    parser.add_argument('--outputs',type=Path,default=Path(__file__).parent/'outputs')
    args=parser.parse_args();print(run_cnn(args.data,args.outputs))
