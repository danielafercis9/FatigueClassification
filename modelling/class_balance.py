"""Descriptive class counts only; never fit a classifier or alter the labels."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold


def count_table(w, keys=()):
    if keys:
        table=w.groupby(list(keys)+['Label'],observed=True).size().unstack('Label',fill_value=0)
    else:
        table=pd.DataFrame([w.Label.value_counts().to_dict()])
    table=table.reindex(columns=[0,1],fill_value=0).fillna(0).astype(int)
    table.columns=['early_0','late_1']
    table['total']=table.early_0+table.late_1
    table['early_percent']=100*table.early_0/table.total
    table['late_percent']=100*table.late_1/table.total
    table['majority_to_minority_ratio']=table[['early_0','late_1']].max(axis=1)/table[['early_0','late_1']].min(axis=1).replace(0,np.nan)
    table['both_classes_present']=(table.early_0>0)&(table.late_1>0)
    return table.reset_index(drop=not bool(keys))


def audit_balance(w, output_dir, seed=20261018, inner_folds=3):
    """Summarize retained modelling windows, not independent physiological trials."""
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    tables={
        'overall':count_table(w),
        'by_subject':count_table(w,['subject_id']),
        'by_leg':count_table(w,['leg']),
        'by_subject_leg':count_table(w,['subject_id','leg']),
        'by_recording':count_table(w,['subject_id','recording_id','leg']),
    }
    outer=[];inner=[]
    for fold,subject in enumerate(sorted(w.subject_id.unique()),start=1):
        train=w.loc[w.subject_id.ne(subject)].reset_index(drop=True)
        test=w.loc[w.subject_id.eq(subject)]
        for name,part in [('train',train),('test',test)]:
            counts=count_table(part).iloc[0].to_dict()
            outer.append(dict(fold=fold,held_out_subject=subject,partition=name,
                n_subjects=part.subject_id.nunique(),**counts))
        splitter=GroupKFold(n_splits=inner_folds,shuffle=True,random_state=seed)
        for k,(a,b) in enumerate(splitter.split(train,train.Label,train.subject_id),start=1):
            assert not set(train.iloc[a].subject_id)&set(train.iloc[b].subject_id)
            for name,indices in [('inner_train',a),('inner_validation',b)]:
                part=train.iloc[indices];counts=count_table(part).iloc[0].to_dict()
                inner.append(dict(outer_fold=fold,held_out_subject=subject,inner_fold=k,
                    partition=name,n_subjects=part.subject_id.nunique(),**counts))
    tables['loso_partitions']=pd.DataFrame(outer)
    tables['inner_partitions']=pd.DataFrame(inner)
    for name,table in tables.items():table.to_csv(out/f'class_balance_{name}.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained',gridspec_kw={'width_ratios':[1,3]})
    overall=tables['overall'].iloc[0]
    bars=axes[0].bar(['Early (0)','Late (1)'],[overall.early_0,overall.late_1],color=['#3b77a8','#dd8b3b'])
    axes[0].bar_label(bars,labels=[f'{overall.early_0:,}\n{overall.early_percent:.2f}%',
        f'{overall.late_1:,}\n{overall.late_percent:.2f}%'],padding=4)
    axes[0].set(ylabel='Retained labelled windows',title='Overall counts',ylim=(0,overall[['early_0','late_1']].max()*1.22))
    subjects=tables['by_subject']
    axes[1].bar(subjects.subject_id,subjects.early_percent,label='Early (0)',color='#3b77a8')
    axes[1].bar(subjects.subject_id,subjects.late_percent,bottom=subjects.early_percent,label='Late (1)',color='#dd8b3b')
    axes[1].axhline(50,color='black',linestyle='--',linewidth=.8)
    axes[1].set(ylabel='Within-participant windows (%)',ylim=(0,100),title='Class composition by participant')
    axes[1].tick_params(axis='x',rotation=45);axes[1].legend(loc='upper center',ncol=2)
    fig.suptitle('Early / late proxy labels — descriptive balance, not independent sample counts')
    fig.savefig(out/'class_balance.png',dpi=180);plt.close(fig)
    # Data-specific interpretation, without an arbitrary threshold or significance test.
    train=tables['loso_partitions'].query("partition == 'train'")
    test=tables['loso_partitions'].query("partition == 'test'")
    text=(f'Retained windows: {int(overall.total)}; early: {int(overall.early_0)} '
        f'({overall.early_percent:.2f}%); late: {int(overall.late_1)} ({overall.late_percent:.2f}%).\n'
        f'Majority/minority count ratio: {overall.majority_to_minority_ratio:.4f}.\n'
        f'LOSO training late-class percentage range: {train.late_percent.min():.2f}–{train.late_percent.max():.2f}%.\n'
        f'LOSO held-out participant late-class percentage range: {test.late_percent.min():.2f}–{test.late_percent.max():.2f}%.\n'
        'Counts describe overlapping windows from repeated sets; participants remain the validation groups.\n'
        'No significance test, resampling, relabelling or model fitting was performed.\n'
        'Equal-duration thirds do not guarantee identical retained window counts: window alignment, timing and quality eligibility can differ.\n')
    (out/'class_balance_notes.txt').write_text(text,encoding='utf-8')
    return tables


if __name__=='__main__':
    from modelling.core import load_data
    here=Path(__file__).resolve().parent
    w,_,manifest=load_data(here/'data')
    out=here/'outputs/class_balance'/manifest['data_sha256'][:12]
    audit_balance(w,out)
    print((out/'class_balance_notes.txt').read_text(encoding='utf-8'))
