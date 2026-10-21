"""Plot saved reviewed trajectories; no feature extraction or model training.

From the project root: py -3.13 processing/plot_report_features.py
Outputs are private report figures, not public repository assets.
"""
from pathlib import Path
import argparse
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEATURES = [
    'Haar_A4_energy', 'Haar_D4_energy', 'Haar_D3_energy', 'ZC_rate',
    'Spectral_bandwidth', 'Permutation_entropy', 'MDF', 'Sample_entropy',
    'SSC_rate', 'Spectral_entropy', 'Relative_power_150_450',
    'Relative_power_60_150', 'Relative_power_20_60', 'Haar_D1_energy',
    'AR1', 'Peak_frequency', 'Haar_D2_energy', 'Hjorth_complexity',
    'Low_high_power_ratio',
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, help='Reviewed processing run directory.')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'report/figures')
    args = parser.parse_args()
    if args.run:
        run = args.run.resolve()
    else:
        pointer = json.loads((ROOT / 'processing/outputs/latest_run.json').read_text())
        run = ROOT / pointer['path'].replace('\\', '/')
    manifest = json.loads((run / 'run_manifest.json').read_text())
    status = json.loads((ROOT / 'processing/outputs/latest_run.json').read_text())
    # The non-preview table is produced only for reviewed, eligible intervals.
    source = run / 'trajectories_by_subject_leg.csv'
    frame = pd.read_csv(source)
    dictionary = pd.read_csv(run / 'feature_dictionary.csv').set_index('feature')
    if frame.duplicated(['subject_id', 'leg', 'bin']).any():
        raise ValueError('Duplicate subject/leg/bin rows.')
    if not set(frame.leg).issubset({'L', 'R'}):
        raise ValueError('Unexpected leg labels.')
    bins = int(manifest['config']['trajectory_bins'])
    grid = (np.arange(bins) + .5) * 100 / bins
    long = frame.melt(id_vars=['subject_id', 'leg', 'bin'], value_vars=FEATURES,
                      var_name='feature', value_name='value')
    long['value'] = long['value'].replace([np.inf, -np.inf], np.nan)
    legs = long.pivot(index=['subject_id', 'bin', 'feature'], columns='leg', values='value')
    # Require defined L and R values for each feature/bin; no unilateral fallback.
    bilateral = legs[['L', 'R']].mean(axis=1).where(legs[['L', 'R']].notna().all(axis=1))
    bilateral = bilateral.rename('value').reset_index()
    summary = bilateral.groupby(['feature', 'bin']).value.agg(
        mean='mean', sd='std', n_subjects='count').reset_index()
    summary['position_percent'] = (summary.bin + .5) * 100 / bins
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    stem = out / 'selected_feature_trajectories_bilateral'

    with plt.rc_context({'font.size': 8, 'axes.titlesize': 9,
                         'axes.labelsize': 8, 'xtick.labelsize': 7,
                         'ytick.labelsize': 7, 'pdf.fonttype': 42}):
        fig, axes = plt.subplots(10, 2, figsize=(12, 18), squeeze=False)
        for index, name in enumerate(FEATURES):
            ax = axes.flat[index]
            for _, group in bilateral[bilateral.feature.eq(name)].groupby('subject_id'):
                values = group.set_index('bin').value.reindex(range(bins)).to_numpy()
                ax.plot(grid, values, color='#abb7bf', lw=.6, alpha=.55, zorder=1)
            stat = summary[summary.feature.eq(name)].set_index('bin').reindex(range(bins))
            ax.fill_between(grid, stat['mean'] - stat.sd, stat['mean'] + stat.sd,
                            color='#126782', alpha=.18, linewidth=0, zorder=2)
            ax.plot(grid, stat['mean'], color='#126782', lw=1.7, zorder=3)
            ns = stat.n_subjects.dropna()
            coverage = f'n/bin {int(ns.min())}–{int(ns.max())}'
            title = name.replace('_', ' ')
            ax.set_title(f'{index + 1}. {title}  |  {coverage}', loc='left', pad=5)
            ax.set_ylabel(str(dictionary.loc[name, 'units']))
            ax.set_xlabel('Reviewed active time (%)', labelpad=2)
            ax.set_xlim(0, 100)
            ax.set_xticks([0, 25, 50, 75, 100])
            ax.grid(alpha=.2)
            ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 4), useMathText=True)
            ax.spines[['top', 'right']].set_visible(False)
        axes.flat[-1].axis('off')
        fig.suptitle('Feature trajectories across reviewed active exercise\nBoth legs averaged within each participant',
                     fontsize=14, y=.989)
        handles = [Line2D([0], [0], color='#abb7bf', lw=.8, label='Participant mean (both legs)'),
                   Line2D([0], [0], color='#126782', lw=1.7, label='Group mean'),
                   Patch(facecolor='#126782', alpha=.18, label='Between-participant ±1 SD')]
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .956),
                   ncol=3, frameon=False, fontsize=9)
        fig.text(.06, .013,
                 'Windows → equal-weight sets within each leg → equal-weight legs within participant → group mean and sample SD.\n'
                 'Both legs required per feature/bin; missing values are not interpolated. Time progression is not a verified fatigue label.',
                 fontsize=8, va='bottom')
        fig.subplots_adjust(left=.08, right=.98, top=.925, bottom=.055, hspace=.95, wspace=.28)
        fig.savefig(stem.with_suffix('.png'), dpi=300, facecolor='white')
        fig.savefig(stem.with_suffix('.pdf'), facecolor='white')
        # A compact preview for visual checking; the report PNG remains full resolution.
        fig.savefig(stem.with_name(stem.name + '_preview').with_suffix('.png'), dpi=85, facecolor='white')
        plt.close(fig)
    bilateral.to_csv(stem.with_name(stem.name + '_participant_values').with_suffix('.csv'), index=False)
    summary.to_csv(stem.with_name(stem.name + '_summary').with_suffix('.csv'), index=False)
    provenance = {
        'source_run': run.name, 'feature_order': FEATURES, 'n_features': len(FEATURES),
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'processing_manifest_sha256': hashlib.sha256((run / 'run_manifest.json').read_bytes()).hexdigest(),
        'aggregation': 'Saved window means per set/bin -> equal set means per subject/leg/bin -> equal L/R mean (both required) -> equal subject mean and sample SD (ddof=1).',
        'interpretation': 'Descriptive reviewed active-time trajectories, including middle thirds; no training, feature selection or physiological fatigue verification.',
        'counts_are_not_selection_frequencies': True,
    }
    stem.with_name(stem.name + '_provenance').with_suffix('.json').write_text(
        json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
    print(f'Created {len(FEATURES)} panels in the requested row-wise order: {stem.with_suffix(".png")}')
    print('Participant support per feature/bin:', int(summary.n_subjects.min()), 'to', int(summary.n_subjects.max()))
    print('Vector PDF and source summaries saved alongside the PNG. No training performed.')


if __name__ == '__main__':
    main()
