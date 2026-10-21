"""Noninteractive runner for the exact code in the clean notebook 09.

python -m modelling.band --check-only
python -m modelling.band --definitions-only
python -m modelling.band

The default performs real training; run it only in authorized private Colab.
The first two options never read participant data or fit any model.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true', help='Compile all cells without executing them.')
    parser.add_argument('--definitions-only', action='store_true', help='Run analytical synthetic checks, then stop before reading data.')
    args = parser.parse_args()
    notebook = ROOT / 'modelling/09_LFC_HFC_Fixed_RBF.ipynb'
    nb = json.loads(notebook.read_text(encoding='utf-8'))
    sources = [(i, ''.join(c['source'])) for i, c in enumerate(nb['cells'])
               if c['cell_type'] == 'code']
    compiled = []
    for index, source in sources:
        tree = ast.parse(source, filename=f'{notebook.name}:cell{index}')
        # Replace notebook-only presentation with console output; analysis unchanged.
        tree.body = [node for node in tree.body if not (
            isinstance(node, ast.ImportFrom) and node.module == 'IPython.display')]
        compiled.append((index, source, compile(tree, f'{notebook}:cell{index}', 'exec')))
    if args.check_only:
        print(f'Validated {len(compiled)} code cells. No execution or participant-data access.')
        return
    import matplotlib
    matplotlib.use('Agg')
    os.environ['EMG_PROJECT_ROOT'] = str(ROOT)
    namespace = {'__name__': '__band_notebook__', 'display': print,
                 'Image': lambda filename=None, **kwargs: f'Saved figure: {filename}'}
    for index, source, code in compiled:
        if args.definitions_only and source.lstrip().startswith('paths='):
            print('Stopped after definitions and analytical checks; no data read or model fitted.')
            return
        print(f'Executing notebook 09 cell {index}', flush=True)
        exec(code, namespace)
        if source.lstrip().startswith('stamp=') and 'OUT' in namespace:
            manifest_path = namespace['OUT'] / 'run_manifest.json'
            manifest = json.loads(manifest_path.read_text())
            manifest['noninteractive_runner_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            manifest['execution_method'] = 'modelling.band; clean notebook scientific cells, console display, Agg backend'
            manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
