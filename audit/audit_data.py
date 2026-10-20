"""Register private raw CSVs with the original stable ID ordering; no audit reports."""
import csv
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from project_paths import raw_data_dir


def register():
    data = raw_data_dir()
    folders = sorted(p for p in data.iterdir() if p.is_dir())
    if not folders:
        raise ValueError('Expected participant subfolders inside the raw-data directory.')
    private = ROOT / 'audit/private'
    key_path = private / 'subject_key.json'
    key = json.loads(key_path.read_text(encoding='utf-8')) if key_path.exists() else {}
    if len(set(key.values())) != len(key) or any(
            not re.fullmatch(r'S\d{3}', sid) for sid in key.values()):
        raise ValueError('Invalid or repeated subject IDs in the private key.')
    records = []
    for folder in folders:
        if folder.name not in key:
            next_id = max((int(sid[1:]) for sid in key.values()), default=0) + 1
            key[folder.name] = f'S{next_id:03d}'
        files = sorted(folder.glob('*.csv'))
        if not files:
            raise ValueError('A participant folder has no CSV files; check the dataset layout.')
        for index, path in enumerate(files, 1):
            if not re.match(r'^.*?_(dr|iz)_', path.stem):
                raise ValueError('Expected original filenames containing _dr_ or _iz_.')
            with path.open(encoding='utf-8-sig', newline='') as handle:
                columns = next(csv.reader(handle), [])
            if not {'Relative_Time_s', 'Voltage_mV', 'ADC_Value'}.issubset(columns):
                raise ValueError('CSV schema differs from the original three-column recordings.')
            records.append(dict(
                recording_id=f'{key[folder.name]}_REC{index:02d}',
                source_relative_path=str(Path('Dataset') / 'EMG - RAW data' / path.relative_to(data))))
    private.mkdir(parents=True, exist_ok=True)
    key_path.write_text(json.dumps(key, indent=2) + '\n', encoding='utf-8')
    (private / 'recording_paths.json').write_text(
        json.dumps(records, indent=2) + '\n', encoding='utf-8')
    print(f'Registered {len(folders)} participants and {len(records)} recordings privately.')


if __name__ == '__main__':
    register()
