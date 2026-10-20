"""Configure private raw input location without publishing local paths."""
from pathlib import Path
import json
import os

CODE_ROOT=Path(__file__).resolve().parent

def raw_data_dir():
    value=os.environ.get('EMG_RAW_DATA_DIR')
    config=CODE_ROOT/'config.local.json'
    if not value and config.exists():
        value=json.loads(config.read_text(encoding='utf-8')).get('raw_data_dir')
    if not value:
        raise FileNotFoundError('Set EMG_RAW_DATA_DIR or copy config.local.example.json '
            'to ignored config.local.json and set raw_data_dir to the directory '
            'containing the private participant folders. No recordings are distributed.')
    p=Path(value).expanduser()
    return (CODE_ROOT/p).resolve() if not p.is_absolute() else p.resolve()

def resolve_recording_path(value):
    # Stable audit mappings use a virtual legacy prefix; raw input may live elsewhere.
    p=Path(str(value).replace('\\','/'))
    if p.is_absolute():return p
    prefix=('Dataset','EMG - RAW data')
    if p.parts[:2]==prefix:return raw_data_dir().joinpath(*p.parts[2:])
    return CODE_ROOT/p
