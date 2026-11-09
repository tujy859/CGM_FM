"""Independent read-only comparison of official BIG IDEAs 1.1.2/1.1.3 files."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1],
                    help='CGM_FM project root (defaults to the parent of scripts/)')
parser.add_argument('--output', type=Path,
                    help='JSON evidence path; relative paths resolve under the project root')
args = parser.parse_args()
ROOT = args.root.resolve()
OUT = ((ROOT / args.output).resolve() if args.output else
       ROOT / 'reports/audits/bigideas_1_1_3_2026-11-09/four_subject_source_check.json')
OLD = json.loads((ROOT / 'datasets/BIG_IDEAs/1.1.2/results/source_manifest.json').read_text(encoding='utf-8'))
NEW = json.loads((ROOT / 'datasets/BIG_IDEAs/1.1.3/results/source_manifest.json').read_text(encoding='utf-8'))


def source(manifest, sid, modality):
    matches = [x for x in manifest['files'] if x.get('subject') == sid and x.get('modality') == modality]
    if len(matches) == 1:
        return ROOT / matches[0]['local_path']
    if not matches and modality == 'glucose' and manifest['version'] == '1.1.2':
        return ROOT / f'datasets/BIG_IDEAs/Dexcom_{sid}.csv'
    raise ValueError((sid, modality, len(matches)))


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def table(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return reader.fieldnames, rows


def times(values):
    return pd.to_datetime(pd.Series(values), errors='coerce', format='mixed')


def span(t):
    good = t.dropna()
    return {'valid_rows': len(good), 'invalid_rows': int(t.isna().sum()),
            'min': str(good.min()) if len(good) else None,
            'max': str(good.max()) if len(good) else None}


def shifts(old, new):
    a, b = times(old), times(new)
    good = a.notna() & b.notna()
    seconds = (b[good] - a[good]).dt.total_seconds()
    return {'paired_parseable_rows': int(good.sum()),
            'new_minus_old_seconds_counts': {str(int(k)): v for k, v in sorted(Counter(seconds.tolist()).items())},
            'new_minus_old_days_counts': {str(k / 86400): v for k, v in sorted(Counter(seconds.tolist()).items())}}


results = []
for sid in ['007', '013', '015', '016']:
    old_food, new_food = source(OLD, sid, 'food_log'), source(NEW, sid, 'food_log')
    old_head, old_rows = table(old_food)
    new_head, new_rows = table(new_food)
    assert len(old_rows) == len(new_rows), (sid, len(old_rows), len(new_rows))
    common = [x for x in old_head if x in new_head]
    changed = {c: sum(a[c] != b[c] for a, b in zip(old_rows, new_rows)) for c in common}
    changed = {c: n for c, n in changed.items() if n}
    timecols = [x for x in ['date', 'time_begin', 'time_end'] if x in common]
    delta = {c: shifts([r[c] for r in old_rows], [r[c] for r in new_rows]) for c in timecols}
    nutrients = ['calorie', 'total_carb', 'dietary_fiber', 'sugar', 'protein', 'total_fat']
    assert all(x in common for x in nutrients)
    nutrient_changed = {c: sum(a[c] != b[c] for a, b in zip(old_rows, new_rows)) for c in nutrients}
    nutrient_numeric_changes = {}
    numeric_missingness_changes = {}
    for c in nutrients:
        a = pd.to_numeric(pd.Series([r[c] for r in old_rows]), errors='coerce')
        b = pd.to_numeric(pd.Series([r[c] for r in new_rows]), errors='coerce')
        same = a.eq(b) | (a.isna() & b.isna())
        nutrient_numeric_changes[c] = int((~same).sum())
        numeric_missingness_changes[c] = int(a.isna().ne(b.isna()).sum())
    new_time = times([r['time_begin'] for r in new_rows])
    clock = 'time_of_day' if 'time_of_day' in new_head else 'time'
    composed = times([r['date'] + ' ' + r[clock] for r in new_rows])
    old_cgm, new_cgm = source(OLD, sid, 'glucose'), source(NEW, sid, 'glucose')
    cg = pd.read_csv(new_cgm, encoding='utf-8-sig')
    cgtime = times(cg.loc[cg['Event Type'].eq('EGV'), 'Timestamp (YYYY-MM-DDThh:mm:ss)'].tolist())
    unique_meals = new_time.dropna().drop_duplicates().sort_values()
    inspan = (unique_meals >= cgtime.min()) & (unique_meals <= cgtime.max())
    hrfile = source(NEW, sid, 'heart_rate')
    hrtime = pd.read_csv(hrfile, usecols=['datetime'], encoding='utf-8-sig')['datetime']
    hrtime = times(hrtime.tolist())
    both = new_time.notna() & composed.notna()
    results.append({
        'subject': sid, 'comparison': 'same CSV row position; no inferred reorder or timestamp shift applied',
        'old_food_path': old_food.relative_to(ROOT).as_posix(),
        'new_food_path': new_food.relative_to(ROOT).as_posix(),
        'food_rows_old': len(old_rows), 'food_rows_new': len(new_rows),
        'old_columns': old_head, 'new_columns': new_head,
        'added_columns': [x for x in new_head if x not in old_head],
        'removed_columns': [x for x in old_head if x not in new_head],
        'changed_shared_columns_counts': changed, 'timestamp_changes': delta,
        'nutrition_string_changes_by_column': nutrient_changed,
        'nutrition_numeric_value_changes_by_column': nutrient_numeric_changes,
        'nutrition_numeric_missingness_changes_by_column': numeric_missingness_changes,
        'all_six_nutrition_numeric_fields_unchanged_by_row': not any(nutrient_numeric_changes.values()),
        'all_six_nutrition_fields_byte_text_unchanged_by_row': not any(nutrient_changed.values()),
        'non_temporal_shared_fields_unchanged_by_row': all(c in {'date', 'time', 'time_of_day', 'time_begin', 'time_end'} for c in changed),
        'old_food_start_span': span(times([r['time_begin'] for r in old_rows])),
        'new_food_start_span': span(new_time),
        'new_food_separate_date_clock_span': span(composed),
        'start_equals_separate_date_clock': {'paired_parseable_rows': int(both.sum()), 'equal_rows': int((new_time[both] == composed[both]).sum())},
        'new_cgm_egv_span': span(cgtime), 'new_hr_span': span(hrtime),
        'new_exact_start_events': len(unique_meals),
        'new_exact_start_events_within_cgm_span': int(inspan.sum()),
        'old_cgm_sha256': digest(old_cgm), 'new_cgm_sha256': digest(new_cgm),
        'cgm_byte_identical_between_versions': digest(old_cgm) == digest(new_cgm),
        'old_food_sha256': digest(old_food), 'new_food_sha256': digest(new_food),
    })
    print(json.dumps({'subject': sid, 'timestamp_changes': delta, 'nutrition_numeric_changes': nutrient_numeric_changes, 'new_meals_in_cgm_span': int(inspan.sum()), 'cgm_unchanged': digest(old_cgm) == digest(new_cgm), 'start_equals_separate_clock': int((new_time[both] == composed[both]).sum()), 'parseable_clock_rows': int(both.sum())}, ensure_ascii=False), flush=True)
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps({'method': 'Independent direct CSV parse; source files read-only; source time_begin and Dexcom EGV timestamp columns used by name; no heuristic clock shifts', 'versions': ['1.1.2', '1.1.3'], 'subjects': results}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(str(OUT))
