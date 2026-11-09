"""Read-only subject/time/meal pairing audit for BUT CGM and BIG IDEAs.

All outputs are small audit records under reports/. No source or training file
is rewritten, and no timestamp offset is fitted to maximize overlap.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from audit_multimodal_common import (
    METRIC_NOTES, coverage_bins, pair_summary, meal_window_audit,
    merge_events_within_minutes,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/audits/multimodal_pairing_2026-11-09'
BIGIDEAS_MANIFEST = ROOT / 'datasets/BIG_IDEAs/1.1.2/results/source_manifest.json'


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def seconds(values):
    ts = pd.to_datetime(values, errors='coerce', format='mixed')
    return ts[ts.notna()].to_numpy(dtype='datetime64[s]').astype(np.int64)


def cgm_reader(frame, time_column, value_column):
    t = pd.to_datetime(frame[time_column], errors='coerce', format='mixed')
    v = pd.to_numeric(frame[value_column], errors='coerce')
    good = t.notna() & v.notna() & (v > 0)
    data = pd.DataFrame({'time': t[good], 'value': v[good]})
    conflicts = data.groupby('time')['value'].nunique()
    conflict_times = conflicts[conflicts > 1].index
    usable = data.loc[~data.time.isin(conflict_times)].drop_duplicates('time')
    return np.sort(seconds(usable.time)), {
        'cgm_source_rows': len(frame), 'cgm_valid_numeric_rows': int(good.sum()),
        'cgm_invalid_time_or_value_rows': int((~good).sum()),
        'cgm_conflicting_timestamps_excluded_from_windows': len(conflict_times),
        'cgm_exact_time_value_duplicate_rows': int(data.duplicated().sum()),
        'cgm_unique_usable_timestamps': len(usable),
    }


def empty_subject():
    return {'cg': [], 'hr': set(), 'hr_positive': set(), 'meals': [],
            'meal_recovered': [], 'meal_rows': 0, 'meal_exact_duplicates': 0,
            'meal_invalid_start_rows': 0, 'sessions': set(), 'hr_rows': 0,
            'hr_plausible_rows': 0, 'hr_positive_rows': 0, 'cgm_source_rows': 0,
            'cgm_valid_numeric_rows': 0, 'cgm_invalid_time_or_value_rows': 0,
            'cgm_conflicting_timestamps_excluded_from_windows': 0,
            'cgm_exact_time_value_duplicate_rows': 0,
            'cgm_unique_usable_timestamps': 0,
            'steps_days': set(), 'sleep_bins': set(), 'sleep_invalid_rows': 0}


def build_result(dataset, subjects, metadata, session_rows=None):
    records, meal_rows, merged_rows = [], [], []
    for sid, source in sorted(subjects.items()):
        cg = np.unique(np.concatenate(source['cg'])) if source['cg'] else np.array([], dtype=np.int64)
        hr = np.array(sorted(source['hr']), dtype=np.int64)
        hr_pos = np.array(sorted(source['hr_positive']), dtype=np.int64)
        cb = coverage_bins(cg, 5)
        meals = np.unique(np.array(source['meals'], dtype=np.int64))
        audits = meal_window_audit(meals, cg, hr, 5)
        merged = merge_events_within_minutes(meals, 30)
        merged_audits = meal_window_audit(merged, cg, hr, 5)
        recovered = np.unique(np.r_[meals, source['meal_recovered']].astype(np.int64))
        recovered_audits = meal_window_audit(recovered, cg, hr, 5)
        positive_audits = meal_window_audit(meals, cg, hr_pos, 5)
        common = set(cb) & source['hr']
        row = {
            'subject': sid, 'session_count': len(source['sessions']),
            **{k: source[k] for k in ['cgm_source_rows', 'cgm_valid_numeric_rows',
                'cgm_invalid_time_or_value_rows', 'cgm_conflicting_timestamps_excluded_from_windows',
                'cgm_exact_time_value_duplicate_rows', 'cgm_unique_usable_timestamps',
                'hr_rows', 'hr_plausible_rows', 'hr_positive_rows', 'meal_rows',
                'meal_exact_duplicates', 'meal_invalid_start_rows']},
            **pair_summary(cb, hr),
            'paired_hours_all_positive_hr': pair_summary(cb, hr_pos)['paired_person_hours'],
            'cgm_first': str(pd.Timestamp(cg.min(), unit='s')) if len(cg) else '',
            'cgm_last': str(pd.Timestamp(cg.max(), unit='s')) if len(cg) else '',
            'hr_first_occupied_bin': str(pd.Timestamp(hr.min() * 300, unit='s')) if len(hr) else '',
            'hr_last_occupied_bin': str(pd.Timestamp(hr.max() * 300, unit='s')) if len(hr) else '',
            'meal_first': str(pd.Timestamp(meals.min(), unit='s')) if len(meals) else '',
            'meal_last': str(pd.Timestamp(meals.max(), unit='s')) if len(meals) else '',
            'meal_events_within_cgm_date_span': int(((meals >= cg.min()) & (meals <= cg.max())).sum()) if len(cg) else 0,
            'meal_exact_start_events': len(meals),
            'meal_30min_cluster_events': len(merged),
            'meal_events_in_paired_5min_bins': sum(int(t // 300) in common for t in meals),
            'meal_cgm_window_eligible': sum(x['cgm_window_eligible'] for x in audits),
            'meal_cgm_and_premeal_hr_eligible': sum(x['cgm_and_premeal_hr_eligible'] for x in audits),
            'meal_isolated_cgm_and_premeal_hr_eligible': sum(x['isolated_cgm_and_premeal_hr_eligible'] for x in audits),
            'meal_30min_cluster_cgm_and_premeal_hr_eligible': sum(x['cgm_and_premeal_hr_eligible'] for x in merged_audits),
            'meal_cgm_and_all_positive_premeal_hr_eligible': sum(x['cgm_and_premeal_hr_eligible'] for x in positive_audits),
            'meal_with_source_date_time_recovery_events': len(recovered),
            'meal_recovered_cgm_and_premeal_hr_eligible': sum(x['cgm_and_premeal_hr_eligible'] for x in recovered_audits),
            'cgm_days_with_daily_steps': len(set(cg // 86400) & source['steps_days']),
            'cgm_sleep_occupied_hours': len(set(cb) & source['sleep_bins']) / 12,
            'sleep_invalid_rows': source['sleep_invalid_rows'],
        }
        if not len(cg): row['zero_pair_reason'] = 'no_usable_CGM'
        elif not len(hr): row['zero_pair_reason'] = 'no_plausible_HR'
        elif not row['paired_5min_bins']: row['zero_pair_reason'] = 'no_same_clock_bin_overlap'
        else: row['zero_pair_reason'] = ''
        row['zero_eligible_meal_reason'] = ''
        if not row['meal_cgm_and_premeal_hr_eligible']:
            if not len(meals): row['zero_eligible_meal_reason'] = 'no_verified_meal_starts'
            elif not len(cg): row['zero_eligible_meal_reason'] = 'no_usable_CGM'
            elif not row['meal_events_within_cgm_date_span']: row['zero_eligible_meal_reason'] = 'meal_dates_do_not_overlap_CGM_dates'
            elif not len(hr): row['zero_eligible_meal_reason'] = 'no_HR'
            else: row['zero_eligible_meal_reason'] = 'CGM_or_premeal_HR_window_coverage_below_threshold'
        records.append(row)
        meal_rows.extend({'subject': sid, 'meal_time': str(pd.Timestamp(a['meal_epoch_seconds'], unit='s')), **a} for a in audits)
        merged_rows.extend({'subject': sid, 'meal_time': str(pd.Timestamp(a['meal_epoch_seconds'], unit='s')), **a} for a in merged_audits)
    additive = [k for k in records[0] if k not in {
        'subject', 'session_count', 'fraction_cgm_coverage_with_wearable',
        'longest_paired_run_hours', 'cgm_first', 'cgm_last', 'zero_pair_reason',
        'hr_first_occupied_bin', 'hr_last_occupied_bin', 'meal_first', 'meal_last',
        'zero_eligible_meal_reason'}]
    totals = {k: sum(r[k] for r in records) for k in additive}
    totals['fraction_cgm_coverage_with_wearable'] = totals['paired_5min_bins'] / totals['cgm_occupied_5min_bins'] if totals['cgm_occupied_5min_bins'] else None
    summary = {
        'dataset': dataset, 'date': '2026-11-09', 'training_changed': False,
        'metric_notes': METRIC_NOTES, 'participants_in_study_source': len(records),
        'participants_with_cgm': sum(x['cgm_unique_usable_timestamps'] > 0 for x in records),
        'participants_with_hr': sum(x['wearable_occupied_5min_bins'] > 0 for x in records),
        'participants_with_paired_cgm_hr': sum(x['paired_5min_bins'] > 0 for x in records),
        'participants_with_meal_start_events': sum(x['meal_exact_start_events'] > 0 for x in records),
        'participants_with_meals_in_cgm_date_span': sum(x['meal_events_within_cgm_date_span'] > 0 for x in records),
        'participants_with_nonoverlapping_meal_dates': [x['subject'] for x in records if x['zero_eligible_meal_reason'] == 'meal_dates_do_not_overlap_CGM_dates'],
        'participants_with_eligible_meal_windows': sum(x['meal_cgm_and_premeal_hr_eligible'] > 0 for x in records),
        'participants_with_isolated_eligible_meal_windows': sum(x['meal_isolated_cgm_and_premeal_hr_eligible'] > 0 for x in records),
        'participants_with_cgm_and_watch_steps_days': sum(x['cgm_days_with_daily_steps'] > 0 for x in records),
        'participants_with_cgm_and_watch_sleep': sum(x['cgm_sleep_occupied_hours'] > 0 for x in records),
        'totals': totals, **metadata,
    }
    target = OUT / dataset
    write_json(target / 'summary.json', summary)
    write_csv(target / 'per_subject.csv', records)
    write_csv(target / 'meal_events.csv', meal_rows)
    write_csv(target / 'meal_30min_cluster_sensitivity.csv', merged_rows)
    if session_rows is not None:
        write_csv(target / 'per_session.csv', session_rows)
        summary['sessions_total'] = len(session_rows)
        summary['sessions_with_cgm_hr_pair'] = sum(r['paired_5min_bins'] > 0 for r in session_rows)
        write_json(target / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def audit_but():
    subjects = defaultdict(empty_subject)
    base = ROOT / 'datasets/BUT_CGM/1.0.0/source'
    hrbase = ROOT / 'datasets/BUT_CGM/1.0.0/results/source'
    sessions = []
    session_ids = sorted({x.name for x in base.iterdir() if x.is_dir()} | {x.name for x in hrbase.iterdir() if x.is_dir()})
    for i, session in enumerate(session_ids, 1):
        folder = base / session
        sub = subjects[session[:3]]
        sub['sessions'].add(session)
        cf = folder / f'{session}_CGM.csv'
        cg = np.array([], dtype=np.int64)
        if cf.exists():
            cg, qc = cgm_reader(pd.read_csv(cf, sep=';', encoding='utf-8-sig'), 'Timestamp', 'Glycemia')
            sub['cg'].append(cg)
            for k, n in qc.items(): sub[k] += n
        hr, hr_pos = set(), set()
        for f in sorted((hrbase / session).glob('*_HR.csv')):
            day = int(np.datetime64(f.name.split('_')[1], 's').astype(np.int64))
            for d in pd.read_csv(f, sep=';', usecols=['HR', 'QRS'], chunksize=250000, encoding='utf-8-sig'):
                h = pd.to_numeric(d.HR, errors='coerce').to_numpy()
                q = pd.to_numeric(d.QRS, errors='coerce').to_numpy()
                usable_q = np.isfinite(q) & (q >= 0) & (q < 86400000)
                positive = usable_q & np.isfinite(h) & (h > 0)
                plausible = positive & (h >= 30) & (h <= 220)
                sub['hr_rows'] += len(d)
                sub['hr_positive_rows'] += int(positive.sum())
                sub['hr_plausible_rows'] += int(plausible.sum())
                hr.update(np.unique((day + np.floor(q[plausible] / 1000).astype(np.int64)) // 300).tolist())
                hr_pos.update(np.unique((day + np.floor(q[positive] / 1000).astype(np.int64)) // 300).tolist())
        sub['hr'].update(hr)
        sub['hr_positive'].update(hr_pos)
        mf = folder / f'{session}_meals.csv'
        if mf.exists():
            d = pd.read_csv(mf, sep=';', encoding='utf-8-sig')
            sub['meal_rows'] += len(d)
            sub['meal_exact_duplicates'] += int(d.duplicated().sum())
            d = d.drop_duplicates()
            mt = seconds(d.Time)
            sub['meal_invalid_start_rows'] += len(d) - len(mt)
            sub['meals'].extend(mt.tolist())
        sf = folder / f'{session}_daily_steps_summary.csv'
        if sf.exists():
            d = pd.read_csv(sf, sep=';', encoding='utf-8-sig')
            value = pd.to_numeric(d.Steps, errors='coerce')
            sub['steps_days'].update((seconds(d.loc[value.notna() & (value >= 0), 'Date']) // 86400).tolist())
        sf = folder / f'{session}_sleep.csv'
        if sf.exists():
            d = pd.read_csv(sf, sep=';', encoding='utf-8-sig')
            for row in d.to_dict('records'):
                start = pd.to_datetime(str(row['Date']) + ' ' + str(row['Sleep_Start']), errors='coerce')
                end = pd.to_datetime(str(row['Date']) + ' ' + str(row['Sleep_End']), errors='coerce')
                if pd.isna(start) or pd.isna(end):
                    sub['sleep_invalid_rows'] += 1
                    continue
                if start > end: start -= pd.Timedelta(days=1)
                if not 0 < (end - start).total_seconds() <= 86400:
                    sub['sleep_invalid_rows'] += 1
                    continue
                # Date denotes wake date; mark covered grid positions, not sleep stages.
                a, b = int(start.value // 10**9 // 300), int(end.value // 10**9 // 300)
                sub['sleep_bins'].update(range(a, b))
        sessions.append({'session': session, 'subject': session[:3], **pair_summary(coverage_bins(cg, 5), list(hr))})
        if i % 10 == 0: print(f'BUT read {i} sessions', flush=True)
    build_result('but', subjects, {
        'clock_status': 'Official within-participant same local clock, participant-specific whole-week date shifts; explicit daily QRS origin at 00:00, 1000 Hz. No artificial time offset applied.',
        'source_page': 'https://physionet.org/content/but-cgm/1.0.0/',
        'device_type': 'Bittium Faros chest ECG recorder supplies HR; assorted sports watches supply daily steps, sleep and activity, so HR is not smartwatch PPG.',
        'meal_definition': 'Published meals.csv already aggregates food items within 10-minute windows. Exact-start events counted, with separate 30-minute consecutive-gap clustering sensitivity.',
        'meal_nutrition': 'Official meal-level calories, protein, carbohydrate, sugar and fat available; audit did not filter food by nutritional content.',
        'cgm_cadence_minutes': 5,
        'limitations': ['No waveform or clock-calibration study performed.', 'Daily steps cannot be used before the day finishes; whole-day totals would leak future activity into meal prediction.', 'Sleep intervals are prior-night context, not time-resolved wearable physiology.', 'Pair hours are occupied-grid estimates, not continuous valid wear hours.'],
    }, sessions)


def audit_big():
    subjects = defaultdict(empty_subject)
    base = ROOT / 'datasets/BIG_IDEAs'
    manifest = json.loads(BIGIDEAS_MANIFEST.read_text(encoding='utf-8'))
    headerless_evidence = {}
    recovered_rows = []
    hr_precision = {}
    for sid in [f'{s:03d}' for s in range(1, 17)]:
        sub = subjects[sid]
        sub['sessions'].add(sid)
        glucose_entries = [r for r in manifest['files']
                           if r.get('subject') == sid and r.get('modality') == 'glucose']
        if len(glucose_entries) > 1:
            raise ValueError(f'Multiple glucose source files for subject {sid}')
        glucose_path = (ROOT / glucose_entries[0]['local_path']
                        if glucose_entries else base / f'Dexcom_{sid}.csv')
        d = pd.read_csv(glucose_path, encoding='utf-8-sig')
        cgm = d.loc[d['Event Type'].eq('EGV')]
        cg, qc = cgm_reader(cgm, 'Timestamp (YYYY-MM-DDThh:mm:ss)', 'Glucose Value (mg/dL)')
        sub['cg'].append(cg)
        for k, n in qc.items(): sub[k] += n
        for entry in [r for r in manifest['files'] if r.get('subject') == sid]:
            path = ROOT / entry['local_path']
            if entry['modality'] == 'heart_rate':
                precision = {'rows': 0, 'rows_with_nonzero_seconds': 0}
                for d in pd.read_csv(path, chunksize=250000, encoding='utf-8-sig'):
                    d.columns = d.columns.str.strip()
                    t = pd.to_datetime(d.datetime, errors='coerce', format='mixed')
                    h = pd.to_numeric(d.hr, errors='coerce')
                    positive = t.notna() & h.notna() & (h > 0)
                    plausible = positive & h.between(30, 220)
                    precision['rows'] += len(d)
                    precision['rows_with_nonzero_seconds'] += int((t.dt.second.fillna(0) != 0).sum())
                    sub['hr_rows'] += len(d)
                    sub['hr_positive_rows'] += int(positive.sum())
                    sub['hr_plausible_rows'] += int(plausible.sum())
                    sub['hr'].update(coverage_bins(seconds(t[plausible])).tolist())
                    sub['hr_positive'].update(coverage_bins(seconds(t[positive])).tolist())
                hr_precision[sid] = precision
            elif entry['modality'] == 'food_log':
                with path.open(encoding='utf-8-sig', newline='') as f: first = next(csv.reader(f))
                has_header = first[0] == 'date'
                d = pd.read_csv(path, header=0 if has_header else None, encoding='utf-8-sig')
                sub['meal_rows'] += len(d)
                sub['meal_exact_duplicates'] += int(d.duplicated().sum())
                d = d.drop_duplicates()
                if has_header:
                    mt = pd.to_datetime(d.time_begin, errors='coerce', format='mixed')
                    clock_column = 'time' if 'time' in d.columns else 'time_of_day'
                    # Inspect only missing starts; other published clock text has
                    # inconsistent formats and is not needed to replace valid starts.
                    separate = pd.Series(pd.NaT, index=d.index, dtype='datetime64[ns]')
                    missing = mt.isna()
                    separate.loc[missing] = pd.to_datetime(d.loc[missing, 'date'].astype(str) + ' ' + d.loc[missing, clock_column].astype(str), errors='coerce', format='mixed')
                    recovery = mt.isna() & separate.notna()
                    sub['meal_recovered'].extend(seconds(separate[recovery]).tolist())
                    recovered_rows.extend({'subject': sid, 'source_row_index_zero_based': int(idx), 'date': str(d.loc[idx, 'date']), 'time': str(d.loc[idx, clock_column]), 'recovered_time': str(separate.loc[idx]), 'primary_analysis': 'excluded; source date/time reconstruction sensitivity only'} for idx in d.index[recovery])
                else:
                    # Confirm ONLY redundant time fields, not any unknown nutrient column.
                    assert len(d.columns) == 11 and sid == '003'
                    mt = pd.to_datetime(d[2], errors='coerce', format='mixed')
                    separate = pd.to_datetime(d[0].astype(str) + ' ' + d[1].astype(str), errors='coerce', format='mixed')
                    assert mt.notna().all() and (mt == separate).all()
                    headerless_evidence = {'subject': sid, 'rows': len(d), 'columns': 11, 'third_column_parseable_rows': int(mt.notna().sum()), 'third_column_equals_date_plus_clock_rows': int((mt == separate).sum()), 'scope': 'Time identity only; no nutrient/food schema mapping or source edits.'}
                sub['meal_invalid_start_rows'] += int(mt.isna().sum())
                sub['meals'].extend(seconds(mt).tolist())
        print(f'BIG read {sid}', flush=True)
    write_json(OUT / 'bigideas/headerless_003_time_evidence.json', headerless_evidence)
    write_json(OUT / 'bigideas/date_time_recovery_sensitivity.json', recovered_rows)
    nonoverlapping = []
    for sid, sub in sorted(subjects.items()):
        cg = np.concatenate(sub['cg']) if sub['cg'] else np.array([], dtype=np.int64)
        meals = np.asarray(sub['meals'], dtype=np.int64)
        if len(cg) and len(meals) and not ((meals >= cg.min()) & (meals <= cg.max())).any():
            nonoverlapping.append(sid)
    limitations = [
        'Some HR files retain only minute precision despite multiple rows per minute; timestamp precision varies by subject, so no universal second-level alignment claim.',
    ]
    if headerless_evidence:
        limitations.append('Subject 003 has 11 columns without a header versus the usual 14; only redundant date/time fields confirmed, unknown nutrient columns not mapped.')
    if nonoverlapping:
        limitations.append(f"Subjects {'/'.join(nonoverlapping)} have food dates outside their CGM date ranges; no heuristic shift applied, and those meals cannot currently be paired.")
    if recovered_rows:
        limitations.append('Primary excludes missing time_begin values; explicit date+time reconstruction shown only as sensitivity.')
    limitations.extend([
        'Logged intake events may include snacks or drinks and may be incomplete; meal counts are operational.',
        'Source values retained; CGM duplicate conflict timestamps excluded from target-window eligibility.',
    ])
    build_result('bigideas', subjects, {
        'clock_status': 'Matched published naive wall-clock datetimes with no fitted shift. Official release states dates were shifted consistently, including food logs. UTC offset, clock calibration and DST conversion are not explicitly documented; absolute synchronization remains provisional.',
        'source_page': manifest['source_page'],
        'version': manifest['version'],
        'device_type': 'Empatica E4 wristband HR plus Dexcom G6 CGM; no local steps/sleep result table.',
        'meal_definition': 'Food-item rows grouped by subject and identical verified start timestamp; not all rows are meals. Thirty-minute grouping is separate sensitivity.',
        'headerless_003': headerless_evidence,
        'missing_start_recovery_rows': recovered_rows,
        'hr_timestamp_precision_by_subject': hr_precision,
        'cgm_cadence_minutes': 5,
        'limitations': limitations,
    })


def audit_nutrition(dataset):
    """Optional eligibility only: every component row has four named nutrients."""
    events = defaultdict(lambda: {'rows': 0, 'schema_known': True, 'all_component_nutrients_present': True})
    if dataset == 'but':
        sources = [(f.parent.name[:3], f, 'Time', ['Calories', 'Carbs', 'Protein', 'Fats']) for f in (ROOT / 'datasets/BUT_CGM/1.0.0/source').glob('*/*_meals.csv')]
    else:
        manifest = json.loads(BIGIDEAS_MANIFEST.read_text(encoding='utf-8'))
        sources = [(x['subject'], ROOT / x['local_path'], 'time_begin', ['calorie', 'total_carb', 'protein', 'total_fat']) for x in manifest['files'] if x['modality'] == 'food_log']
    for sid, path, time_col, nutrients in sources:
        unknown = dataset == 'bigideas' and sid == '003'
        frame = pd.read_csv(path, sep=';' if dataset == 'but' else ',', header=None if unknown else 0, encoding='utf-8-sig').drop_duplicates()
        tt = pd.to_datetime(frame[2 if unknown else time_col], errors='coerce', format='mixed')
        vv = None if unknown else frame[nutrients].apply(pd.to_numeric, errors='coerce')
        for index in frame.index[tt.notna()]:
            key = (sid, int(tt.loc[index].value // 10**9))
            row = events[key]
            row['rows'] += 1
            row['schema_known'] &= not unknown
            row['all_component_nutrients_present'] &= bool(not unknown and vv.loc[index].notna().all() and vv.loc[index].ge(0).all())
    qc = pd.read_csv(OUT / dataset / 'meal_events.csv', dtype={'subject': str})
    eligible = {(str(r.subject), int(r.meal_epoch_seconds)): bool(r.cgm_and_premeal_hr_eligible) for r in qc.itertuples()}
    rows = [{'subject': sid, 'meal_epoch_seconds': epoch, **record, 'cgm_and_premeal_hr_eligible': eligible.get((sid, epoch), False)} for (sid, epoch), record in sorted(events.items())]
    complete = [r for r in rows if r['all_component_nutrients_present']]
    complete_eligible = [r for r in complete if r['cgm_and_premeal_hr_eligible']]
    summary = {
        'dataset': dataset, 'definition': 'All exact-start food component rows must have nonmissing nonnegative numeric energy, carbohydrate, protein and fat in officially named columns. Zero retained; no imputation. This checks completeness, not nutritional accuracy.',
        'recorded_exact_start_events': len(rows),
        'events_with_known_nutrient_schema': sum(r['schema_known'] for r in rows),
        'events_with_complete_four_nutrients': len(complete),
        'eligible_cgm_hr_events_with_complete_four_nutrients': len(complete_eligible),
        'participants_with_complete_nutrients_and_eligible_windows': len({r['subject'] for r in complete_eligible}),
        'unknown_schema_subjects': sorted({r['subject'] for r in rows if not r['schema_known']}),
        'training_changed': False,
    }
    write_json(OUT / dataset / 'nutrition_completeness.json', summary)
    write_csv(OUT / dataset / 'nutrition_event_flags.csv', rows)


def main():
    global BIGIDEAS_MANIFEST, OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['both', 'but', 'bigideas'], default='both')
    parser.add_argument('--bigideas-manifest', type=Path, default=BIGIDEAS_MANIFEST,
                        help='BIG IDEAs source manifest; relative paths resolve from the project root')
    parser.add_argument('--output-dir', type=Path, default=OUT,
                        help='Audit output root; dataset subdirectories are created beneath it')
    args = parser.parse_args()
    BIGIDEAS_MANIFEST = (ROOT / args.bigideas_manifest).resolve()
    OUT = (ROOT / args.output_dir).resolve()
    if args.dataset in ['both', 'but']:
        audit_but()
        audit_nutrition('but')
    if args.dataset in ['both', 'bigideas']:
        audit_big()
        audit_nutrition('bigideas')


if __name__ == '__main__':
    main()
