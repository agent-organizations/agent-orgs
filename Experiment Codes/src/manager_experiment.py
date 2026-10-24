"""200 independent manager evaluations per attempt; never derive swapped outputs."""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import csv
import json
import re
import sys

from workbench_protocol import (ROOT, build_cases, prepare_inputs, randomized_execution_order,
    read_json, write_json, write_json_once, bind_run_sources, sha_text, seed_for, spawn_instance, validate_cases)
from workbench_runtime import run_parallel, require_execution_permission, validate_run_id


def parse_manager_output(text, attempt, permitted):
    """Retain malformed/forbidden responses; flag them without regenerating."""
    key = 'VERDICT' if int(attempt) == 3 else 'ENDORSEMENT'
    cleaned = text.removesuffix('</s>').removesuffix('<|im_end|>').removesuffix('<|endoftext|>').strip()
    pattern = (r'\AYOUR OWN ANS(?:W?ER):\s*(.*?)\n\s*' + key +
               r':\s*(P1|P2|BOTH|NONE)\s*\n\s*RATIONALE:\s*(.*?)\s*\Z')
    match = re.fullmatch(pattern, cleaned, flags=re.IGNORECASE | re.DOTALL)
    issues = []
    own, choice, rationale = '', '', ''
    if not match:
        issues.append('Output does not follow the required three fields in order')
    else:
        own, raw_choice, rationale = match.groups()
        own, rationale = own.strip(), rationale.strip()
        choice = {'BOTH':'Both','NONE':'None','P1':'P1','P2':'P2'}[raw_choice.upper()]
        if choice not in permitted:
            issues.append('Category is not permitted for this row')
        if not own:
            issues.append('Own answer is empty')
        if not rationale:
            issues.append('Rationale is empty')
        sentence_count = len(re.findall(r'[.!?](?:\s+|$)', rationale))
        if sentence_count < 2 or sentence_count > 5:
            issues.append('Check two-to-five sentence requirement manually')
    return {'own_answer':own, 'choice':choice, 'rationale':rationale,
            'format_valid':bool(match), 'permitted_choice':choice in permitted,
            'review_flags':issues, 'raw_output':text}


def make_jobs(cases, run_id, config):
    validate_run_id(run_id)
    jobs = []
    for c in randomized_execution_order(cases):
        jobs.append({'call_id': run_id + '_' + c['row_id'], 'kind':'manager_experiment',
            'model':c['agent_instance']['adapter_alias'], 'input_ids':c['input_ids'],
            'serialized':c['serialized'], 'temperature':config['temperature'],
            'max_tokens':config['max_output_tokens'], 'seed':c['agent_instance']['seed'],
            'stop_token_ids':config['stop_token_ids']})
    return jobs


def collect_results(cases, records, run_id):
    lookup = {r['call_id']:r for r in records}
    results = []
    for c in cases:
        call_id = run_id + '_' + c['row_id']
        r = lookup[call_id]
        parsed = parse_manager_output(r['output_text'], c['attempt'], c['allowed_choices'])
        if r['status'] != 'complete':
            parsed['review_flags'].append('Inference failed: ' + r.get('error','unknown'))
        if r['finish_reason'] == 'length':
            parsed['review_flags'].append('Output reached the generation limit; retained without repair')
        choice = parsed['choice']
        actual = c['labels_to_actual'].get(choice, choice)
        results.append({'row_id':c['row_id'], 'call_id':call_id, 'attempt':c['attempt'],
            'article_id':c['article_id'], 'article_title':c['article_title'], 'pair':c['pair'],
            'manager':c['manager'], 'orientation':c['orientation'], 'base_row_id':c['base_row_id'],
            'instance_id':c['agent_instance']['instance_id'],
            'adapter':c['agent_instance']['adapter_alias'], 'seed':c['agent_instance']['seed'],
            'P1':c['labels_to_actual']['P1'], 'P2':c['labels_to_actual']['P2'],
            'question':c['question'], 'source_outcome':c['source_outcome_offline_only'],
            **parsed, 'actual_choice':actual,
            'readable_choice':f'{choice}={actual}' if choice in ['P1','P2'] else choice,
            'status':r['status'], 'finish_reason':r['finish_reason'],
            'input_tokens':r['input_tokens'], 'output_tokens':r.get('output_tokens'),
            'queue_seconds':r.get('client_queue_seconds'),
            'generation_seconds':r.get('elapsed_seconds'), 'total_seconds':r.get('total_seconds'),
            'time_to_first_text_seconds':r.get('time_to_first_text_seconds'),
            'input_sha256':sha_text(c['serialized']), 'raw_call_path':'logs/calls/'+call_id+'.json'})
    return results


def position_comparison(rows):
    grouped = {}
    for r in rows:
        grouped.setdefault(r['base_row_id'], {})[r['orientation']] = r
    result = []
    for key, pair in grouped.items():
        a, b = pair['original'], pair['swapped']
        valid = all(r['status']=='complete' and r['format_valid'] and r['permitted_choice'] for r in [a,b])
        result.append({'base_case':key, 'article_id':a['article_id'], 'pair':a['pair'], 'manager':a['manager'],
            'original_choice':a['choice'], 'original_actual_choice':a['actual_choice'],
            'swapped_choice':b['choice'], 'swapped_actual_choice':b['actual_choice'],
            'same_actual_choice':int(a['actual_choice']==b['actual_choice']) if valid else None,
            'both_valid':valid, 'independent_model_calls':2})
    return result


def export_results(rows, summary, destination):
    """Export completed model responses and their recorded metadata."""
    # Pure-Python wheels travel with the package. No pip install or server
    # environment change is needed merely to export a future spreadsheet.
    for wheel in sorted((ROOT/'vendor/wheels').glob('*.whl')):
        if str(wheel) not in sys.path:
            sys.path.insert(0,str(wheel))
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination/'results.json', rows)
    comparisons = position_comparison(rows)
    wb = Workbook()
    wb.remove(wb.active)

    def sheet(name, data, columns):
        ws = wb.create_sheet(name)
        ws.append(columns)
        for row in data:
            ws.append([json.dumps(row.get(k),ensure_ascii=False) if isinstance(row.get(k),(list,dict)) else row.get(k) for k in columns])
        ws.freeze_panes = 'A2'
        ws.auto_filter.ref = ws.dimensions
        for c in ws[1]:
            c.font = Font(bold=True, color='FFFFFF')
            c.fill = PatternFill('solid', fgColor='17365D')
        for col in ws.columns:
            heading = col[0].value
            ws.column_dimensions[col[0].column_letter].width = 60 if heading.endswith(('rationale','own_answer','question')) else 22
            for cell in col[1:]:
                cell.alignment = Alignment(vertical='top', wrap_text=True)
        return ws

    columns = ['article_id','pair','manager','orientation','P1','P2','own_answer','readable_choice','rationale',
               'input_tokens','output_tokens','queue_seconds','generation_seconds','total_seconds',
               'time_to_first_text_seconds','source_outcome','status','finish_reason','format_valid',
               'permitted_choice','review_flags','instance_id','call_id','seed','input_sha256']
    sheet('Long_Results', rows, columns)
    wide = {}
    for r in rows:
        key = (r['article_id'], r['pair'], r['orientation'])
        w = wide.setdefault(key, {k:r[k] for k in ['article_id','pair','orientation','P1','P2','source_outcome']})
        for field in ['own_answer','readable_choice','rationale','input_tokens','output_tokens','total_seconds']:
            w[r['manager']+'_'+field] = r[field]
    wide_cols = ['article_id','pair','orientation','P1','P2','source_outcome'] + [a+'_'+f for a in 'ABCDE'
        for f in ['own_answer','readable_choice','rationale','input_tokens','output_tokens','total_seconds']]
    sheet('Wide_Results', list(wide.values()), wide_cols)
    sheet('Position_Comparison', comparisons, list(comparisons[0]))
    sheet('Method_Notes', [{'note':s} for s in [
        '200 independent model calls; swapped orientations are not copied or rewritten responses.',
        'Fresh request contexts share frozen weights. There is no additional training or carried conversation.',
        'Own answer is generated first in the same call, after exposure to all supplied materials.',
        'Original Participant Agent Interaction outcomes are reference metadata; they are not manager correctness scores.',
        'Malformed, disallowed and length-limited responses remain visible. No outcome-based repairs or quotas.',
        'Per-call totals overlap in parallel; use batch elapsed_seconds for overall wall-clock duration.',
        'Different seeds and numerical batch effects can also contribute to differences between orientations.'
    ]], ['note'])
    wb.save(destination/'Results.xlsx')
    with (destination/'Tokens_and_Runtimes.csv').open('w',newline='') as f:
        fields = ['call_id','input_tokens','output_tokens','queue_seconds','generation_seconds','total_seconds','time_to_first_text_seconds','status']
        writer = csv.DictWriter(f,fields,extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    with (destination/'All_Responses.txt').open('w') as f:
        for r in rows:
            f.write(r['call_id']+'\n'+r['raw_output']+'\n\n')


async def run_attempt(attempt, run_id, *, allow_inference=False, root=ROOT):
    require_execution_permission(allow_inference)
    root = Path(root)
    attempt = f'{int(attempt):02d}'
    validate_run_id(run_id)
    cases = prepare_inputs(attempt, root, save=False)
    # Runtime identities are specific to this run, while input text is unchanged.
    for c in cases:
        c['agent_instance'] = spawn_instance(run_id+':'+c['row_id'], c['agent_instance']['adapter_alias']).__dict__
    validate_cases(cases)
    config = read_json(root/f'Attempt_{attempt}/config.json')
    config['stop_token_ids'] = read_json(root/'config/model.json')['stop_token_ids']
    jobs = make_jobs(cases, run_id, config)
    run_root = root/f'Attempt_{attempt}/runs'/run_id
    sources = [root/f'Attempt_{attempt}/prompts/Prompt_Verbatim.txt',
        root/f'Attempt_{attempt}/config.json', root/'inputs/frozen_sessions.json',
        root/'inputs/belief_differences.json', root/'config/adapter_registry.json']
    sources += list((root.parent/'src').glob('*.py'))
    sources += list((root.parents[1]/'Parallel Setup/src').glob('*.py'))
    sources += [root/'config/model.json', root/f'Attempt_{attempt}/run_attempt_{attempt}.py', root.parent/'run.py']
    sources += list((root/'vendor/wheels').glob('*.whl'))
    bind_run_sources(run_root, sources, root.parents[1])
    write_json_once(run_root/'inputs/all_input_texts.json', [c['serialized'] for c in cases])
    write_json_once(run_root/'inputs/all_input_token_ids.json', [c['input_ids'] for c in cases])
    write_json_once(run_root/'inputs/cases.json', cases)
    summary, records = await run_parallel(jobs, run_id, run_root, allow_inference=True, root=root)
    rows = collect_results(cases, records, run_id)
    summary['source_outcomes_are_not_manager_truth_scores'] = True
    summary['decision_counts'] = dict(Counter(r['choice'] or 'unparsed' for r in rows))
    summary['cases_with_review_flags'] = sum(bool(r['review_flags']) for r in rows)
    destination = root/f'Attempt_{attempt}/deliverables'/run_id
    export_results(rows, summary, destination)
    export_exact_io(cases, records, run_id, destination)
    write_json(run_root/'data/results.json', rows)
    return rows, summary


def export_exact_io(cases, records, run_id, destination):
    destination = Path(destination)
    by_id = {r['call_id']:r for r in records}
    token_rows = []
    with (destination/'All_Exact_Input_and_Output.txt').open('w') as file:
        for case in cases:
            record = by_id[run_id+'_'+case['row_id']]
            file.write(case['row_id']+'\nINPUT\n'+case['serialized']+'\nOUTPUT\n'+record['output_text']+'\n\n')
            token_rows.append({'row_id':case['row_id'],'input_ids':case['input_ids'],
                'output_ids':record['output_token_ids'],'input_tokens':record['input_tokens'],
                'output_tokens':record.get('output_tokens')})
    write_json(destination/'All_Input_and_Output_Token_IDs.json', token_rows)
