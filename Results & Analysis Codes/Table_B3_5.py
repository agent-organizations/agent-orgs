#!/usr/bin/env python3


"""Reproduce Table B3 5 from the six locked workbooks.

Run: python Table_B3_5.py --inputs "YOUR PATHWAY/Final Output/inputs"
This is a self-contained program; no other exhibit script is imported.
Supply protected workbooks and reviewed codes with --inputs (see README.md).
All 200 outcomes per workbook are loaded; stated subgroup denominators apply.
No saved numerical results are used as inputs.
"""


from __future__ import annotations

import argparse

import csv

import hashlib

import itertools

import json

import math

import os

import platform

import re

import tempfile

import unicodedata

from collections import Counter

from decimal import Decimal, ROUND_HALF_UP

from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'appendix_python_mpl'))

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

from matplotlib.cm import ScalarMappable

from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap, to_hex

from matplotlib.lines import Line2D

from matplotlib.patches import Patch

import numpy as np

import openpyxl

from scipy import stats

COMPONENTS = ('MAGNITUDE', 'ACTOR', 'MECHANISM')

REQUIRED = {'AB': ('MAGNITUDE',), 'AC': ('ACTOR',), 'AD': ('MECHANISM',), 'AE': COMPONENTS}

MODELS = ('Mistral', 'Qwen')

MODEL_ORDER = ('1a', '2a', '3a', '1b', '2b', '3b')

ATTEMPT_ORDER = ('1a', '1b', '2a', '2b', '3a', '3b')

GROUP_NAMES = {1: 'All three beliefs match',
               2: 'One shared belief; other participant shares none',
               3: 'One shared belief; other participant shares two',
               4: 'Two shared beliefs; other participant shares one'}

GROUP_LABELS = ['Group 1:\nFocal shares all 3\nOther shares 0 or 2\nN=80',
                'Group 2:\nFocal shares 1\nOther shares none\nN=30',
                'Group 3:\nFocal shares 1\nOther shares 2\nN=30',
                'Group 4:\nFocal shares 2\nOther shares 1\nN=60']

COLORS = {1: '#2a78d6', 2: '#eb6834', 3: '#6250d6'}

R, C, O = 'Retained initial answer', 'Conceded', 'Others'

GROUPS = [R, C, O]

REASONS = [('accuracy', 'Accuracy / article evidence', '#eb6834'),
           ('debate', 'Debate performance', '#a8441b'),
           ('belief', 'Belief similarity', '#f2a47f')]

PARTNER_GROUPS = ['Manager has the original belief profile',
                  "Manager shares all of partner's altered beliefs",
                  'Partial similarity: A retained initial answer',
                  'Partial similarity: A conceded',
                  'Partial similarity: Others/Unresolved']

LABELS = ['Manager A\n(original belief profile)',
          'Manager shares all of\npartner’s altered beliefs',
          'Partial similarity:\nA retained initial answer',
          'Partial similarity:\nA conceded',
          'Partial similarity:\nOthers/Unresolved']

UNIT_KEY = ('article', 'pair', 'manager', 'orientation')

def require(condition, message):
    """Checks remain active even if Python is invoked with -O."""
    if not condition:
        raise ValueError(message)

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def write_csv(path, rows):
    require(bool(rows), f'No rows for {path}')
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def parse_reference(text):
    parts = re.split(r'\b(MAGNITUDE|ACTOR|MECHANISM):\s*', text)
    require(not parts[0].strip() and len(parts) > 1, 'Unexpected assigned-reference format')
    return {parts[i]: re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', parts[i + 1])).strip().casefold()
            for i in range(1, len(parts), 2)}

def n_shared(a, b):
    return sum(a[c] == b[c] for c in COMPONENTS)

def classify_group(profiles, article, manager, pair):
    """Belief groups are defined before, and independently of, the choice."""
    manager_profile = profiles[article, manager]
    matches = {agent: n_shared(manager_profile, profiles[article, agent]) for agent in pair}
    distance = 3 - n_shared(profiles[article, pair[0]], profiles[article, pair[1]])
    if 3 in matches.values():
        require(list(matches.values()).count(3) == 1, 'Ambiguous exact match')
        group, target = 1, 3
    elif set(matches.values()) == {0, 1}:
        group, target = 2, 1
    elif set(matches.values()) == {1, 2} and distance == 3:
        group, target = 3, 1
    elif set(matches.values()) == {1, 2} and distance == 1:
        group, target = 4, 2
    else:
        raise ValueError(f'Unclassified belief group: {(article, manager, pair, matches)}')
    focal = next(agent for agent, count in matches.items() if count == target)
    other = next(agent for agent in pair if agent != focal)
    expected = 1 if manager in pair else 2 if manager == 'E' else 3 if pair == 'AE' else 4
    require(group == expected, 'Independent agent/pair group check failed')
    return group, focal, other, matches[focal], matches[other]

def decode_choice(row, model, attempt):
    header = 'Endorsement' if attempt < 3 else 'Manager choice' if model == 'Mistral' else 'Verdict'
    raw = row[header]
    require(isinstance(raw, str) and raw.strip(), 'Missing decision')
    decision = raw.strip()
    if decision in ('Both', 'None'):
        require(attempt == 3, 'Nonbinary decision outside Experiment 3')
        selected = set(row['Pair']) if decision == 'Both' else set()
    else:
        match = re.fullmatch(r'P([12])\s*=\s*([A-E])', decision)
        require(match is not None, f'Unrecognized decision: {decision}')
        require(row['P' + match[1]] == match[2], 'Position and participant identity disagree')
        if attempt < 3:
            require(row['Endorsed agent'].strip() == match[2], 'Endorsement fields disagree')
        selected = {match[2]}
        decision = match[2]
    return raw, decision, selected

def load_data(inputs):
    """Validate and retain all 1,200 source decisions; attach frozen behavior codes."""
    integrity = read_json(inputs / 'input_checksums.json')
    for relative, expected_sha in integrity.items():
        require(sha256(inputs / relative) == expected_sha, f'Input changed: {relative}')
    manifest = read_json(inputs / 'manifest.json')
    require({(f['model'], f['attempt']) for f in manifest['files']} ==
            set(itertools.product(MODELS, (1, 2, 3))), 'Need the six locked experiments')
    discussion_rows = read_csv(inputs / 'discussion_codes.csv')
    participant_rows = read_csv(inputs / 'participant_behavior_codes.csv')
    discussions = {(int(r['article']), r['pair']): r for r in discussion_rows}
    behaviors = {(int(r['article']), r['pair'], r['participant']): r for r in participant_rows}
    require(len(discussions) == len(discussion_rows) == 20, 'Discussion coding keys changed')
    require(len(behaviors) == len(participant_rows) == 40, 'Participant coding keys changed')
    require(Counter(r['behavior'] for r in participant_rows) == {R: 17, C: 15, O: 8},
            'Participant behavior counts changed')
    expected_grid = set(itertools.product(range(1, 6), REQUIRED, 'ABCDE', ('original', 'swapped')))
    output, source_checks, common_profiles = [], [], {}
    for entry in sorted(manifest['files'], key=lambda f: (MODELS.index(f['model']), f['attempt'])):
        model, attempt = entry['model'], entry['attempt']
        experiment = f'{attempt}' + ('a' if model == 'Mistral' else 'b')
        path = inputs / 'workbooks' / entry['filename']
        require(sha256(path) == entry['sha256'], f'Locked workbook changed: {path}')
        workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
        try:
            values = list(workbook['Results'].values)
        finally:
            workbook.close()
        headers = [str(h).strip() if h is not None else '' for h in values[0]]
        rows = [(i, dict(zip(headers, row))) for i, row in enumerate(values[1:], 2) if row[0] is not None]
        require(len(rows) == 200, f'{experiment}: expected all 200 outcomes')
        require(len({r['Outcome ID'] for _, r in rows}) == 200, 'Duplicate outcome ID')
        grid = {(int(r['Article']), r['Pair'], r['Manager'], r['Orientation']) for _, r in rows}
        require(grid == expected_grid, f'{experiment}: incomplete design grid')
        profiles = {}
        for _, r in rows:
            if r['Pair'] == 'AE':
                key = int(r['Article']), r['Manager']
                profile = parse_reference(r['Assigned-belief reference answer'])
                require(set(profile) == set(COMPONENTS), 'Incomplete combined profile')
                require(key not in profiles or profiles[key] == profile, 'Inconsistent within-file profile')
                require(key not in common_profiles or common_profiles[key] == profile, 'Profiles differ across files')
                profiles[key] = common_profiles[key] = profile
        require(len(profiles) == 25, 'Missing article/manager profiles')
        own_header = next(h for h in headers if 'manipulation check' in h.lower())
        for row_number, r in rows:
            article, pair, manager = int(r['Article']), r['Pair'], r['Manager']
            p1, p2 = r['P1'], r['P2']
            require({p1, p2} == set(pair), 'Participant identities disagree with pair')
            require((p1 == 'A') == (r['Orientation'] == 'original'), 'Orientation disagrees with position')
            requested = parse_reference(r['Assigned-belief reference answer'])
            require(set(requested) == set(REQUIRED[pair]), 'Wrong requested components')
            require(all(requested[c] == profiles[article, manager][c] for c in requested), 'Assigned reference mismatch')
            group, focal, other, focal_matches, other_matches = classify_group(profiles, article, manager, pair)
            raw, decision, selected = decode_choice(r, model, attempt)
            behavior_a = behaviors[article, pair, 'A']['behavior']
            behavior_partner = behaviors[article, pair, pair[1]]['behavior']
            discussion = discussions[article, pair]
            require(behavior_a == discussion['group'], 'A and discussion codes disagree')
            require(isinstance(r['Manager rationale'], str) and r['Manager rationale'].strip(), 'Missing rationale')
            altered = [c for c in COMPONENTS if profiles[article, 'A'][c] != profiles[article, pair[1]][c]]
            if profiles[article, manager] == profiles[article, 'A']:
                partner_group = 0
            elif all(profiles[article, manager][c] == profiles[article, pair[1]][c] for c in altered):
                partner_group = 1
            else:
                partner_group = 2 + GROUPS.index(behavior_a)
            output.append(dict(model=model, attempt=attempt, experiment=experiment,
                outcome_id=r['Outcome ID'], article=article, pair=pair, manager=manager,
                orientation=r['Orientation'], p1=p1, p2=p2, partner=pair[1], group=group,
                focal=focal, other=other, focal_matches=focal_matches, other_matches=other_matches,
                recorded_choice=raw, choice=decision, selected_agents='|'.join(sorted(selected)),
                supports_A=int('A' in selected), supports_partner=int(pair[1] in selected),
                supports_focal=int(focal in selected), supports_P1=int(p1 in selected),
                focal_only=int(selected == {focal}), other_only=int(selected == {other}),
                A_only=int(selected == {'A'}), partner_only=int(selected == {pair[1]}),
                both=int(decision == 'Both'), none=int(decision == 'None'),
                behavior_A=behavior_a, behavior_partner=behavior_partner,
                discussion_subtype=discussion['subtype'],
                revised_before_agreement=int(behavior_a == C or discussion['subtype'] == 'Joint revision'),
                partner_group=partner_group, partial=int(group in (3, 4)),
                assigned_reference=r['Assigned-belief reference answer'], own_answer=r[own_header],
                rationale=r['Manager rationale'], rationale_words=len(r['Manager rationale'].split()),
                source_file=entry['filename'], source_sha256=entry['sha256'], source_row=row_number))
        subset = [r for r in output if r['experiment'] == experiment]
        require(Counter(r['group'] for r in subset) == {1: 80, 2: 30, 3: 30, 4: 60}, 'Wrong belief group sizes')
        require(Counter(r['partner_group'] for r in subset) == {0: 40, 1: 70, 2: 36, 3: 30, 4: 24},
                'Wrong partner-support group sizes')
        source_checks.append(dict(experiment=experiment, filename=entry['filename'], sha256=entry['sha256'],
                                  outcomes=200, original=100, swapped=100, complete_design_grid=True))
    require(len(output) == 1200, 'Expected 1,200 total decisions')
    return output, dict(reference_set=manifest['reference_set_id'], sources=source_checks,
                        total_outcomes=1200, input_checksums_verified=len(integrity),
                        behavior_coding='Frozen prior semantic review of histories in locked rationales',
                        no_old_results_used=True, overleaf_modified=False)

def select(data, experiment):
    rows = [r for r in data if r['experiment'] == experiment]
    require(len(rows) == 200, f'{experiment}: a source row was lost')
    return rows

def unit_key(row):
    return tuple(row[k] for k in UNIT_KEY)

def tex_row(cells):
    return ' & '.join(str(c) for c in cells) + r' \\'

def emit(exhibit, records, inputs, output, tex_rows=None):
    """Export fresh calculations plus the current manuscript title/notes."""
    write_csv(output / f'{exhibit}.csv', records)
    write_json(output / f'{exhibit}.json', records)
    metadata = read_json(inputs / 'exhibits.json')[exhibit]
    template = (inputs / 'templates' / metadata['template']).read_text(encoding='utf-8')
    if tex_rows is not None:
        require(template.count('@@ROWS@@') == 1, 'Table template must contain one row placeholder')
        template = template.replace('@@ROWS@@', '\n'.join(tex_rows) + '\n')
    require('@@ROWS@@' not in template, 'Unfilled table template')
    (output / f'{exhibit}.tex').write_text(template, encoding='utf-8')
    match = re.search(r'\\exnote\{(.*)\}\s*\\label', template, re.DOTALL)
    if match:
        (output / f'{exhibit}_notes.tex').write_text(r'\exnote{' + match[1].strip() + '}\n', encoding='utf-8')

def table_b34(data, inputs, output):
    """Matched changes in binary support for A, holding article/pair/manager/order."""
    names = {R: 'A retained initial answer', C: 'A conceded', O: 'Others/Unresolved', 'Total': 'Total'}
    records = [{'A_behavior': names[group], 'N': 0} for group in [R, C, O, 'Total']]
    transitions = []
    for model, suffix in [('Mistral', 'a'), ('Qwen', 'b')]:
        for before, after in [(1, 2), (2, 3)]:
            previous = {unit_key(r): r for r in select(data, f'{before}{suffix}')}
            current = {unit_key(r): r for r in select(data, f'{after}{suffix}')}
            require(previous.keys() == current.keys(), 'Matched experiment keys differ')
            label = f'{model} {before}{suffix} to {after}{suffix}'
            matched = []
            for key in sorted(previous):
                a, b = previous[key], current[key]
                require(a['behavior_A'] == b['behavior_A'], 'Discussion classification changed across attempts')
                change = b['supports_A'] - a['supports_A']
                matched.append(dict(model=model, transition=label,
                    **{field: a[field] for field in UNIT_KEY}, behavior_A=a['behavior_A'],
                    before_support_A=a['supports_A'], after_support_A=b['supports_A'], change=change))
            transitions.extend(matched)
            for result, group, expected_n in zip(records, [R, C, O, 'Total'], [90, 70, 40, 200]):
                rows = matched if group == 'Total' else [r for r in matched if r['behavior_A'] == group]
                require(len(rows) == expected_n, 'Wrong matched transition denominator')
                toward = sum(r['change'] == 1 for r in rows)
                away = sum(r['change'] == -1 for r in rows)
                unchanged = sum(r['change'] == 0 for r in rows)
                result['N'] = len(rows)
                result[label] = f'{toward} / {away} / {unchanged}'
    latex = []
    for r in records:
        cells = list(r.values())
        if r['A_behavior'] == 'Total':
            latex.append(r'\midrule')
            cells = [r'\textbf{' + str(c) + '}' for c in cells]
        latex.append(tex_row(cells))
    write_csv(output / 'matched_transitions.csv', transitions)
    emit('Table_B3_5', records, inputs, output, latex)

def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    return re.sub(r"\s+", " ", text.replace("-", " ")).strip()

def values(article: int, component: str, text: str) -> set[str]:
    """Extract the contrast-bearing claims, allowing reviewed paraphrases.

    Multiple opposing values remain multiple values and fail comparison.
    Article 1 negated milestone conditions are not positive milestone claims.
    Actor coding requires identity and the relevant role/affiliation/agency pair.
    """
    t = normalize(text)
    found: set[str] = set()

    def add(label: str, condition: bool) -> None:
        if condition:
            found.add(label)

    if component == "MAGNITUDE":
        if article == 1:
            add("USD_1500000000", bool(re.search(r"\$\s*1\.5\s+billion\b", t)))
            add("USD_150000000", bool(re.search(r"\$\s*150\s+million\b", t)))
        elif article == 2:
            add("7_stars", bool(re.search(r"\b(?:seven|7)\s+stars\b", t)))
            add("70_stars", bool(re.search(r"\b(?:seventy|70)\s+stars\b", t)))
        elif article == 3:
            add("600_gigatons", bool(re.search(r"\b600\s+gigatons\b", t)))
            add("6000_gigatons", bool(re.search(r"\b6000\s+gigatons\b", t)))
        elif article == 4:
            # Preserve both units when both are stated. An inconsistent conversion
            # will therefore fail, even if the other speed is correct.
            for num, unit in re.findall(r"\b(\d+)\s+(mph|kph|miles per hour|kilometers per hour)\b", t):
                suffix = "mph" if unit in ("mph", "miles per hour") else "kph"
                found.add(f"{num}_{suffix}")
        elif article == 5:
            for num in re.findall(r"(?<![\d.])(0\.4|4)\s+miles?\b", t):
                found.add(f"{num}_miles")

    elif component == "ACTOR":
        if article == 1:
            add("Brian_Stone_NSF_director_duties", "brian stone" in t and "duties of the nsf director" in t)
            add("Erwin_Gianchandani_NSF_TIP", "erwin gianchandani" in t and "nsf assistant director" in t
                and ("technology, innovation and partnerships" in t or "nsf tip" in t))
        elif article == 2:
            add("Andrea_Bernardi_UDP", "andrea bernardi" in t and "universidad diego portales" in t)
            add("Lucas_Cieza_IEA", "lucas cieza" in t and "instituto de estudios astrofisicos" in t)
        elif article == 3:
            add("ECMWF", "european centre for medium range weather forecasts" in t or "ecmwf" in t)
            add("NOAA", "national oceanic and atmospheric administration" in t or "noaa" in t)
        elif article == 4:
            add("NASA_JAXA", "nasa" in t and "jaxa" in t)
            add("ESA_JAXA", bool(re.search(r"\besa\b", t)) and "jaxa" in t)
        elif article == 5:
            add("Intuitive_Machines", "intuitive machines" in t)
            add("Johns_Hopkins_APL", "johns hopkins applied physics laboratory" in t)

    elif component == "MECHANISM":
        if article == 1:
            add("time_based", "time based" in t or "fixed schedule" in t or "fixed timetable" in t)
            add("milestone_based", bool(re.search(r"milestone (?:based|dependent)", t))
                or "milestones being met" in t or "achievement of research milestones" in t)
        elif article == 2:
            add("amplify_host_star_light", "amplif" in t and "host" in t and "light" in t)
            add("block_host_star_light", "block" in t and "host" in t and "light" in t)
        elif article == 3:
            add("station_fixed", "station" in t and ("fixed" in t or "does not change" in t))
            add("station_moves_with_load_crust", "station" in t and
                ("mov" in t or "displac" in t) and ("crust" in t or "weight" in t))
        elif article == 4:
            add("insufficient_angular_momentum", "angular momentum" in t and
                ("insufficient" in t or "too little" in t or "lacked sufficient" in t))
            add("excessive_angular_momentum", "angular momentum" in t and
                ("too much" in t or "excessive" in t))
        elif article == 5:
            add("compaction_increased_density", "compact" in t and "regolith" in t and
                ("more dense" in t or "increas" in t or "higher density" in t))
            add("loosening_decreased_density", ("fluff" in t or "loosen" in t) and "regolith" in t and
                ("less dense" in t or "decreas" in t or "reduc" in t))
    else:
        raise ValueError(component)
    return found

RUNNERS = {'Table_B3_5': table_b34}

def main():
    package = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, default=Path('YOUR PATHWAY')/'Final Output'/'inputs')
    parser.add_argument('--output', type=Path, default=package/'results'/Path(__file__).stem)
    parser.add_argument('--only', nargs='+', choices=list(RUNNERS), help='Optional selection of exhibits')
    args = parser.parse_args()
    inputs, output = args.inputs.resolve(), args.output.resolve()
    require(inputs.is_dir(), f'Inputs not found: {inputs}')
    require(output != inputs and inputs not in output.parents, 'Outputs must be outside the protected inputs')
    output.mkdir(parents=True, exist_ok=True)
    data, checks = load_data(inputs)
    write_json(output/'source_audit.json', checks)
    write_csv(output/'outcome_audit.csv', data)
    for exhibit in args.only or RUNNERS:
        destination = output/exhibit
        destination.mkdir(exist_ok=True)
        RUNNERS[exhibit](data, inputs, destination)
        print(f'{exhibit}: calculated and saved to {destination}', flush=True)
    write_json(output/'run_metadata.json', dict(python=platform.python_version(), numpy=np.__version__,
        scipy=__import__('scipy').__version__, openpyxl=openpyxl.__version__, matplotlib=matplotlib.__version__,
        source_decisions=len(data), experiments=6, decisions_per_experiment=200,
        exhibits=list(args.only or RUNNERS), overleaf_modified=False))

if __name__ == '__main__':
    main()
