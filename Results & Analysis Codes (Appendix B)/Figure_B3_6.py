#!/usr/bin/env python3


"""Reproduce Figure B3 6 from the six locked workbooks.

Run: python Figure_B3_6.py --inputs "YOUR PATHWAY/Final Output/inputs"
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

def figure_b35(data, inputs, output):
    """Partner support (including Both), across five exhaustive groups."""
    records = []
    for exp in MODEL_ORDER:
        for group, label in enumerate(PARTNER_GROUPS):
            rows = [r for r in select(data, exp) if r['partner_group'] == group]
            k, n = sum(r['supports_partner'] for r in rows), len(rows)
            records.append(dict(model=rows[0]['model'], attempt=rows[0]['attempt'], experiment=exp,
                group_index=group, group=label, n=n, supported_partner=k,
                partner_only=sum(r['partner_only'] for r in rows), both=sum(r['both'] for r in rows),
                percent=100*k/n))
    emit('Figure_B3_6', records, inputs, output)
    plot_b35(records, output)

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

def plot_b35(records, output):
    """Preserve the approved chart type, labels, palette and layout."""
    plt.rcdefaults()
    data = records
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'svg.fonttype':'none','pdf.fonttype':42,'axes.unicode_minus':False})
    fig,axes=plt.subplots(1,2,figsize=(12.6,8.5),sharey=True)
    fig.subplots_adjust(left=.315,right=.985,bottom=.12,top=.89,wspace=.095)
    y=np.arange(5)*1.18;offsets={1:-.245,2:0,3:.245};bindings=[]
    for ax,model,suffix in zip(axes,['Mistral','Qwen'],['a','b']):
        for attempt in (1,2,3):
            rows=sorted([r for r in data if r['model']==model and r['attempt']==attempt],key=lambda r:r['group_index']);assert len(rows)==5
            widths=np.array([r['percent'] for r in rows]);bars=ax.barh(y+offsets[attempt],widths,height=.18,color=COLORS[attempt],edgecolor=COLORS[attempt],linewidth=.7,zorder=3)
            for bar,r in zip(bars,rows):
                assert abs(bar.get_width()-100*r['supported_partner']/r['n'])<1e-10
                yy=bar.get_y()+bar.get_height()/2
                if r['supported_partner']==0:ax.plot(0,yy,marker='|',markersize=9,color=COLORS[attempt],markeredgewidth=2,zorder=4,clip_on=False)
                label=f"{r['supported_partner']}/{r['n']}"
                ax.text(r['percent']+2.1,yy,label,va='center',ha='left',fontsize=12.5,color='#222222')
                bindings.append(dict(experiment=r['experiment'],group_index=r['group_index'],count_label=label,width_percent=float(bar.get_width()),expected_percent=r['percent'],color=COLORS[attempt]))
        ax.set_xlim(0,124);ax.set_xticks([0,25,50,75,100],[f'{v}%' for v in [0,25,50,75,100]],fontsize=12)
        ax.set_yticks(y);ax.set_axisbelow(True);ax.grid(axis='x',color='#E9E9E9',linewidth=.7);ax.grid(axis='y',visible=False)
        for boundary in (y[:-1]+y[1:])/2:ax.axhline(boundary,color='#ECECEC',linewidth=.6,zorder=0)
        for side in ['top','right','left']:ax.spines[side].set_visible(False)
        ax.spines['bottom'].set_color('#B8B8B8');ax.spines['bottom'].set_linewidth(.8)
        ax.tick_params(axis='y',length=0,pad=12);ax.tick_params(axis='x',length=3,color='#888888',pad=6)
        ax.set_title(f'{model} managers',fontsize=17,pad=17)
        ax.set_xlabel('Support for A’s partner (%)',fontsize=13.5,labelpad=12)
        handles=[Patch(facecolor=COLORS[a],edgecolor=COLORS[a],label=f'Exp. {a}{suffix}') for a in (1,2,3)]
        ax.legend(handles=handles,ncol=3,frameon=False,fontsize=12,handlelength=.9,handletextpad=.4,columnspacing=.85,loc='upper left',bbox_to_anchor=(-.01,1.025),borderaxespad=0)
    axes[0].set_yticklabels(LABELS,fontsize=14);axes[0].set_ylim(y[-1]+.60,-.60)
    fig.canvas.draw();renderer=fig.canvas.get_renderer();extent=fig.bbox
    for ax in axes:
        for artist in ax.texts+ax.get_yticklabels()+ax.get_xticklabels()+[ax.title,ax.xaxis.label]:
            if not artist.get_visible():continue
            box=artist.get_window_extent(renderer)
            assert box.x0>=-1 and box.y0>=-1 and box.x1<=extent.width+1 and box.y1<=extent.height+1,artist.get_text()
    png=output/'Figure_B3_6.png';fig.savefig(png,dpi=300,facecolor='white')
    fig.savefig(output/'Figure_B3_6_preview.png',dpi=145,facecolor='white');plt.close(fig)
    write_json(output / 'plotted_values.json', bindings)

RUNNERS = {'Figure_B3_6': figure_b35}

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
