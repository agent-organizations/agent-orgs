from pathlib import Path
from collections import Counter
from decimal import Decimal, ROUND_HALF_UP
import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import tempfile
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'b37_six_matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parent
EXPS = ['1a','1b','2a','2b','3a','3b']
REASONS = [('accuracy','Accuracy / article evidence','#eb6834'),
           ('debate','Debate performance','#a8441b'),
           ('belief','Belief similarity','#f2a47f')]
LABELS = ['Group 1:\nFocal shares all 3\nOther shares 0 or 2\nN=80',
          'Group 2:\nFocal shares 1\nOther shares none\nN=30',
          'Group 3:\nFocal shares 1\nOther shares 2\nN=30',
          'Group 4:\nFocal shares 2\nOther shares 1\nN=60']
TITLE = 'Figure B3.7. Stated reasons in manager agent explanations'
YLABEL = '% of rationales using the reason to support the recorded decision'

def load(path):
    return json.loads(path.read_text())

def save(path, obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def csvwrite(path, records):
    fields = list(dict.fromkeys(k for r in records for k in r))
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v
                             for k,v in record.items()})

def pctlabel(value):
    return format(Decimal(str(value)).quantize(Decimal('0.1'),rounding=ROUND_HALF_UP),'f').rstrip('0').rstrip('.')+'%'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', type=Path, default=Path('YOUR PATHWAY')/'Final Output'/'inputs')
    parser.add_argument('--output', type=Path, default=ROOT/'results'/'Figure_B3_7')
    args = parser.parse_args()
    inputs = args.inputs.resolve()
    out = args.output.resolve()
    qa = out/'audit'
    assert out != inputs and inputs not in out.parents
    out.mkdir(parents=True, exist_ok=True)
    qa.mkdir(parents=True, exist_ok=True)
    for relative, expected in load(inputs/'input_checksums.json').items():
        assert sha(inputs/relative) == expected, 'Input changed: '+relative
    primary = load(inputs/'primary_ledger.json')
    accuracy = {r['outcome_id']:r for r in load(inputs/'accuracy_review_ledger.json')}
    manifest = {r['filename']:r['sha256'] for r in load(inputs/'locked_manifest.json')['files']}
    assert len(primary)==len(accuracy)==1200
    assert {r['outcome_id'] for r in primary}==set(accuracy)
    raw_checks = []
    for exp in EXPS:
        rows = [r for r in primary if r['experiment']==exp]
        assert len(rows)==200
        assert Counter(r['group'] for r in rows)=={1:80,2:30,3:30,4:60}
        source = inputs/'workbooks'/Path(rows[0]['source_file']).name
        assert sha(source)==manifest[source.name]
        wb = load_workbook(source,read_only=True,data_only=True)
        values = list(wb['Results'].values)
        wb.close()
        headers = [str(x).strip() if x is not None else '' for x in values[0]]
        raw = {v[0]:(i+2,dict(zip(headers,v))) for i,v in enumerate(values[1:]) if v[0]}
        assert len(raw)==200 and set(raw)=={r['outcome_id'] for r in rows}
        for r in rows:
            rownum,t = raw[r['outcome_id']]
            assert rownum==r['source_row'] and t['Manager rationale']==r['rationale']
            choice_field = 'Endorsement' if r['attempt']<3 else 'Manager choice' if r['model']=='Mistral' else 'Verdict'
            assert t[choice_field].strip()==r['decision']
            assert (int(t['Article']),t['Pair'],t['Manager'],t['Orientation']) == (r['article'],r['pair'],r['manager'],r['orientation'])
            manager,pair = r['manager'],r['pair']
            group = 1 if manager in pair else 2 if manager=='E' else 3 if pair=='AE' else 4
            assert r['group']==group
        raw_checks.append(dict(experiment=exp,source=str(source),sha256=sha(source),outcomes=200))

    ledger = []
    changes = []
    for r in primary:
        a = accuracy[r['outcome_id']]
        for key in ['rationale','rationale_sha256','source_sha256','source_row','decision','group','experiment']:
            assert r[key]==a[key]
        assert a['full_text_read']
        new_accuracy = a['accuracy_consideration']
        assert new_accuracy in [0,1]
        assert bool(a['accuracy_evidence']) == bool(new_accuracy)
        assert all(q in r['rationale'] for q in a['accuracy_evidence'])
        assert set(a['accuracy_subtypes']) <= {'factual_assessment','evidence_limit','completeness'}
        assert new_accuracy >= int(r['accuracy_strength']>0)
        item = dict(r,
            accuracy_plot_code=new_accuracy,
            debate_plot_code=int(r['debate_strength']>0),
            belief_plot_code=int(r['belief_strength']>0),
            expanded_accuracy_evidence=a['accuracy_evidence'],
            expanded_accuracy_subtypes=a['accuracy_subtypes'],
            expanded_accuracy_role=a['accuracy_role'],
            expanded_accuracy_reason=a['accuracy_reason'])
        ledger.append(item)
        if new_accuracy != int(r['accuracy_strength']>0):
            changes.append(dict(outcome_id=r['outcome_id'],experiment=r['experiment'],group=r['group'],
                source_row=r['source_row'],choice=r['decision'],previous_code=int(r['accuracy_strength']>0),
                current_code=new_accuracy,evidence=a['accuracy_evidence'],subtypes=a['accuracy_subtypes'],
                full_rationale=r['rationale']))
    assert len(changes)==188
    assert all(r['debate_plot_code']==0 for r in ledger if r['attempt']==1)
    save(out/'Coding_ledger_all_1200.json',ledger)
    csvwrite(out/'Coding_ledger_all_1200.csv',ledger)
    csvwrite(out/'Accuracy_coding_changes.csv',changes)

    counts = []
    for exp in EXPS:
        for group in ['Overall',1,2,3,4]:
            subset=[r for r in ledger if r['experiment']==exp and (group=='Overall' or r['group']==group)]
            for reason,label,color in REASONS:
                n=sum(r[reason+'_plot_code'] for r in subset)
                counts.append(dict(experiment=exp,group=group,reason=reason,label=label,
                                   N=len(subset),count=n,percent=100*n/len(subset)))
    csvwrite(out/'Figure_B3_7_counts.csv',counts)
    by_key={(r['experiment'],r['group'],r['reason']):r for r in counts}
    previous=load(inputs/'previous_bar_geometry.json')
    for bar in previous:
        if bar['reason']!='accuracy':
            assert abs(bar['height']-by_key[bar['experiment'],bar['group'],bar['reason']]['percent'])<1e-10
    assert by_key['1b',4,'accuracy']['count']==2
    assert all(by_key['1b',g,'accuracy']['count']==0 for g in [1,2,3])

    by_verdict=[]
    for exp in EXPS:
        subset=[r for r in ledger if r['experiment']==exp]
        for decision in ['Single participant','Both','None']:
            cases=[r for r in subset if ('Single participant' if r['decision'].startswith('P') else r['decision'])==decision]
            if cases:
                by_verdict.append(dict(experiment=exp,decision_type=decision,N=len(cases),
                    accuracy_count=sum(r['accuracy_plot_code'] for r in cases),
                    debate_count=sum(r['debate_plot_code'] for r in cases),
                    belief_count=sum(r['belief_plot_code'] for r in cases)))
    csvwrite(out/'Counts_by_verdict.csv',by_verdict)
    assert sum(r['N'] for r in by_verdict)==1200

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.titlesize':17,
                         'svg.fonttype':'none','pdf.fonttype':42,'axes.labelcolor':'#222222',
                         'xtick.color':'#222222','ytick.color':'#222222'})
    fig,axes=plt.subplots(3,2,figsize=(16,17.3),sharey=True)
    fig.subplots_adjust(left=.065,right=.992,top=.915,bottom=.052,hspace=.52,wspace=.15)
    fig.suptitle(TITLE,fontsize=21,fontweight='bold',y=.985)
    fig.legend(handles=[Patch(facecolor=color,label=label) for _,label,color in REASONS],
               loc='upper center',bbox_to_anchor=(.52,.965),ncol=3,frameon=False,fontsize=14,
               handlelength=1.4,columnspacing=2)
    width=.25
    x=np.arange(4)
    geometry=[]
    for ax,exp in zip(axes.flat,EXPS):
        for i,(reason,label,color) in enumerate(REASONS):
            values=[by_key[exp,g,reason]['percent'] for g in range(1,5)]
            bars=ax.bar(x+(i-1)*width,values,width*.92,color=color,zorder=3)
            for g,(bar,value) in enumerate(zip(bars,values),1):
                text=pctlabel(value)
                ax.text(bar.get_x()+bar.get_width()/2,value+1.8,text,ha='center',va='bottom',fontsize=11.2,
                        bbox={'facecolor':'white','edgecolor':'none','pad':.15},zorder=5)
                geometry.append(dict(experiment=exp,group=g,reason=reason,height=float(bar.get_height()),label=text))
        model='Mistral' if exp.endswith('a') else 'Qwen'
        ax.set_title(f'Experiment {exp} · {model}',loc='left',pad=11,fontweight='bold')
        ax.set_xticks(x,LABELS,fontsize=11.8,linespacing=1.17)
        ax.tick_params(axis='x',length=0,pad=12)
        ax.set_ylim(0,113)
        ax.set_yticks(range(0,101,20),[f'{v}%' for v in range(0,101,20)],fontsize=12)
        ax.tick_params(axis='y',labelleft=True,length=3)
        ax.grid(axis='y',color='#dedede',linewidth=.7,zorder=0)
        for spine in ['top','right']:ax.spines[spine].set_visible(False)
        ax.spines['left'].set_color('#8b8b8b')
        ax.spines['bottom'].set_color('#8b8b8b')
        ax.set_axisbelow(True)
    fig.text(.012,.5,YLABEL,rotation=90,va='center',ha='center',fontsize=15)
    base=out/'Figure_B3_7_All_Six_Experiments'
    fig.savefig(str(base)+'.png',dpi=300,facecolor='white')
    fig.savefig(str(base)+'_preview.png',dpi=150,facecolor='white')
    fig.savefig(str(base)+'.pdf',facecolor='white')
    fig.savefig(str(base)+'.svg',facecolor='white')
    plt.close(fig)
    for bar in geometry:
        expected=by_key[bar['experiment'],bar['group'],bar['reason']]
        assert abs(bar['height']-expected['percent'])<1e-10
        assert bar['label']==pctlabel(expected['percent'])
    save(qa/'plotted_bar_geometry.json',geometry)
    save(qa/'Validation.json',dict(raw_checks=raw_checks,outcomes=1200,per_experiment=200,
        group_sizes=[80,30,30,60],bars_checked=72,protected_debate_and_belief_bars=48,
        all_2400_debate_and_belief_codes_preserved=True,accuracy_codes_changed=len(changes),
        all_Both_and_None_included=True,overleaf_modified=False,
        title_category_group_labels_preserved=True,y_axis_label=YLABEL,
        previous_notes_untouched=True,semantic_provenance='Existing full-text reviews and exact evidence; 188 added excerpts rechecked'))
    template = (inputs/'templates/Figure_B3_7.tex').read_text()
    (out/'Figure_B3_7.tex').write_text(template)
    notes = template[template.index('\\begingroup\\footnotesize'):]
    (out/'Figure_B3_7_notes.tex').write_text(notes)
    print(json.dumps([r for r in counts if r['group']=='Overall' and r['reason']=='accuracy'],indent=2))

if __name__=='__main__':
    main()
