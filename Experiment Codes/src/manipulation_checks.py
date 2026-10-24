"""Optional one-shot unaided recall checks, separate from manager decisions."""
from pathlib import Path
import json
from workbench_protocol import ROOT, build_cases, FrozenTokenizer, write_json, write_json_once, seed_for, bind_run_sources
from workbench_runtime import run_parallel, require_execution_permission, validate_run_id

def prepare_checks(root=ROOT):
    tokenizer=FrozenTokenizer(root)
    cases=build_cases(1,root)
    jobs=[]
    for case in cases:
        content='Based on your own knowledge and beliefs, answer every requested part. State your answer concisely.\n\n'+case['question']
        text,ids=tokenizer.prepare(content)
        jobs.append({'row_id':case['row_id'],'model':case['agent_instance']['adapter_alias'],
                     'serialized':text,'input_ids':ids,'seed':seed_for('unaided:'+case['row_id']),
                     'temperature':.6,'max_tokens':160,'kind':'unaided_manipulation_check',
                     'stop_token_ids':json.loads((Path(root)/'config/model.json').read_text())['stop_token_ids']})
    return jobs

async def run_checks(run_id, *, allow_inference=False, root=ROOT):
    require_execution_permission(allow_inference);validate_run_id(run_id)
    folder=Path(root)/'Manipulation_Checks/runs'/run_id
    root=Path(root)
    sources=list((root.parent/'src').glob('*.py'))+list((root.parents[1]/'Parallel Agent Setup/src').glob('*.py'))
    sources += [root/'config/model.json',root/'config/adapter_registry.json',root/'inputs/frozen_sessions.json']
    sources += list(root.glob('Attempt_*/run_attempt_*.py'))
    bind_run_sources(folder,sources,root.parents[1])
    jobs=prepare_checks(root)
    for job in jobs:job['call_id']=run_id+'_'+job['row_id']
    write_json_once(folder/'inputs/jobs.json',jobs)
    timing,records=await run_parallel(jobs,run_id,folder,allow_inference=True,root=root)
    # Human semantic scoring preserves the original response; it never overwrites an answer.
    rubric=json.loads((Path(root).parents[1]/'DATA/assigned_belief_answer_key.json').read_text())
    keys={(x['article_id'],x['manager'],x['component']):x for x in rubric}
    cases={x['row_id']:x for x in build_cases(1,root)}
    byid={r['call_id']:r for r in records};review=[]
    for job in jobs:
        c=cases[job['row_id']];r=byid[job['call_id']]
        review.append({'row_id':c['row_id'],'article_id':c['article_id'],'manager':c['manager'],
            'question':c['question'],'response':r['output_text'],
            'expected_answers':{part:keys[(c['article_id'],c['manager'],part)]['answer'] for part in c['components']},
            'pass_0_or_1':None,'reviewer':'','review_reason':'',
            'input_tokens':r['input_tokens'],'output_tokens':r.get('output_tokens')})
    write_json(folder/'scoring_required.json',review)
    return review
