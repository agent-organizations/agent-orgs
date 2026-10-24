"""Participant Agent Interaction protocol on vLLM: parallel independent pairs, sequential reviewed turns.

Consensus review is explicit and bound to the exact current messages. A manager
attempt never silently consumes these new conversations: its frozen inputs are
separate until a new source set is deliberately registered.
"""
from __future__ import annotations
from pathlib import Path
import ast
import copy
import json

from workbench_protocol import (ROOT, FrozenTokenizer, read_json, write_json, sha_text,
                                seed_for, spawn_instance, write_json_once, bind_run_sources)
from workbench_runtime import run_parallel, require_execution_permission, validate_run_id


def interaction_prompts(root=ROOT):
    root = Path(root)
    path = root/'Participant_Agent_Interaction/prompts/prompt.py'
    tree = ast.parse(path.read_text())
    debate = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id=='DEBATE_SYSTEM' for t in node.targets))
    return (root/'Participant_Agent_Interaction/prompts/initial_system.txt').read_text(), debate


def question_for(plan):
    return 'Regarding ' + plan['row']['context'] + ':\n' + '\n'.join(
        c.capitalize() + ': ' + plan['row']['questions'][c] for c in plan['components'])


def initial_template(parts):
    descriptions = {'magnitude':'number and unit only',
        'actor':'name, plus role or affiliation if the question asks for it',
        'mechanism':'brief description of the requested mechanism'}
    lines = []
    for c in parts:
        lines.extend([c.upper()+': <'+descriptions[c]+'>',
                      'QUOTE_'+c.upper()+': "<sentence copied word for word from YOUR article>"'])
    return '\n\nFill in this template. Replace each <...> and keep every label exactly as written:\n' + '\n'.join(lines) + '\n<END>'


def new_interaction_sessions(run_id, root=ROOT):
    validate_run_id(run_id)
    plans = read_json(Path(root)/'Participant_Agent_Interaction/inputs/session_plan.json')
    sessions = []
    for plan in plans:
        s = copy.deepcopy(plan)
        s.update(question=question_for(plan), history=[], latest={}, status='ready',
                 debate_turns=0, reviews=[], run_id=run_id,
                 agents_instances={a:spawn_instance(run_id+':'+plan['session_id']+':'+a, 'participant_'+a.lower()).__dict__ for a in plan['agents']})
        sessions.append(s)
    return sessions


def interaction_input(session, participant, stage, root=ROOT):
    """Reproduce the original Participant Agent Interaction system/user contents and one-user serialization."""
    root = Path(root)
    agent = session['agents'][participant-1]
    peer = session['agents'][2-participant]
    article = (root/'Participant_Agent_Interaction/inputs/articles'/agent/(session['article_id']+'.txt')).read_text()
    initial, debate = interaction_prompts(root)
    if stage == 'initial':
        system = initial
        user = ('Your article:\n<article>\n'+article+'\n</article>\n\nQuestion:\n'+
                session['question']+initial_template(session['components']))
    else:
        system = debate.format(agent=agent,peer=peer)
        history = '\n\n'.join('Agent '+h['agent']+', '+h['stage']+', turn '+str(h['turn'])+':\n'+h['output'] for h in session['history'])
        user = ('ASSIGNED ARTICLE\n'+article+'\n\nQUESTION\n'+session['question']+
                '\n\nDISCUSSION SO FAR\n'+history+'\n\nRespond only as Agent '+agent+'.')
    return system, user


def review_packet(session):
    return {'session_id':session['session_id'], 'question':session['question'],
            'components':session['components'], 'debate_turns':session['debate_turns'],
            'latest':session['latest'], 'history':session['history'],
            'instruction':'Review whole latest messages without consulting truth. Every requested component must have the same determinate meaning. Shared uncertainty, missing components or contradictory commitments do not establish consensus.'}


def packet_hash(packet):
    return sha_text(json.dumps(packet,ensure_ascii=False,sort_keys=True))


def apply_review(session, review):
    expected = packet_hash(review_packet(session))
    if review['packet_sha256'] != expected:
        raise ValueError('Stale review; current messages do not match')
    determinate, components = review['determinate'], review['component_agreement']
    if len(determinate)!=2 or any(type(v) is not bool for v in determinate):
        raise ValueError('Two determinacy judgments are required')
    if review['components'] != session['components'] or len(components)!=len(session['components']):
        raise ValueError('Every requested component needs a judgment')
    if any(type(v) is not bool for v in components) or type(review['agreement']) is not bool:
        raise ValueError('Use explicit Boolean judgments')
    if review['agreement'] != (all(determinate) and all(components)):
        raise ValueError('Overall agreement conflicts with component assessments')
    if not review.get('rationale','').strip() or not review.get('reviewer','').strip():
        raise ValueError('Record reviewer and substantive reason')
    session['reviews'].append(review)
    if review['agreement']:
        session.update(status='complete', stop_reason='agreement_before_debate' if not session['debate_turns'] else 'substantive_agreement')
    elif session['debate_turns'] >= 20:
        session.update(status='complete', stop_reason='ten_round_limit')
    else:
        session['status'] = 'ready'
    return session


def submit_review(run_id, session_id, *, agreement, determinate, component_agreement,
                  rationale, reviewer, inspected_packet_sha256, reviewed_debate_turn, root=ROOT):
    validate_run_id(run_id)
    folder = Path(root)/'Participant_Agent_Interaction/runs'/run_id
    session = read_json(folder/'data/sessions'/(session_id+'.json'))
    if session['status'] != 'awaiting_review':
        raise ValueError('Session is not waiting for a consensus review')
    packet = review_packet(session)
    if reviewed_debate_turn != session['debate_turns'] or inspected_packet_sha256 != packet_hash(packet):
        raise ValueError('The inspected review packet is stale; inspect the current messages')
    review = {'packet_sha256':inspected_packet_sha256, 'agreement':agreement,
        'determinate':determinate, 'components':session['components'],
        'component_agreement':component_agreement, 'rationale':rationale, 'reviewer':reviewer}
    apply_review(copy.deepcopy(session), review)  # Validate before saving.
    name = session_id+f'_t{session["debate_turns"]:02d}.json'
    path = folder/'data/review_inbox'/name
    if path.exists() and read_json(path) != review:
        raise ValueError('Review already exists; never overwrite a stopping decision')
    write_json(path, review)
    return review


async def advance_interactions(run_id, *, allow_inference=False, root=ROOT):
    """Run one ready wave; return immediately when only semantic reviews remain.

    Both initials are independent jobs. Thereafter each pair contributes at most
    one next turn per wave, and only after its preceding review has been applied.
    """
    require_execution_permission(allow_inference)
    root = Path(root)
    validate_run_id(run_id)
    folder = root/'Participant_Agent_Interaction/runs'/run_id
    sources = list((root/'Participant_Agent_Interaction/inputs').rglob('*.json'))
    sources += list((root/'Participant_Agent_Interaction/inputs/articles').rglob('*.txt'))
    sources += list((root/'Participant_Agent_Interaction/prompts').glob('*'))
    sources += list((root.parent/'src').glob('*.py'))
    sources += list((root.parents[1]/'Parallel Setup/src').glob('*.py'))
    sources += [root/'config/model.json', root/'Participant_Agent_Interaction/run_interactions.py']
    sources += [root/'config/adapter_registry.json']
    bind_run_sources(folder, sources, root.parents[1])
    states = []
    for fresh in new_interaction_sessions(run_id, root):
        path = folder/'data/sessions'/(fresh['session_id']+'.json')
        s = read_json(path) if path.exists() else fresh
        if s['status']=='awaiting_review':
            review = folder/'data/review_inbox'/(s['session_id']+f'_t{s["debate_turns"]:02d}.json')
            if review.exists():
                apply_review(s, read_json(review))
                write_json(path, s)
        states.append(s)
    ready = []
    tokenizer = FrozenTokenizer(root)
    for s in states:
        if s['status']!='ready':
            continue
        if s['history']:
            if s['debate_turns'] >= 20:
                raise ValueError('Turn limit reached without a stopping review')
            turn = s['debate_turns']+1
            participant = s['first_speaker'] if turn%2 else 3-s['first_speaker']
            specs = [(participant,'debate',turn)]
        else:
            specs = [(1,'initial',0),(2,'initial',0)]
        for participant, stage, turn in specs:
            logical = f'participant-{s["session_id"]}-' + (f'initial-{participant}' if stage=='initial' else f'turn{turn:02d}')
            call_id = run_id+'_'+logical.replace('-','_')
            system, user = interaction_input(s, participant, stage, root)
            serialized, ids = tokenizer.prepare(system+'\n\n'+user)
            agent = s['agents'][participant-1]
            job = {'call_id':call_id,'kind':'participant_interaction','model':'participant_'+agent.lower(),
                'input_ids':ids,'serialized':serialized,'temperature':0.9,'max_tokens':384,
                'seed':seed_for(logical,20230930),'stop':['<END>'],'stop_token_ids':[2]}
            ready.append((s,participant,stage,turn,job))
    if not ready:
        return {'new_calls':0,'complete':sum(s['status']=='complete' for s in states),
                'awaiting_review':[s['session_id'] for s in states if s['status']=='awaiting_review']}
    jobs = [x[4] for x in ready]
    wave = 'wave_'+sha_text('|'.join(j['call_id'] for j in jobs))[:12]
    write_json_once(folder/'inputs'/(wave+'_input_texts.json'), [j['serialized'] for j in jobs])
    summary, records = await run_parallel(jobs,wave,folder,allow_inference=True,root=root)
    by_call = {r['call_id']:r for r in records}
    if any(r['status']!='complete' for r in records):
        raise RuntimeError('A Participant Agent Interaction call failed. Preserve all logs and resolve the technical failure; no automatic resampling.')
    touched = {}
    for s,participant,stage,turn,job in ready:
        r = by_call[job['call_id']]
        message = {'call_id':job['call_id'],'agent':s['agents'][participant-1],
            'participant':participant,'stage':stage,'turn':turn,'output':r['output_text'].removesuffix('</s>'),
            'input_tokens':r['input_tokens'],'output_tokens':r['output_tokens'],
            'finish_reason':r['finish_reason'],'runtime_seconds':r['elapsed_seconds']}
        s['history'].append(message)
        s['latest'][str(participant)] = message
        s['debate_turns'] = turn
        touched[s['session_id']] = s
    for sid,s in touched.items():
        s['status']='awaiting_review'
        write_json(folder/'data/sessions'/(sid+'.json'),s)
        packet = review_packet(s)
        write_json_once(folder/'data/review_pending'/(sid+f'_t{s["debate_turns"]:02d}.json'),
                        {**packet, 'packet_sha256':packet_hash(packet)})
    return {'new_calls':len(jobs),'batch':summary,
            'awaiting_review':[s['session_id'] for s in states if s['status']=='awaiting_review'],
            'complete':sum(s['status']=='complete' for s in states)}


def export_interactions(run_id, root=ROOT):
    """Export completed new sessions only; source Participant Agent Interaction data stay untouched."""
    import csv
    root = Path(root)
    validate_run_id(run_id)
    folder = root/'Participant_Agent_Interaction/runs'/run_id
    sessions = [read_json(folder/'data/sessions'/(p['session_id']+'.json'))
                for p in read_json(root/'Participant_Agent_Interaction/inputs/session_plan.json')]
    if any(s['status']!='complete' for s in sessions):
        raise ValueError('All20 sessions must have a stopping review before final export')
    dest = root/'Participant_Agent_Interaction/deliverables'/run_id
    dest.mkdir(parents=True, exist_ok=True)
    write_json(dest/'sessions.json',sessions)
    with (dest/'All_Interaction_Transcripts.txt').open('w') as f:
        for s in sessions:
            f.write(s['session_id']+'\n'+s['question']+'\n\n')
            for h in s['history']:
                f.write(f'{h["agent"]} | {h["stage"]} | turn {h["turn"]}\n{h["output"]}\n\n')
    with (dest/'Participant_Agent_Interaction_Summary.csv').open('w',newline='') as f:
        fields = ['session_id','article_id','debate_turns','stop_reason','input_tokens','output_tokens','source_accuracy']
        w = csv.DictWriter(f,fields)
        w.writeheader()
        for s in sessions:
            w.writerow({**{k:s[k] for k in fields[:4]},
                'input_tokens':sum(h['input_tokens'] for h in s['history']),
                'output_tokens':sum(h['output_tokens'] for h in s['history']),
                'source_accuracy':'requires separate source-based assessment'})
    return dest
