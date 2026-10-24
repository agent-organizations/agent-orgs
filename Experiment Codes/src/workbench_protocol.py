"""Pure input construction: no network, GPU, model generation, or prior responses."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
import random
import re
import uuid

ROOT = Path(__file__).resolve().parents[1] / "Mistral"
TRANSCRIPT_INTRO = 'The following is a conversation between two other participants. You did not take part in it.'
OWN_ANSWER_INSTRUCTION = 'Before reviewing, state your own answer to the question.'
NEUTRAL = {'P1': 'P1', 'P2': 'P2'}


def read_json(path):
    return json.loads(Path(path).read_text())


def sha_text(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.part')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def write_json_once(path, value):
    """Create immutable run provenance, or require an identical existing file."""
    path = Path(path)
    data = json.dumps(value, ensure_ascii=False, indent=2) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open('x') as file:
            file.write(data)
    except FileExistsError:
        if path.read_text() != data:
            raise ValueError('Existing run provenance differs; use a new approved run ID: ' + str(path))


def bind_run_sources(run_root, source_paths, root=ROOT):
    root = Path(root)
    files = {str(Path(path).relative_to(root)): hashlib.sha256(Path(path).read_bytes()).hexdigest()
             for path in sorted(map(Path, source_paths))}
    write_json_once(Path(run_root)/'inputs/source_hashes.json', files)
    return files


def seed_for(identifier, base=20261018):
    return (base + int(sha_text(identifier)[:8], 16)) % 2**32


@dataclass(frozen=True)
class AgentInstance:
    """A distinct request context sharing read-only base/adapter tensors.

    There is no mutable conversation, optimizer, or cross-row cache here. Sharing
    a frozen adapter does not share a conversation or update anyone's weights.
    """
    instance_id: str
    adapter_alias: str
    seed: int
    history: tuple = ()


def spawn_instance(key, adapter_alias):
    return AgentInstance(str(uuid.uuid5(uuid.NAMESPACE_URL, 'parallel-workbench:' + key)),
                         adapter_alias, seed_for(key))


def spawn_original_agents(run_id='v8_prepared'):
    return {a: spawn_instance(run_id + ':original:' + a, 'v8_' + a.lower()) for a in 'ABCDE'}


def neutralize(text, actual_to_neutral):
    """Rename explicit speakers once, never by chained replacements.

    Do not replace bare single letters: 'A' may be a word, scientific label or
    article content. Frozen V8 speaker references use 'Agent A', etc.
    """
    def replace(match):
        actual = match.group(1)
        if actual not in actual_to_neutral:
            raise ValueError('Transcript references an agent outside the pair: ' + actual)
        return actual_to_neutral[actual]
    return re.sub(r'\bAgent\s+([ABCDE])\b', replace, text)


def build_cases(attempt, root=ROOT):
    """100 article/pair/manager cases x two separately generated orientations."""
    root = Path(root)
    attempt = f'{int(attempt):02d}'
    if attempt not in ('01', '02', '03'):
        raise ValueError('Unknown attempt')
    baseline = (root/f'Attempt_{attempt}/prompts/Prompt_Verbatim.txt').read_text()
    model_config = read_json(root/'config/model.json')
    sessions = read_json(root/'inputs/frozen_sessions.json')
    differences = {(d['article_id'], d['manager'], d['participant']): d
                   for d in read_json(root/'inputs/belief_differences.json')}
    cases = []
    for session in sessions:
        for manager in 'ABCDE':
            base = f'{session["session_id"]}_manager_{manager}'
            for orientation in ['original', 'swapped']:
                pair = session['agents']
                actuals = pair if orientation == 'original' else list(reversed(pair))
                labels = dict(zip(['P1','P2'], actuals))
                actual_to_neutral = {actual: NEUTRAL[p] for p, actual in labels.items()}
                row_id = f'{model_config["family"].lower()}_a{attempt}_{base}_{orientation}'
                instance = spawn_instance(row_id, model_config['manager_alias_prefix'] + manager.lower())
                difference_texts = {
                    p: differences[(session['article_id'], manager, actual)]['text'].format(participant=NEUTRAL[p])
                    for p, actual in labels.items()}
                messages = [{**m, 'speaker': actual_to_neutral[m['actual_agent']],
                             'text': neutralize(m['text'], actual_to_neutral)} for m in session['messages']]
                finals = {p: session['final_message_ids'][actual] for p, actual in labels.items()}
                allowed = ['P1', 'P2'] if attempt != '03' else (
                    ['P1','P2','None'] if session['source_outcome'] == 'Unresolved'
                    else ['P1','P2','Both','None'])
                # Only substitute the explicitly requested manager placeholder.
                # Its neutral role name does not reveal the participant's letter.
                prompt = baseline.replace('{manager}', 'manager')
                sections = [prompt, OWN_ANSWER_INSTRUCTION,
                    'FACTUAL QUESTION\n' + session['question'],
                    'PARTICIPANT IDENTITIES FOR THIS EVALUATION\nP1 and P2 identify the two participants consistently in every supplied material.',
                    'BELIEF DIFFERENCES RELATIVE TO MY LEARNED BELIEFS\n' +
                    '\n'.join(f'Beliefs of {NEUTRAL[p]} ({p}): {difference_texts[p]}' for p in ['P1','P2']),
                    'FINAL MESSAGE IDS\n' + '\n'.join(f'{p} ({NEUTRAL[p]}): {finals[p]}' for p in ['P1','P2'])]
                if attempt in ['02','03']:
                    transcript = '\n\n'.join(f'{m["id"]} | {m["speaker"]} | {m["stage"]}\n{m["text"]}' for m in messages)
                    sections.append(TRANSCRIPT_INTRO + '\n<OTHER_PARTICIPANTS_CONVERSATION>\n' +
                                    transcript + '\n</OTHER_PARTICIPANTS_CONVERSATION>')
                else:
                    sections.append('No interaction transcript is supplied in this attempt. The final message IDs identify the participants\' final messages; their contents are not supplied.')
                if attempt == '03':
                    sections.append('PERMITTED VERDICTS\n' + ', '.join(allowed))
                content = '\n\n'.join(sections)
                cases.append({'row_id': row_id, 'base_row_id': base, 'attempt': attempt,
                    'article_id': session['article_id'], 'article_title': session['article_title'],
                    'session_id': session['session_id'], 'pair': session['pair'], 'manager': manager,
                    'orientation': orientation, 'labels_to_actual': labels, 'neutral_to_actual':
                        {NEUTRAL[p]: a for p, a in labels.items()},
                    'agent_instance': asdict(instance), 'question': session['question'],
                    'components': session['components'], 'belief_differences': difference_texts,
                    'final_message_ids': finals, 'messages': messages, 'allowed_choices': allowed,
                    'source_outcome_offline_only': session['source_outcome'],
                    'baseline_prompt_sha256': sha_text(baseline), 'content': content,
                    'content_sha256': sha_text(content),
                    'source_session_sha256': session['source_session_sha256'],
                    'own_answer_measurement': 'same-call pre-decision answer; exposed to all input materials'})
    validate_cases(cases)
    return cases


def validate_cases(cases):
    if len(cases) != 200 or len({c['row_id'] for c in cases}) != 200:
        raise ValueError('Exactly 200 independently identified calls are required')
    if len({c['agent_instance']['instance_id'] for c in cases}) != 200:
        raise ValueError('Manager contexts must be unique')
    if len({c['agent_instance']['seed'] for c in cases}) != 200:
        raise ValueError('Request seeds unexpectedly collide')
    grouped = {}
    for c in cases:
        if c['agent_instance']['history'] not in [(), []]:
            raise ValueError('Manager context must start empty')
        grouped.setdefault(c['base_row_id'], []).append(c)
        if OWN_ANSWER_INSTRUCTION not in c['content']:
            raise ValueError('Missing own-answer instruction')
        if re.search(r'\bAgent\s+[ABCDE]\b|Manager_Agent_Pair_|P[12]\s*=\s*[ABCDE]\b', c['content']):
            raise ValueError('Original agent identity leaked into manager prompt')
        if c['attempt'] in ['02','03'] and TRANSCRIPT_INTRO not in c['content']:
            raise ValueError('Missing third-party transcript frame')
        for p, mid in c['final_message_ids'].items():
            matches = [m for m in c['messages'] if m['id'] == mid]
            if len(matches) != 1 or matches[0]['actual_agent'] != c['labels_to_actual'][p]:
                raise ValueError('Final-message attribution mismatch')
    if len(grouped) != 100:
        raise ValueError('Expected 100 matched cases')
    for pair in grouped.values():
        if len(pair) != 2 or {c['orientation'] for c in pair} != {'original','swapped'}:
            raise ValueError('Missing independently requested orientation')
        a, b = sorted(pair, key=lambda c:c['orientation'])
        if a['labels_to_actual']['P1'] != b['labels_to_actual']['P2']:
            raise ValueError('Participant positions were not exchanged')
        if a['labels_to_actual']['P2'] != b['labels_to_actual']['P1']:
            raise ValueError('Participant positions were not exchanged')
    return True


def randomized_execution_order(cases, seed=20261018):
    """Random scheduling order; reporting retains adjacent matched orientations."""
    rows = list(cases)
    random.Random(seed).shuffle(rows)
    return rows


class FrozenTokenizer:
    """CPU-only tokenizer; exact local base vocabulary, no weight download."""
    def __init__(self, root=ROOT):
        from model_tokenization import NativeTokenizer
        self.impl = NativeTokenizer(root)
    def serialize(self, content):
        return self.impl.prepare(content)[0]
    def prepare(self, content):
        return self.impl.prepare(content)


def prepare_inputs(attempt, root=ROOT, save=True):
    """Author 200 exact input strings and token IDs. Never generate responses."""
    root = Path(root)
    cases = build_cases(attempt, root)
    tokenizer = FrozenTokenizer(root)
    config = read_json(root/f'Attempt_{int(attempt):02d}/config.json')
    for c in cases:
        c['serialized'], c['input_ids'] = tokenizer.prepare(c['content'])
        c['input_tokens'] = len(c['input_ids'])
        if c['input_tokens'] + config['max_output_tokens'] > read_json(root/'config/model.json')['context_tokens']:
            raise ValueError('Context overflow; inputs must not be truncated: ' + c['row_id'])
    if save:
        dest = root/f'Attempt_{int(attempt):02d}/inputs'
        write_json(dest/'prepared_cases.json', cases)
        write_json(dest/'all_input_texts.json', [c['serialized'] for c in cases])
        write_json(dest/'all_input_token_ids.json', [c['input_ids'] for c in cases])
        write_json(dest/'row_order.json', [c['row_id'] for c in cases])
        write_json(dest/'execution_order.json', [c['row_id'] for c in randomized_execution_order(cases)])
    return cases
