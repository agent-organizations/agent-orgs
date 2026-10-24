"""One immutable multi-LoRA server, bounded concurrency, exact per-call audits."""
from __future__ import annotations
from pathlib import Path
import datetime as dt
import fcntl
import hashlib
import json
import os
import re
import shutil

from workbench_protocol import ROOT, read_json, write_json
from transport_core import run_batch, validate_job

ENDPOINT = 'http://127.0.0.1:8000'
LOCK_PATH = Path(__file__).resolve().parents[2] / '.runtime' / 'api_127_0_0_1_8000.lock'


def require_execution_permission(allow_inference):
    if allow_inference is not True:
        raise PermissionError('Prepared only. Set RUN_INFERENCE=True only after user approval to execute.')


def registry(root=ROOT):
    return {a['alias']: a for a in read_json(Path(root)/'config/adapter_registry.json')}


def choose_admission(jobs, root=ROOT):
    if not jobs:
        raise ValueError('No independent ready requests')
    plan = read_json(Path(root)/'config/recommended_runtime.json')
    aliases = registry(root)
    for job in jobs:
        validate_job(job, aliases, plan['max_context_tokens_including_output'])
    maximum = max(len(j['input_ids']) + j['max_tokens'] for j in jobs)
    if all(len(j['input_ids']) <= 584 and j['max_tokens'] <= 128 for j in jobs):
        profile_cap, basis = 32, 'validated short-context capacity'
    elif all(len(j['input_ids']) <= plan['manager_validated_input_tokens'] and j['max_tokens'] <= 600 for j in jobs):
        profile_cap, basis = plan['manager_validated_concurrency'], 'validated manager profile'
    elif all(len(j['input_ids']) <= plan['v8_validated_input_tokens'] and j['max_tokens'] <= 384 for j in jobs):
        profile_cap, basis = plan['v8_validated_concurrency'], 'validated V8 profile'
    elif all(j['max_tokens'] <= 600 for j in jobs):
        profile_cap, basis = plan['near_context_limit_validated_concurrency'], 'conservative long-context profile'
    else:
        raise ValueError('Output budget is outside the validated capacity profiles')
    cap = min(len(jobs), profile_cap, plan['server_max_sequences'],
              plan['conservative_total_reserved_tokens'] // maximum)
    if cap < 1:
        raise ValueError('No request fits the token reservation budget')
    cap = min(cap, plan['client_default_concurrency'])
    return {'concurrency': cap, 'maximum_reserved_per_request': maximum,
            'total_reserved_at_cap': cap * maximum, 'profile_cap': profile_cap, 'basis': basis}


def validate_run_id(run_id):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,70}', run_id):
        raise ValueError('Use a unique run ID with only letters, digits, underscore or hyphen')


def initialize_run(folder, root=ROOT):
    """Copy small immutable runtime metadata; never copy model weights."""
    folder, root = Path(folder), Path(root)
    for d in ['config','logs','qa','data','inputs']:
        (folder/d).mkdir(parents=True, exist_ok=True)
    for name in ['server_process.json','requirements.lock.txt','base_model_manifest.json',
                 'adapter_registry.json','recommended_runtime.json']:
        source = (root/'runtime/server_process.json') if name == 'server_process.json' else root/'config'/name
        target = folder/'config'/name
        if target.exists() and target.read_bytes() != source.read_bytes():
            raise ValueError('Cannot resume a run under changed runtime metadata: ' + name)
        if not target.exists():
            shutil.copy2(source, target)


async def preflight_endpoint(root=ROOT):
    """Read-only health check, called only inside the approved execution path."""
    import httpx
    async with httpx.AsyncClient(timeout=20, trust_env=False) as client:
        health = await client.get(ENDPOINT + '/health')
        health.raise_for_status()
        response = await client.get(ENDPOINT + '/v1/models')
        response.raise_for_status()
        models = response.json()
    available = {x['id'] for x in models['data']}
    expected = set(registry(root))
    if not expected <= available:
        raise RuntimeError('Required adapters are not all registered: ' + repr(expected-available))
    return models


async def run_parallel(jobs, run_name, run_root, *, allow_inference=False, root=ROOT):
    """The only inference gateway; calls inside one batch are independent.

    Existing completed calls are resumed from their exact immutable records.
    Incomplete or changed calls fail loudly; no outcome-based retries exist.
    """
    require_execution_permission(allow_inference)
    validate_run_id(run_name)
    admission = choose_admission(jobs, root)
    initialize_run(run_root, root)
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open('a+') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another coordinator owns the GPU endpoint; wait for that batch to finish') from error
        try:
            lock.seek(0)
            lock.truncate()
            json.dump({'pid':os.getpid(),'run_name':run_name,'utc':dt.datetime.now(dt.timezone.utc).isoformat()}, lock)
            lock.flush()
            models = await preflight_endpoint(root)
            write_json(Path(run_root)/'qa/endpoint_models.json', models)
            write_json(Path(run_root)/'qa'/f'{run_name}_admission.json', admission)
            print('PARALLEL_ADMISSION', json.dumps(admission), flush=True)
            summary, records = await run_batch(jobs, admission['concurrency'], run_root,
                                               run_name, registry(root), ENDPOINT)
            # The batch wall clock includes the slowest call, not the sum of
            # overlapping calls. Cached calls are explicitly identified.
            for record in records:
                print('INSTANCE_RUNTIME', record['call_id'],
                    'INPUT_TOKENS', record['input_tokens'],
                    'OUTPUT_TOKENS', record.get('output_tokens', 'unavailable'),
                    'QUEUE_SECONDS', round(record.get('client_queue_seconds', 0), 6),
                    'GENERATION_SECONDS', round(record.get('elapsed_seconds', 0), 6),
                    'TOTAL_SECONDS', round(record.get('total_seconds', 0), 6),
                    'CACHED', bool(record.get('resumed_from_disk')), flush=True)
            if summary.get('resumed_without_generation'):
                print('HISTORICAL_BATCH_WALL_CLOCK_SECONDS', summary['historical_batch_elapsed_seconds'], flush=True)
            else:
                print('BATCH_WALL_CLOCK_SECONDS', summary['elapsed_seconds'], flush=True)
            print('CURRENT_INVOCATION_SECONDS', summary['current_invocation_seconds'], flush=True)
            return summary, records
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
